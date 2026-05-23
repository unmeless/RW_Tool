from __future__ import annotations

import threading
from typing import Literal

import numpy as np

from rw_tool.image_preprocess import build_ocr_variants
from rw_tool.ocr_layout import split_horizontal_strips

BackendName = Literal["rapidocr", "easyocr"]
OcrLayout = Literal["auto", "strip"]

# 首遍 OCR 足够好时跳过第二遍（game_ui + dual_preprocess）
_DUAL_SKIP_MIN_CHARS = 12
_DUAL_SKIP_MIN_CONF = 0.52


def _box_sort_key(item) -> tuple[float, float]:
    box = item[0] if item else []
    try:
        ys = [float(p[1]) for p in box]
        xs = [float(p[0]) for p in box]
        return min(ys), min(xs)
    except (TypeError, IndexError, ValueError):
        return 0.0, 0.0


def _sort_ocr_lines(result: list) -> list:
    return sorted(result, key=_box_sort_key)


class OcrEngine:
    """线程安全的 OCR 封装，支持 rapidocr / easyocr，模型懒加载。"""

    def __init__(
        self,
        backend: BackendName = "rapidocr",
        *,
        det_box_thresh: float = 0.35,
        text_score: float = 0.4,
        preprocess_scale: float = 2.0,
        preprocess_mode: str = "game_ui",
        dual_preprocess: bool = False,
        ocr_max_side: int = 1280,
        ocr_layout: str = "strip",
        ocr_strip_max_lines: int = 2,
        use_angle_cls: bool = False,
    ) -> None:
        self._backend = backend
        self._det_box_thresh = det_box_thresh
        self._text_score = text_score
        self._preprocess_scale = preprocess_scale
        self._preprocess_mode = preprocess_mode
        self._dual_preprocess = dual_preprocess
        self._ocr_max_side = ocr_max_side
        self._ocr_layout: OcrLayout = "strip" if ocr_layout == "strip" else "auto"
        self._ocr_strip_max_lines = max(1, min(3, ocr_strip_max_lines))
        self._use_angle_cls = use_angle_cls
        self._engine = None
        self._lock = threading.Lock()

    def prewarm(self) -> None:
        """后台预加载模型，减少首次识别等待。"""
        with self._lock:
            self._ensure_engine()

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine

        if self._backend == "rapidocr":
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:
                raise ImportError(
                    "未安装 rapidocr-onnxruntime。"
                    "请执行: pip install rapidocr-onnxruntime"
                ) from exc
            self._engine = RapidOCR(use_angle_cls=self._use_angle_cls)
            return self._engine

        if self._backend == "easyocr":
            try:
                import easyocr
            except ImportError as exc:
                raise ImportError("未安装 easyocr，请执行: pip install easyocr") from exc
            self._engine = easyocr.Reader(["ch_sim", "en"], gpu=False)
            return self._engine

        raise ValueError(f"未知 OCR 后端: {self._backend}")

    def _variants_for(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        return build_ocr_variants(
            image_bgr,
            self._preprocess_scale,
            self._preprocess_mode,
            dual_preprocess=self._dual_preprocess,
            max_side=self._ocr_max_side,
        )

    def _rapidocr_rec_only(self, engine, variant: np.ndarray) -> tuple[list[str], float]:
        """跳过 det，整幅条带直接送识别（横排 1 行）。"""
        h, w = variant.shape[:2]
        _, crops = engine.get_boxes_img_without_det(variant, h, w)
        if engine.use_angle_cls:
            crops, _, _ = engine.text_cls(crops)
        rec_res, _ = engine.text_recognizer(crops)
        lines: list[str] = []
        conf_sum = 0.0
        conf_count = 0
        for item in rec_res:
            if not item or len(item) < 2:
                continue
            text, score = str(item[0]), float(item[1])
            if not text or score < self._text_score:
                continue
            lines.append(text)
            conf_sum += score
            conf_count += 1
        avg_conf = conf_sum / conf_count if conf_count else 0.0
        return lines, avg_conf

    def _rapidocr_once(
        self, engine, variant: np.ndarray
    ) -> tuple[list[str], float, float]:
        """完整 det + rec。"""
        result, _ = engine(
            variant,
            box_thresh=self._det_box_thresh,
            text_score=self._text_score,
        )
        if not result:
            return [], 0.0, 0.0
        ordered = _sort_ocr_lines(result)
        lines = [str(item[1]) for item in ordered if len(item) >= 2 and item[1]]
        if not lines:
            return [], 0.0, 0.0
        conf_sum = 0.0
        conf_count = 0
        for item in ordered:
            if len(item) >= 3 and item[2] is not None:
                conf_sum += float(item[2])
                conf_count += 1
        avg_conf = conf_sum / conf_count if conf_count else 0.0
        score = len(lines) * 10 + avg_conf
        return lines, score, avg_conf

    def _first_pass_good_enough(self, lines: list[str], avg_conf: float) -> bool:
        text = "".join(lines)
        if len(text) < _DUAL_SKIP_MIN_CHARS:
            return False
        return avg_conf >= _DUAL_SKIP_MIN_CONF

    def _recognize_rapidocr_strip(self, engine, image_bgr: np.ndarray) -> str:
        """1～2 行：水平切条 + 每条约一次 rec，跳过 det。"""
        strips = split_horizontal_strips(
            image_bgr, max_lines=self._ocr_strip_max_lines
        )
        all_lines: list[str] = []
        for strip in strips:
            best_lines: list[str] = []
            best_score = -1.0
            variants = self._variants_for(strip)
            for index, variant in enumerate(variants):
                lines, avg_conf = self._rapidocr_rec_only(engine, variant)
                if not lines:
                    continue
                score = len(lines) * 10 + avg_conf
                if score > best_score:
                    best_score = score
                    best_lines = lines
                if index == 0 and len(variants) > 1 and self._first_pass_good_enough(
                    lines, avg_conf
                ):
                    break
            all_lines.extend(best_lines)

        if all_lines:
            return "\n".join(all_lines)
        return self._recognize_rapidocr_full(engine, image_bgr)

    def _recognize_rapidocr_full(self, engine, image_bgr: np.ndarray) -> str:
        variants = self._variants_for(image_bgr)
        best_lines: list[str] = []
        best_score = -1.0

        for index, variant in enumerate(variants):
            lines, score, avg_conf = self._rapidocr_once(engine, variant)
            if not lines:
                continue
            if score > best_score:
                best_score = score
                best_lines = lines

            if index == 0 and len(variants) > 1 and self._first_pass_good_enough(
                lines, avg_conf
            ):
                return "\n".join(lines)

        return "\n".join(best_lines)

    def _recognize_rapidocr(self, image_bgr: np.ndarray) -> str:
        engine = self._engine
        if self._ocr_layout == "strip":
            return self._recognize_rapidocr_strip(engine, image_bgr)
        return self._recognize_rapidocr_full(engine, image_bgr)

    def recognize(self, image_bgr: np.ndarray) -> str:
        with self._lock:
            engine = self._ensure_engine()
            if self._backend == "rapidocr":
                return self._recognize_rapidocr(image_bgr)

            rgb = image_bgr[:, :, ::-1]
            rows = engine.readtext(rgb, detail=0, paragraph=True)
            if isinstance(rows, list):
                return "\n".join(str(line) for line in rows if line)
            return str(rows) if rows else ""
