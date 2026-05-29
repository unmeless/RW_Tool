from __future__ import annotations

import cv2
import numpy as np


def scale_image(bgr: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 1.0:
        return bgr
    return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


def capture_fingerprint(bgr: np.ndarray) -> int:
    """低成本画面指纹，用于跳过未变化的 OCR。"""
    if bgr is None or bgr.size == 0:
        return 0
    small = cv2.resize(bgr, (32, 16), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return hash(gray.tobytes())


def cap_max_side(bgr: np.ndarray, max_side: int) -> np.ndarray:
    """限制最长边，避免超大 ROI 拖慢 det+rec。"""
    if max_side <= 0:
        return bgr
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return bgr
    ratio = max_side / longest
    return cv2.resize(
        bgr,
        (max(1, int(w * ratio)), max(1, int(h * ratio))),
        interpolation=cv2.INTER_AREA,
    )


def enhance_game_ui_text(bgr: np.ndarray) -> np.ndarray:
    """
    针对游戏半透明底 + 白字：拉高对比度，转为深色字白底，便于 OCR。
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge([l_channel, a, b]), cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25,
        8,
    )
    # 白字深底 → 反转为黑字白底
    if float(np.mean(gray)) < 128.0:
        binary = cv2.bitwise_not(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def build_ocr_variants(
    bgr: np.ndarray,
    scale: float,
    mode: str,
    *,
    dual_preprocess: bool = False,
    max_side: int = 0,
) -> list[np.ndarray]:
    scaled = cap_max_side(scale_image(bgr, scale), max_side)
    if mode == "game_ui" and dual_preprocess:
        return [scaled, enhance_game_ui_text(scaled)]
    return [scaled]
