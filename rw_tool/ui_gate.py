from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_PROFILE_VERSION = 1
_FEATURE_KEYS = (
    "gray_mean",
    "gray_std",
    "sat_mean",
    "sat_std",
    "gray_panel_ratio",
    "bright_text_ratio",
    "l_mean",
    "l_std",
)


@dataclass(frozen=True)
class UiGateFeatures:
    """描述框 ROI 的轻量统计特征（半透明灰底 + 白字）。"""

    gray_mean: float
    gray_std: float
    sat_mean: float
    sat_std: float
    gray_panel_ratio: float
    bright_text_ratio: float
    l_mean: float
    l_std: float

    def as_dict(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in _FEATURE_KEYS}


def extract_ui_gate_features(bgr: np.ndarray) -> UiGateFeatures:
    if bgr is None or bgr.size == 0:
        return UiGateFeatures(0, 0, 0, 0, 0, 0, 0, 0)

    small = cv2.resize(bgr, (80, 40), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0].astype(np.float32)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    gray_panel = (l_ch >= 28) & (l_ch <= 105) & (sat <= 70)
    bright_text = val >= 145

    return UiGateFeatures(
        gray_mean=float(np.mean(gray)),
        gray_std=float(np.std(gray)),
        sat_mean=float(np.mean(sat)),
        sat_std=float(np.std(sat)),
        gray_panel_ratio=float(np.mean(gray_panel)),
        bright_text_ratio=float(np.mean(bright_text)),
        l_mean=float(np.mean(l_ch)),
        l_std=float(np.std(l_ch)),
    )


def heuristic_dialog_present(features: UiGateFeatures) -> bool:
    """
    内置启发式：半透明灰底对话框 + 文字。
    阈值偏宽松，避免误拦真实游戏 UI。
    """
    if features.gray_panel_ratio < 0.20:
        return False
    if features.sat_mean > 95:
        return False
    # 灰底占主导时，允许较低的文字高亮占比
    if features.gray_panel_ratio >= 0.45:
        if features.l_std < 4:
            return False
        if features.bright_text_ratio < 0.006:
            return False
        return True
    if features.bright_text_ratio < 0.008:
        return False
    if features.l_std < 4:
        return False
    return True


def _aggregate_stats(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not samples:
        return {}
    agg: dict[str, dict[str, float]] = {}
    for key in _FEATURE_KEYS:
        vals = [float(s[key]) for s in samples if key in s]
        if not vals:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        agg[key] = {"mean": mean, "std": max(std, 1.5)}
    return agg


def _profile_similarity(features: UiGateFeatures, aggregate: dict[str, dict[str, float]]) -> float:
    if not aggregate:
        return 0.0
    scores: list[float] = []
    for key in _FEATURE_KEYS:
        bounds = aggregate.get(key)
        if bounds is None:
            continue
        val = getattr(features, key)
        mean = bounds["mean"]
        std = bounds["std"]
        z = abs(val - mean) / std
        scores.append(max(0.0, 1.0 - z / 2.5))
    return float(np.mean(scores)) if scores else 0.0


class UiGate:
    """根据已记录的高置信样本 + 灰底对话框启发式，决定是否启动 OCR。"""

    def __init__(
        self,
        profile_path: Path,
        *,
        match_threshold: float = 0.52,
        record_min_match_score: float = 58.0,
        max_samples: int = 24,
        use_heuristic: bool = True,
        min_samples_to_gate: int = 3,
    ) -> None:
        self._profile_path = profile_path
        self._match_threshold = match_threshold
        self._record_min_match_score = record_min_match_score
        self._max_samples = max(1, max_samples)
        self._use_heuristic = use_heuristic
        self._min_samples_to_gate = max(0, min_samples_to_gate)
        self._samples: list[dict[str, Any]] = []
        self._aggregate: dict[str, dict[str, float]] = {}
        self._load()

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def _load(self) -> None:
        if not self._profile_path.is_file():
            return
        try:
            data = json.loads(self._profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        raw = data.get("samples")
        if isinstance(raw, list):
            self._samples = [s for s in raw if isinstance(s, dict) and "features" in s]
        agg = data.get("aggregate")
        if isinstance(agg, dict):
            self._aggregate = agg  # type: ignore[assignment]
        else:
            self._rebuild_aggregate()

    def _rebuild_aggregate(self) -> None:
        feats = [s["features"] for s in self._samples if isinstance(s.get("features"), dict)]
        self._aggregate = _aggregate_stats(feats)

    def _save(self) -> None:
        payload = {
            "version": _PROFILE_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "sample_count": len(self._samples),
            "aggregate": self._aggregate,
            "samples": self._samples,
        }
        self._profile_path.parent.mkdir(parents=True, exist_ok=True)
        self._profile_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def should_run_ocr(self, bgr: np.ndarray) -> tuple[bool, float, str]:
        """返回 (是否跑 OCR, 置信分 0~1, 原因简述)。"""
        if self._min_samples_to_gate > 0 and len(self._samples) < self._min_samples_to_gate:
            return True, 1.0, f"bootstrap({len(self._samples)}/{self._min_samples_to_gate})"

        features = extract_ui_gate_features(bgr)
        heuristic = heuristic_dialog_present(features) if self._use_heuristic else False

        if self._aggregate:
            sim = _profile_similarity(features, self._aggregate)
            if sim >= self._match_threshold:
                return True, sim, f"profile({sim:.2f})"
            if heuristic:
                return True, sim, f"heuristic+profile({sim:.2f})"
            return False, sim, f"no_dialog({sim:.2f})"

        if heuristic:
            return True, 1.0, "heuristic"
        return False, 0.0, "no_dialog"

    def record_success(self, bgr: np.ndarray, *, match_score: float, pet_name: str = "") -> bool:
        """高置信匹配成功后追加样本特征。"""
        if match_score < self._record_min_match_score:
            return False
        features = extract_ui_gate_features(bgr)
        entry = {
            "features": features.as_dict(),
            "match_score": float(match_score),
            "pet_name": pet_name,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        self._samples.append(entry)
        if len(self._samples) > self._max_samples:
            self._samples = self._samples[-self._max_samples :]
        self._rebuild_aggregate()
        self._save()
        return True
