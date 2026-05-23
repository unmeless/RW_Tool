from __future__ import annotations

import numpy as np


def split_horizontal_strips(
    bgr: np.ndarray,
    *,
    max_lines: int = 2,
    min_line_h: int = 18,
) -> list[np.ndarray]:
    """
    将 ROI 按水平切成最多 max_lines 条，供「仅识别、不检测」快路径使用。
    适合游戏内 1～2 行固定横排描述。
    """
    if bgr is None or bgr.size == 0:
        return []
    h, w = bgr.shape[:2]
    if h < 2 or w < 2:
        return [bgr]

    max_lines = max(1, min(3, max_lines))
    if max_lines == 1 or h < int(min_line_h * 1.6):
        return [bgr]

    if max_lines == 2:
        mid = h // 2
        pad = max(1, min(4, h // 24))
        top = bgr[: mid + pad, :]
        bottom = bgr[max(0, mid - pad) :, :]
        strips: list[np.ndarray] = []
        if top.shape[0] >= min_line_h:
            strips.append(top)
        if bottom.shape[0] >= min_line_h:
            strips.append(bottom)
        return strips if strips else [bgr]

    # 最多 3 行：等高切分
    n = max_lines
    step = h / n
    strips = []
    overlap = max(1, int(step * 0.08))
    for i in range(n):
        y0 = max(0, int(i * step) - (overlap if i else 0))
        y1 = min(h, int((i + 1) * step) + (overlap if i < n - 1 else 0))
        part = bgr[y0:y1, :]
        if part.shape[0] >= min_line_h:
            strips.append(part)
    return strips if strips else [bgr]
