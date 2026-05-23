"""RW_Tool 入口：置顶可缩放区域 OCR。"""

from __future__ import annotations

import rw_tool.qt6_bootstrap  # noqa: F401 — Windows 下须在 PyQt6 之前

try:
    import mss  # noqa: F401 — 须在 QApplication 之前，避免 DPI awareness 被降级
except ImportError:
    pass

import argparse
from pathlib import Path

from rw_tool.overlay_window import run


def main() -> None:
    parser = argparse.ArgumentParser(description="屏幕区域 OCR（置顶框体）")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="配置文件路径（默认项目根目录 config.ini）",
    )
    args = parser.parse_args()
    raise SystemExit(run(args.config))


if __name__ == "__main__":
    main()
