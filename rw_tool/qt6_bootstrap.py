"""在导入 PyQt6 之前：DLL 路径 + Windows Per-Monitor DPI（须在 Qt 之前）。"""

from __future__ import annotations

import os
import sys
from importlib.util import find_spec
from pathlib import Path


def ensure_process_dpi_aware() -> None:
    """保证 LogicalToPhysicalPointForPerMonitorDPI / mss 使用同一套物理坐标。"""
    if sys.platform != "win32":
        return

    import ctypes

    try:
        # PROCESS_PER_MONITOR_DPI_AWARE = 2
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass

    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def ensure_pyqt6_dll_path() -> None:
    if sys.platform != "win32":
        return

    spec = find_spec("PyQt6")
    if spec is None or not spec.submodule_search_locations:
        return

    qt_bin = Path(spec.submodule_search_locations[0]).resolve() / "Qt6" / "bin"
    dirs: list[Path] = []
    if qt_bin.is_dir():
        dirs.append(qt_bin)
    # 由 Anaconda 创建的 venv：补充基环境里的 VC 运行库路径
    base = Path(sys.base_prefix)
    for extra in (base, base / "Library" / "bin"):
        if extra.is_dir():
            dirs.append(extra)

    for folder in dirs:
        folder_str = str(folder)
        os.environ["PATH"] = folder_str + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(folder_str)


ensure_process_dpi_aware()
ensure_pyqt6_dll_path()
