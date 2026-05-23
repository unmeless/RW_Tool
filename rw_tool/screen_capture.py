from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from PyQt6.QtCore import QRect
    from PyQt6.QtGui import QScreen
    from PyQt6.QtWidgets import QWidget


@dataclass(frozen=True)
class CaptureRegion:
    """识别区坐标。"""

    logical: tuple[int, int, int, int]  # 虚拟桌面逻辑 x,y,w,h
    screen_rel: tuple[int, int, int, int]  # 相对当前屏左上角逻辑 x,y,w,h — grabWindow 参数
    device: tuple[int, int, int, int]  # 期望截图像素 x,y,w,h（屏内设备像素，通常 = 逻辑×DPR）
    physical: tuple[int, int, int, int]  # mss 虚拟桌布物理 x,y,w,h
    screen_name: str
    dpr: float
    physical_method: str
    mss_usable: bool
    last_grab: str = ""


def qimage_to_bgr(image) -> np.ndarray:
    from PyQt6.QtGui import QImage

    image = image.convertToFormat(QImage.Format.Format_RGB888)
    w, h = image.width(), image.height()
    stride = image.bytesPerLine()
    buf = image.bits()
    buf.setsize(h * stride)
    arr = np.frombuffer(buf, dtype=np.uint8).copy()
    rgb = arr.reshape((h, stride))[:, : w * 3].reshape((h, w, 3))
    return np.ascontiguousarray(rgb[:, :, ::-1])


def global_rect_from_widget(widget: QWidget, local_rect: QRect) -> QRect:
    from PyQt6.QtCore import QPoint, QRect, QSize

    tl = widget.mapToGlobal(local_rect.topLeft())
    return QRect(QPoint(tl), QSize(local_rect.width(), local_rect.height()))


def _logical_to_physical_win32(x: int, y: int) -> tuple[int, int]:
    if sys.platform != "win32":
        return x, y

    import ctypes
    from ctypes import wintypes

    class POINT(ctypes.Structure):
        _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

    user32 = ctypes.windll.user32
    pt = POINT(int(x), int(y))

    if hasattr(user32, "LogicalToPhysicalPointForPerMonitorDPI"):
        if user32.LogicalToPhysicalPointForPerMonitorDPI(ctypes.byref(pt)):
            return int(pt.x), int(pt.y)

    if user32.LogicalToPhysicalPoint(ctypes.byref(pt)):
        return int(pt.x), int(pt.y)

    return x, y


def _win32_converted(lx: int, ly: int, px: int, py: int, dpr: float) -> bool:
    if abs(px - lx) > 1 or abs(py - ly) > 1:
        return True
    return dpr <= 1.01


def _mss_coords_valid(ptx: int, pty: int, pw: int, ph: int) -> bool:
    return ptx >= 0 and pty >= 0 and pw >= 1 and ph >= 1


def _physical_rect_for_mss(
    lx: int,
    ly: int,
    lw: int,
    lh: int,
    device: tuple[int, int, int, int],
    dpr: float,
) -> tuple[int, int, int, int, str, bool]:
    _, _, dw, dh = device
    ptx, pty = _logical_to_physical_win32(lx, ly)
    pbrx, pbry = _logical_to_physical_win32(lx + lw, ly + lh)
    pw = max(1, pbrx - ptx)
    ph = max(1, pbry - pty)

    converted = _win32_converted(lx, ly, ptx, pty, dpr)
    size_ok = abs(pw - dw) <= 2 and abs(ph - dh) <= 2
    valid = converted and size_ok and _mss_coords_valid(ptx, pty, pw, ph)

    if valid:
        return ptx, pty, pw, ph, "Win32", True

    return 0, 0, 0, 0, "n/a", False


def _screen_for_logical_point(lx: int, ly: int, widget: QWidget) -> QScreen | None:
    from PyQt6.QtCore import QPoint
    from PyQt6.QtGui import QGuiApplication

    app = QGuiApplication.instance()
    if app is None:
        return widget.screen()
    return app.screenAt(QPoint(lx, ly)) or widget.screen()


def resolve_capture_region(widget: QWidget, local_rect: QRect) -> CaptureRegion | None:
    if local_rect.width() < 2 or local_rect.height() < 2:
        return None

    g = global_rect_from_widget(widget, local_rect)
    lx, ly, lw, lh = g.x(), g.y(), g.width(), g.height()

    screen = _screen_for_logical_point(lx, ly, widget)
    if screen is None:
        return None

    geo = screen.geometry()
    dpr = float(screen.devicePixelRatio())
    rel_x = lx - geo.x()
    rel_y = ly - geo.y()

    # grabWindow 在 Windows+高 DPI 下按「屏内逻辑像素」解释 x,y,w,h，不能先乘 DPR
    screen_rel = (
        max(0, int(rel_x)),
        max(0, int(rel_y)),
        max(1, int(lw)),
        max(1, int(lh)),
    )
    device = (
        max(0, int(round(rel_x * dpr))),
        max(0, int(round(rel_y * dpr))),
        max(1, int(round(lw * dpr))),
        max(1, int(round(lh * dpr))),
    )
    ptx, pty, pw, ph, method, mss_usable = _physical_rect_for_mss(
        lx, ly, lw, lh, device, dpr
    )

    return CaptureRegion(
        logical=(lx, ly, lw, lh),
        screen_rel=screen_rel,
        device=device,
        physical=(ptx, pty, pw, ph),
        screen_name=screen.name(),
        dpr=dpr,
        physical_method=method,
        mss_usable=mss_usable,
    )


def _normalize_grab_image(
    img: np.ndarray,
    log_w: int,
    log_h: int,
    dpr: float,
) -> np.ndarray:
    """将 grab 结果规范到「逻辑×DPR」设备像素尺寸，修正误传 2× 逻辑参数导致的 4× 图。"""
    ew = max(1, int(round(log_w * dpr)))
    eh = max(1, int(round(log_h * dpr)))
    h, w = img.shape[:2]

    if w == ew and h == eh:
        return img

    # 曾把 (逻辑×DPR) 当作 grab 参数 → 出图约为期望的 2×（宽×2、高×2）
    if w >= ew * 2 - 2 and h >= eh * 2 - 2:
        return cv2.resize(img, (ew, eh), interpolation=cv2.INTER_AREA)

    return cv2.resize(img, (ew, eh), interpolation=cv2.INTER_LINEAR)


def _grab_qt_screen(screen: QScreen, region: CaptureRegion) -> np.ndarray | None:
    rx, ry, rw, rh = region.screen_rel
    pixmap = screen.grabWindow(0, rx, ry, rw, rh)
    if pixmap.isNull():
        return None
    img = qimage_to_bgr(pixmap.toImage())
    if img.shape[0] < 2 or img.shape[1] < 2:
        return None
    return _normalize_grab_image(img, rw, rh, region.dpr)


def _grab_mss_physical(ptx: int, pty: int, pw: int, ph: int) -> np.ndarray | None:
    if not _mss_coords_valid(ptx, pty, pw, ph):
        return None

    try:
        import mss
    except ImportError:
        return None

    with mss.mss() as sct:
        shot = sct.grab({"left": ptx, "top": pty, "width": pw, "height": ph})
    bgra = np.array(shot, dtype=np.uint8)
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)


def format_capture_debug_lines(region: CaptureRegion) -> list[str]:
    lx, ly, lw, lh = region.logical
    gx, gy, gw, gh = region.screen_rel
    dx, dy, dw, dh = region.device
    px, py, pw, ph = region.physical
    grab = region.last_grab or "—"
    if region.mss_usable:
        p_line = f"物理 P({region.physical_method}): {px},{py}  {pw}×{ph}"
    else:
        p_line = "物理 P: 不可用"
    return [
        f"屏: {region.screen_name}  DPR={region.dpr:g}  截屏={grab}",
        f"逻辑 L: {lx},{ly}  {lw}×{lh}",
        f"屏内 G: {gx},{gy}  {gw}×{gh}  (grab)",
        f"设备 D: {dx},{dy}  {dw}×{dh}",
        p_line,
    ]


def grab_capture_bgr(widget: QWidget, local_rect: QRect) -> tuple[np.ndarray | None, CaptureRegion | None]:
    region = resolve_capture_region(widget, local_rect)
    if region is None:
        return None, None

    screen = _screen_for_logical_point(region.logical[0], region.logical[1], widget)

    if screen is not None:
        img = _grab_qt_screen(screen, region)
        if img is not None:
            return img, replace(region, last_grab="Qt")

    if sys.platform == "win32" and region.mss_usable:
        ptx, pty, pw, ph = region.physical
        img = _grab_mss_physical(ptx, pty, pw, ph)
        if img is not None:
            _, _, dw, dh = region.device
            if img.shape[1] != dw or img.shape[0] != dh:
                img = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_LINEAR)
            return img, replace(region, last_grab="mss")

    return None, replace(region, last_grab="失败")


def save_debug_image(bgr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)


def save_debug_metadata(region: CaptureRegion, path: Path) -> None:
    lx, ly, lw, lh = region.logical
    gx, gy, gw, gh = region.screen_rel
    dx, dy, dw, dh = region.device
    px, py, pw, ph = region.physical
    text = (
        f"screen: {region.screen_name}\n"
        f"dpr: {region.dpr}\n"
        f"last_grab: {region.last_grab}\n"
        f"mss_usable: {region.mss_usable}\n"
        f"physical_method: {region.physical_method}\n"
        f"logical: x={lx} y={ly} w={lw} h={lh}\n"
        f"screen_rel: x={gx} y={gy} w={gw} h={gh}\n"
        f"device: x={dx} y={dy} w={dw} h={dh}\n"
        f"physical: x={px} y={py} w={pw} h={ph}\n"
    )
    path.write_text(text, encoding="utf-8")
