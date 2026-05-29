from __future__ import annotations

import sys
from collections.abc import Iterable

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication, QWidget


class FrameLockHub(QObject):
    """Caps Lock 切换解锁时通知各窗体刷新（默认锁定）。"""

    interaction_changed = pyqtSignal()

    def notify_interaction_changed(self) -> None:
        self.interaction_changed.emit()


def _win32_caps_lock_on() -> bool:
    import ctypes

    # VK_CAPITAL；切换键需 GetKeyState 低 bit 表示开/关
    return bool(ctypes.windll.user32.GetKeyState(0x14) & 1)


def caps_lock_unlock_active() -> bool:
    """Caps Lock 开启时 UI 解锁。"""
    if sys.platform == "win32":
        return _win32_caps_lock_on()
    return False


def frame_locked() -> bool:
    """Caps Lock 关闭时为锁定态。"""
    return not caps_lock_unlock_active()


def frame_interaction_allowed() -> bool:
    """Caps Lock 开启时可拖动、缩放。"""
    return caps_lock_unlock_active()


def set_window_input_passthrough(
    window: QWidget,
    enabled: bool,
    *,
    base_flags: Qt.WindowType,
) -> None:
    """
    锁定图标条：保持窗体与内容可见，鼠标/键盘输入穿透到下层（勿用 setMask，会裁切显示）。
    """
    window.clearMask()
    window.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, enabled)
    if enabled:
        new_flags = base_flags | Qt.WindowType.WindowTransparentForInput
    else:
        new_flags = base_flags & ~Qt.WindowType.WindowTransparentForInput
    if window.windowFlags() != new_flags:
        geo = window.geometry()
        window.setWindowFlags(new_flags)
        window.setGeometry(geo)
        window.show()
    if enabled:
        window.unsetCursor()


def update_frame_lock_state(
    root: QWidget,
    *,
    allowed: Iterable[QWidget],
    always_pass_through: Iterable[QWidget] | None = None,
) -> None:
    """按 Caps Lock 更新鼠标穿透与绘制。"""
    locked = frame_locked()
    interactive = frame_interaction_allowed()
    root.setAttribute(
        Qt.WidgetAttribute.WA_TransparentForMouseEvents,
        locked,
    )
    apply_lock_mouse_policy(
        root,
        allowed=allowed,
        always_pass_through=always_pass_through,
        locked=locked,
    )
    if not interactive:
        root.unsetCursor()
    root.update()


def apply_lock_mouse_policy(
    root: QWidget,
    *,
    allowed: Iterable[QWidget],
    always_pass_through: Iterable[QWidget] | None = None,
    locked: bool,
) -> None:
    """
    锁定时仅 allowed 内控件可接收鼠标；其余子控件穿透。
    解锁时恢复（always_pass_through 仍保持穿透）。
    """
    interactive = not locked
    allowed_set = set(allowed)
    pass_set = set(always_pass_through or ())

    def under_allowed(widget: QWidget) -> bool:
        w: QWidget | None = widget
        while w is not None:
            if w in allowed_set:
                return True
            w = w.parentWidget()
        return False

    for child in root.findChildren(QWidget):
        if child in pass_set:
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            continue
        if interactive:
            if not under_allowed(child):
                child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        elif under_allowed(child):
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        else:
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)


class CapsLockMonitor(QObject):
    """定时轮询 Caps Lock（游戏有焦点时 Qt 收不到按键）。"""

    _POLL_MS = 100

    def __init__(self, widgets: list[QWidget], lock_hub: FrameLockHub) -> None:
        super().__init__()
        self._widgets = widgets
        self._lock_hub = lock_hub
        self._prev_unlock = caps_lock_unlock_active()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_poll)
        self._timer.start(self._POLL_MS)

    def _on_poll(self) -> None:
        cur = caps_lock_unlock_active()
        if cur == self._prev_unlock:
            return
        self._prev_unlock = cur
        for widget in self._widgets:
            widget.update()
            if frame_locked():
                widget.unsetCursor()
        self._lock_hub.notify_interaction_changed()


def install_caps_lock_monitor(
    app: QApplication,
    widgets: list[QWidget],
    lock_hub: FrameLockHub,
) -> CapsLockMonitor:
    monitor = CapsLockMonitor(widgets, lock_hub)
    monitor.setParent(app)
    return monitor
