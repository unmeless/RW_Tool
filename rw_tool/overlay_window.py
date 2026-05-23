from __future__ import annotations

import rw_tool.qt6_bootstrap  # noqa: F401 — 须在 PyQt6 之前加载

import sys
import threading
from enum import IntFlag, auto
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QPoint, QRect, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen, QRegion
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rw_tool.config import AppConfig, load_config
from rw_tool.frame_lock import (
    AltKeyMonitor,
    FrameLockHub,
    frame_interaction_allowed,
    install_alt_key_monitor,
    update_frame_lock_state,
)
from rw_tool.icon_panel_window import PetIconPanelWindow
from rw_tool.window_state import load_window_state, save_window_state, state_path_for_config
from rw_tool.ocr_engine import OcrEngine
from rw_tool.pet_matcher import MatchResult, PetMatcher
from rw_tool.screen_capture import (
    CaptureRegion,
    format_capture_debug_lines,
    grab_capture_bgr,
    resolve_capture_region,
    save_debug_image,
    save_debug_metadata,
)


class ResizeMode(IntFlag):
    NONE = 0
    LEFT = auto()
    RIGHT = auto()
    TOP = auto()
    BOTTOM = auto()


class OcrWorker(QThread):
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        parent: QWidget | None,
        engine: OcrEngine,
        image_bgr: np.ndarray,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._image = image_bgr

    def run(self) -> None:
        try:
            text = self._engine.recognize(self._image)
            self.finished.emit(text)
        except Exception as exc:  # noqa: BLE001 — 展示给用户
            self.failed.emit(str(exc))


class OcrOverlayWindow(QWidget):
    """
    置顶、无边框的屏幕区域 OCR 框。

    默认可配合独立图标条：主框仅保留扫描镂空区 +（可选）调试 Settings。
    icon_panel.enabled=false 时，仍可在主框显示匹配列表与 OCR 原文。
    """

    _BORDER_PX = 2

    def __init__(self, config: AppConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or load_config()
        self._show_text_panels = not self.cfg.icon_panel_enabled
        backend = self.cfg.ocr_backend if self.cfg.ocr_backend in ("rapidocr", "easyocr") else "rapidocr"
        self._engine = OcrEngine(
            backend=backend,  # type: ignore[arg-type]
            det_box_thresh=self.cfg.det_box_thresh,
            text_score=self.cfg.text_score,
            preprocess_scale=self.cfg.preprocess_scale,
            preprocess_mode=self.cfg.preprocess_mode,
            dual_preprocess=self.cfg.dual_preprocess,
            ocr_max_side=self.cfg.ocr_max_side,
            ocr_layout=self.cfg.ocr_layout,
            ocr_strip_max_lines=self.cfg.ocr_strip_max_lines,
            use_angle_cls=self.cfg.ocr_use_angle_cls,
        )
        threading.Thread(target=self._engine.prewarm, daemon=True).start()

        self._capture_w = 360
        self._capture_h = 200
        self._resize_mode = ResizeMode.NONE
        self._drag_start_global = QPoint()
        self._geom_start = QRect()
        self._moving = False
        self._interacting = False
        self._ocr_busy = False
        self._worker: OcrWorker | None = None
        self._last_capture_region: CaptureRegion | None = None
        self._runtime_save_capture = self.cfg.debug_save_capture
        self._matcher: PetMatcher | None = None
        self._icon_panel: PetIconPanelWindow | None = None
        self._icon_panel_placed = False
        self._last_match_result: MatchResult | None = None
        self._lock_hub = FrameLockHub()
        self._alt_monitor: AltKeyMonitor | None = None
        self._match_label: QLabel | None = None
        self._result_label: QLabel | None = None
        if self.cfg.matcher_enabled:
            try:
                self._matcher = PetMatcher.from_path(
                    self.cfg.catalog_path,
                    min_score=self.cfg.match_min_score,
                    min_margin=self.cfg.match_min_margin,
                    min_candidate_score=self.cfg.match_min_candidate_score,
                    top_k=self.cfg.match_top_k,
                )
            except (OSError, ValueError) as exc:
                print(f"[RW_Tool] 图鉴加载失败: {exc}")

        if self.cfg.icon_panel_enabled:
            self._icon_panel = PetIconPanelWindow(self.cfg, self.cfg.icon_dir)

        self._apply_window_flags()
        self._build_ui()
        self._setup_title_bar_controls()
        self._reset_size_constraints()
        if not self._restore_saved_geometry():
            self._apply_geometry()
        self._update_window_mask()
        self._sync_settings_panel()
        if self._debug_mode_active():
            self._refresh_capture_region_info()
            self._update_settings_debug_text()
        self._start_ocr_timer()
        if self._debug_mode_active():
            self._start_debug_refresh_timer()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._position_title_bar_controls()
        if self._icon_panel is not None:
            self._icon_panel.show()
            if not self._icon_panel_placed:
                self._place_icon_panel_initial()
            self._icon_panel_placed = True
        if self._debug_mode_active():
            self._refresh_capture_region_info()
            self._update_settings_debug_text()

    def _state_path(self) -> Path:
        return state_path_for_config(self.cfg.config_path)

    def _restore_saved_geometry(self) -> bool:
        if not self.cfg.restore_saved_geometry:
            return False
        state = load_window_state(self._state_path())
        if not state:
            return False

        overlay = state.get("overlay")
        if isinstance(overlay, dict) and all(k in overlay for k in ("x", "y", "w", "h")):
            chrome = self._chrome_height()
            h_total = int(overlay["h"])
            self._capture_h = max(
                self.cfg.min_capture_height,
                int(overlay.get("capture_h", h_total - chrome)),
            )
            self._capture_w = max(
                self.cfg.min_capture_width,
                int(overlay.get("capture_w", int(overlay["w"]))),
            )
            g = self._clamp_geometry(
                QRect(
                    int(overlay["x"]),
                    int(overlay["y"]),
                    int(overlay["w"]),
                    h_total,
                )
            )
            self.setGeometry(g)
            self._capture_w = g.width()
            self._capture_h = max(self.cfg.min_capture_height, g.height() - chrome)

        icon = state.get("icon_panel")
        if self._icon_panel is not None and isinstance(icon, dict):
            if all(k in icon for k in ("x", "y", "w", "h")):
                self._icon_panel.apply_saved_geometry(
                    int(icon["x"]),
                    int(icon["y"]),
                    int(icon["w"]),
                    int(icon["h"]),
                )
                self._icon_panel_placed = True

        self._apply_geometry()
        self._sync_match_panel()
        self._sync_settings_panel()
        return True

    def _save_window_geometry(self) -> None:
        if not self.cfg.restore_saved_geometry:
            return
        g = self.geometry()
        overlay = {
            "x": g.x(),
            "y": g.y(),
            "w": g.width(),
            "h": g.height(),
            "capture_w": self._capture_w,
            "capture_h": self._capture_h,
        }
        icon_panel = None
        if self._icon_panel is not None and self._icon_panel.isVisible():
            gi = self._icon_panel.geometry()
            icon_panel = {
                "x": gi.x(),
                "y": gi.y(),
                "w": gi.width(),
                "h": gi.height(),
            }
        save_window_state(
            self._state_path(),
            overlay=overlay,
            icon_panel=icon_panel,
        )

    def _place_icon_panel_initial(self) -> None:
        """无缓存时，首次显示摆在主框右侧。"""
        if self._icon_panel is None:
            return
        g = self.geometry()
        self._icon_panel.move(g.right() + 12, g.top())

    def _start_debug_refresh_timer(self) -> None:
        self._debug_timer = QTimer(self)
        self._debug_timer.timeout.connect(self._on_debug_refresh_tick)
        self._debug_timer.start(350)

    def _on_debug_refresh_tick(self) -> None:
        if self._interacting:
            return
        self._refresh_capture_region_info()
        self._update_settings_debug_text()

    def _apply_window_flags(self) -> None:
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)

    def _build_ui(self) -> None:
        panel_style = (
            "background: rgba(20, 20, 24, 230);"
            "color: #e8e8ec;"
            "border: 1px solid #4a9eff;"
        )
        bottom_panel_style = panel_style + " border-top: none;"

        self._capture_spacer = QWidget()
        self._capture_spacer.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self._settings_panel = QWidget()
        settings_layout = QVBoxLayout(self._settings_panel)
        settings_layout.setContentsMargins(8, 4, 8, 4)
        settings_layout.setSpacing(2)

        settings_title = QLabel("Settings · 调试")
        settings_title.setStyleSheet(
            f"color: #8ec8ff; font-size: {self.cfg.font_settings_title}px; font-weight: bold;"
        )
        settings_layout.addWidget(settings_title)

        opts = QHBoxLayout()
        self._chk_save_capture = QCheckBox("保存截屏")
        self._chk_save_capture.setChecked(self._runtime_save_capture)
        self._chk_save_capture.setStyleSheet(
            f"color: #c8c8d0; font-size: {self.cfg.font_settings_checkbox}px;"
        )
        self._chk_save_capture.toggled.connect(self._on_save_capture_toggled)
        opts.addWidget(self._chk_save_capture)
        opts.addStretch(1)
        settings_layout.addLayout(opts)

        self._settings_debug_label = QLabel("（等待坐标…）")
        self._settings_debug_label.setWordWrap(True)
        self._settings_debug_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._settings_debug_label.setStyleSheet(
            "color: #ffdc78; font-family: Consolas, 'Microsoft YaHei UI';"
            f" font-size: {self.cfg.font_settings_debug}px;"
        )
        settings_layout.addWidget(self._settings_debug_label, 1)
        self._settings_panel.setStyleSheet(f"QWidget {{ {bottom_panel_style} }}")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        if self._show_text_panels:
            self._match_label = QLabel("（等待识别…）")
            self._match_label.setWordWrap(True)
            self._match_label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            self._match_label.setStyleSheet(
                f"QLabel {{ {panel_style} border-bottom: none; padding: 6px 10px;"
                f" font-size: {self.cfg.font_match}px; font-weight: bold; }}"
            )
            root.addWidget(self._match_label)

        root.addWidget(self._capture_spacer)
        root.addWidget(self._settings_panel)

        if self._show_text_panels:
            self._result_label = QLabel("（等待识别…）")
            self._result_label.setWordWrap(True)
            self._result_label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
            )
            self._result_label.setStyleSheet(
                f"QLabel {{ {bottom_panel_style} padding: 8px;"
                f" font-size: {self.cfg.font_ocr_result}px; }}"
            )
            root.addWidget(self._result_label)
            self._sync_match_panel()

    def _debug_mode_active(self) -> bool:
        return self.cfg.debug_enabled

    def _settings_panel_height_active(self) -> int:
        return self.cfg.settings_panel_height if self._debug_mode_active() else 0

    def _match_panel_height_active(self) -> int:
        if not self._show_text_panels:
            return 0
        return self.cfg.match_panel_height if self._matcher is not None else 0

    def _result_panel_height_active(self) -> int:
        if not self._show_text_panels:
            return 0
        return self.cfg.result_panel_height

    def _top_panels_height(self) -> int:
        return self._match_panel_height_active()

    def _bottom_panels_height(self) -> int:
        return self._settings_panel_height_active() + self._result_panel_height_active()

    def _chrome_height(self) -> int:
        return self._top_panels_height() + self._bottom_panels_height()

    def _sync_match_panel(self) -> None:
        if self._match_label is None:
            return
        visible = self._matcher is not None
        self._match_label.setVisible(visible)
        h = self._match_panel_height_active()
        self._match_label.setFixedHeight(h if visible else 0)

    def _sync_settings_panel(self) -> None:
        visible = self._debug_mode_active()
        self._settings_panel.setVisible(visible)
        h = self._settings_panel_height_active()
        self._settings_panel.setFixedHeight(h if visible else 0)

    def _update_settings_debug_text(self) -> None:
        if not self._debug_mode_active():
            return
        region = self._last_capture_region
        if region is None:
            self._refresh_capture_region_info()
            region = self._last_capture_region
        if region is None:
            self._settings_debug_label.setText("（无有效镂空区）")
            return
        self._settings_debug_label.setText("\n".join(format_capture_debug_lines(region)))

    def _on_save_capture_toggled(self, checked: bool) -> None:
        self._runtime_save_capture = checked

    def _apply_geometry(self) -> None:
        total_h = self._capture_h + self._chrome_height()
        self.resize(self._capture_w, total_h)
        self._sync_match_panel()
        self._sync_settings_panel()
        self._capture_spacer.setFixedHeight(self._capture_h)
        if self._result_label is not None:
            self._result_label.setFixedHeight(self._result_panel_height_active())
            self._result_label.setVisible(True)

    def _min_window_size(self) -> QSize:
        return QSize(
            self.cfg.min_capture_width,
            self.cfg.min_capture_height + self._chrome_height(),
        )

    def _reset_size_constraints(self) -> None:
        """Windows 上 setMask 会把 minimumSize 撑大，每次 mask 后强制复位。"""
        self.setMinimumSize(self._min_window_size())
        self.setMaximumSize(16777215, 16777215)

    def _virtual_available_rect(self) -> QRect:
        """所有显示器的可用区域并集，支持跨屏移动与缩放。"""
        app = QGuiApplication.instance()
        if app is None:
            return QRect()
        united = QRect()
        for screen in app.screens():
            united = united.united(screen.availableGeometry()) if not united.isNull() else QRect(
                screen.availableGeometry()
            )
        return united

    def _clamp_geometry(self, g: QRect) -> QRect:
        avail = self._virtual_available_rect()
        if avail.isNull():
            return g
        min_sz = self._min_window_size()
        w = max(min_sz.width(), min(g.width(), avail.width()))
        h = max(min_sz.height(), min(g.height(), avail.height()))
        x = max(avail.left(), min(g.x(), avail.right() - w + 1))
        y = max(avail.top(), min(g.y(), avail.bottom() - h + 1))
        return QRect(x, y, w, h)

    def drag_bar_rect(self) -> QRect:
        cr = self.capture_rect
        h = min(self.cfg.drag_bar_height, max(0, cr.height() - 1))
        return QRect(cr.left(), cr.top(), cr.width(), h)

    def close_button_rect(self) -> QRect:
        bar = self.drag_bar_rect()
        w, h = 22, min(18, max(14, bar.height() - 4))
        return QRect(bar.right() - w - 4, bar.top() + max(0, (bar.height() - h) // 2), w, h)

    def _title_bar_right_reserved_width(self) -> int:
        return self.close_button_rect().width() + 12

    def move_grip_rect(self) -> QRect:
        """顶栏中央窄条：唯一可移动区域（与四角缩放分离）。"""
        bar = self.drag_bar_rect()
        c = self.cfg.corner_handle_px
        reserved = self._title_bar_right_reserved_width()
        gw = min(self.cfg.move_grip_width, max(40, bar.width() - c * 2 - reserved))
        x = bar.left() + (bar.width() - reserved - gw) // 2
        return QRect(x, bar.top(), gw, bar.height())

    def _setup_title_bar_controls(self) -> None:
        self._btn_close = QPushButton("×", self)
        self._btn_close.setToolTip("退出 (Esc)；按住 Alt 可拖动/缩放")
        self._btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_close.clicked.connect(self._quit_application)
        px = max(12, self.cfg.font_settings_checkbox + 2)
        self._btn_close.setStyleSheet(
            "QPushButton {"
            f" color: #fff; background: rgba(170, 55, 55, 180); border: none;"
            f" border-radius: 4px; font-size: {px}px; font-weight: bold; padding: 0;"
            " }"
            "QPushButton:hover { background: rgba(210, 65, 65, 230); }"
            "QPushButton:pressed { background: rgba(130, 40, 40, 240); }"
        )
        self._position_title_bar_controls()
        self._update_frame_lock_state()

    def _update_frame_lock_state(self) -> None:
        interactive = self._frame_interactive()
        self._btn_close.setVisible(interactive)
        update_frame_lock_state(
            self,
            allowed=(self._btn_close,) if interactive else (),
            always_pass_through=(self._capture_spacer,),
        )

    def _position_title_bar_controls(self) -> None:
        self._btn_close.setGeometry(self.close_button_rect())
        self._btn_close.raise_()

    def _quit_application(self) -> None:
        QApplication.quit()

    def _frame_interactive(self) -> bool:
        return frame_interaction_allowed()

    def result_move_grip_rect(self) -> QRect:
        """结果区左侧窄条：可拖动整窗（仅主框文本模式）。"""
        if not self._show_text_panels:
            return QRect()
        rr = self.result_rect
        w = min(self.cfg.result_move_grip_width, self.width())
        return QRect(rr.left(), rr.top(), w, rr.height())

    @property
    def settings_rect(self) -> QRect:
        y = self._match_panel_height_active() + self._capture_h
        return QRect(0, y, self.width(), self._settings_panel_height_active())

    def _refresh_capture_region_info(self) -> None:
        hole = self.ocr_hole_rect()
        if hole.isEmpty() or hole.width() < 2 or hole.height() < 2:
            self._last_capture_region = None
            return
        self._last_capture_region = resolve_capture_region(self, hole)

    def ocr_hole_rect(self) -> QRect:
        """实际 OCR 截屏区域（拖动手柄下方镂空）。"""
        cr = self.capture_rect
        top = self.drag_bar_rect().height() + self._BORDER_PX
        if top >= cr.height() - self._BORDER_PX * 2:
            return QRect()
        return cr.adjusted(self._BORDER_PX, top, -self._BORDER_PX, -self._BORDER_PX)

    def _update_window_mask(self) -> None:
        """拖动手柄 + 边框可点；中间镂空用于透屏截屏。"""
        region = QRegion(self.rect())
        hole = self.ocr_hole_rect()
        if hole.width() > 4 and hole.height() > 4:
            region = region.subtracted(QRegion(hole))
        self.setMask(region)
        self._reset_size_constraints()

    def _start_ocr_timer(self) -> None:
        self._ocr_tick_timer = QTimer(self)
        self._ocr_tick_timer.setSingleShot(True)
        self._ocr_tick_timer.timeout.connect(self._on_ocr_tick)
        self._schedule_ocr_tick(self.cfg.interval_ms)

    def _schedule_ocr_tick(self, delay_ms: int) -> None:
        if not hasattr(self, "_ocr_tick_timer"):
            return
        self._ocr_tick_timer.stop()
        self._ocr_tick_timer.start(max(0, delay_ms))

    def _pause_ocr_timer(self) -> None:
        if hasattr(self, "_ocr_tick_timer"):
            self._ocr_tick_timer.stop()

    def _resume_ocr_timer_delayed(self) -> None:
        delay = max(0, self.cfg.resume_delay_ms)
        QTimer.singleShot(delay, self._resume_ocr_timer)

    def _resume_ocr_timer(self) -> None:
        if self._interacting:
            return
        self._schedule_ocr_tick(self.cfg.interval_ms)

    def _begin_interaction(self) -> None:
        if self._interacting:
            return
        self._interacting = True
        self._pause_ocr_timer()

    def _end_interaction(self) -> None:
        if not self._interacting:
            return
        self._interacting = False
        self._resume_ocr_timer_delayed()

    @property
    def capture_rect(self) -> QRect:
        y = self._match_panel_height_active()
        return QRect(0, y, self.width(), self._capture_h)

    @property
    def result_rect(self) -> QRect:
        y = self._match_panel_height_active() + self._capture_h + self._settings_panel_height_active()
        return QRect(0, y, self.width(), self._result_panel_height_active())

    def _corner_hit(self, pos: QPoint) -> ResizeMode:
        cr = self.capture_rect
        s = self.cfg.corner_handle_px
        x, y = pos.x(), pos.y()
        left = x <= cr.left() + s
        right = x >= cr.right() - s
        top = y <= cr.top() + s
        bottom = y >= cr.bottom() - s
        if left and top:
            return ResizeMode.LEFT | ResizeMode.TOP
        if right and top:
            return ResizeMode.RIGHT | ResizeMode.TOP
        if left and bottom:
            return ResizeMode.LEFT | ResizeMode.BOTTOM
        if right and bottom:
            return ResizeMode.RIGHT | ResizeMode.BOTTOM
        return ResizeMode.NONE

    def _edge_hit(self, pos: QPoint) -> ResizeMode:
        """左/右/底边中段可缩放；顶边仅四角，避免与移动握把冲突。"""
        cr = self.capture_rect
        m = self.cfg.resize_border_px
        c = self.cfg.corner_handle_px
        y0 = self.drag_bar_rect().bottom() + 2
        x, y = pos.x(), pos.y()
        mode = ResizeMode.NONE
        if cr.left() + c < x < cr.right() - c and y >= cr.bottom() - m:
            mode |= ResizeMode.BOTTOM
        if y0 + c < y < cr.bottom() - c and x <= cr.left() + m:
            mode |= ResizeMode.LEFT
        if y0 + c < y < cr.bottom() - c and x >= cr.right() - m:
            mode |= ResizeMode.RIGHT
        return mode

    def _hit_test(self, pos: QPoint) -> ResizeMode:
        if not self._frame_interactive():
            return ResizeMode.NONE
        if not self.capture_rect.contains(pos):
            return ResizeMode.NONE
        corner = self._corner_hit(pos)
        if corner != ResizeMode.NONE:
            return corner
        return self._edge_hit(pos)

    def _is_move_zone(self, pos: QPoint) -> bool:
        if not self._frame_interactive():
            return False
        if self.move_grip_rect().contains(pos):
            return True
        grip = self.result_move_grip_rect()
        return not grip.isEmpty() and grip.contains(pos)

    def _cursor_for_mode(self, mode: ResizeMode) -> Qt.CursorShape:
        if mode == (ResizeMode.LEFT | ResizeMode.TOP) or mode == (ResizeMode.RIGHT | ResizeMode.BOTTOM):
            return Qt.CursorShape.SizeFDiagCursor
        if mode == (ResizeMode.RIGHT | ResizeMode.TOP) or mode == (ResizeMode.LEFT | ResizeMode.BOTTOM):
            return Qt.CursorShape.SizeBDiagCursor
        if mode & (ResizeMode.LEFT | ResizeMode.RIGHT):
            return Qt.CursorShape.SizeHorCursor
        if mode & (ResizeMode.TOP | ResizeMode.BOTTOM):
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.ArrowCursor

    def paintEvent(self, _event) -> None:  # noqa: N802
        if not self._frame_interactive():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        capture = self.capture_rect
        bar = self.drag_bar_rect()
        grip = self.move_grip_rect()
        res_grip = self.result_move_grip_rect()

        interactive = self._frame_interactive()
        painter.fillRect(bar, QColor(25, 35, 50, 140))
        if interactive:
            painter.fillRect(grip, QColor(40, 140, 90, 200))
            if not res_grip.isEmpty():
                painter.fillRect(res_grip, QColor(40, 140, 90, 180))
            painter.setPen(QColor(230, 245, 235, 230))
        else:
            painter.fillRect(grip, QColor(50, 55, 65, 120))
            if not res_grip.isEmpty():
                painter.fillRect(res_grip, QColor(50, 55, 65, 100))
            painter.setPen(QColor(140, 145, 155, 180))
        painter.setFont(QFont("Microsoft YaHei UI", self.cfg.font_drag_grip))
        painter.drawText(grip, Qt.AlignmentFlag.AlignCenter, "拖动")
        if not res_grip.isEmpty() and interactive:
            painter.drawText(res_grip, Qt.AlignmentFlag.AlignCenter, "≡")

        pen = QPen(QColor(74, 158, 255, 230))
        pen.setWidth(self._BORDER_PX)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(capture.adjusted(1, 1, -2, -2))

        if self._frame_interactive():
            self._paint_resize_handles(painter, capture)

    def _paint_resize_handles(self, painter: QPainter, capture: QRect) -> None:
        c = self.cfg.corner_handle_px
        arm = min(14, max(8, c // 2))
        thick = 3
        handle_color = QColor(255, 160, 50, 230)
        painter.setPen(Qt.PenStyle.NoPen)

        for x0, y0, dx, dy in (
            (capture.left(), capture.top(), 1, 1),
            (capture.right(), capture.top(), -1, 1),
            (capture.left(), capture.bottom(), 1, -1),
            (capture.right(), capture.bottom(), -1, -1),
        ):
            hx = x0 + (0 if dx > 0 else -arm)
            hy = y0 + (0 if dy > 0 else -arm)
            painter.fillRect(QRect(hx, hy, arm, thick), handle_color)
            painter.fillRect(QRect(hx, hy, thick, arm), handle_color)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._position_title_bar_controls()
        # 拖拽缩放中不在 resizeEvent 里 setMask，否则 Windows 会不断抬高 minimumSize
        if not self._interacting:
            self._update_window_mask()
        if self._debug_mode_active():
            self._refresh_capture_region_info()
            self._update_settings_debug_text()

    def moveEvent(self, event) -> None:  # noqa: N802
        super().moveEvent(event)
        if self._debug_mode_active():
            self._refresh_capture_region_info()
            self._update_settings_debug_text()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position().toPoint()
        if not self._frame_interactive():
            return
        # 缩放优先（四角/边），再判定窄握把移动
        mode = self._hit_test(pos)
        if mode != ResizeMode.NONE:
            self._begin_interaction()
            self._resize_mode = mode
            self._drag_start_global = event.globalPosition().toPoint()
            self._geom_start = self.geometry()
            self.clearMask()
            self._reset_size_constraints()
            self.grabMouse()
            return

        if self._is_move_zone(pos):
            self._begin_interaction()
            self._moving = True
            self._drag_start_global = event.globalPosition().toPoint()
            self._geom_start = self.geometry()
            self.grabMouse()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position().toPoint()
        if self._resize_mode != ResizeMode.NONE:
            if self._frame_interactive():
                self._apply_resize(event.globalPosition().toPoint())
            return
        if self._moving:
            delta = event.globalPosition().toPoint() - self._drag_start_global
            moved = QRect(self._geom_start)
            moved.moveTopLeft(self._geom_start.topLeft() + delta)
            self.setGeometry(self._clamp_geometry(moved))
            return
        if not self._frame_interactive():
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        mode = self._hit_test(pos)
        if mode != ResizeMode.NONE:
            self.setCursor(self._cursor_for_mode(mode))
        elif self._is_move_zone(pos):
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        was_interacting = self._interacting
        self._resize_mode = ResizeMode.NONE
        self._moving = False
        if self.mouseGrabber() is self:
            self.releaseMouse()
        self.setCursor(Qt.CursorShape.ArrowCursor)
        if was_interacting:
            self._update_window_mask()
            if self._debug_mode_active():
                self._refresh_capture_region_info()
                self._update_settings_debug_text()
            self._end_interaction()

    def _apply_resize(self, global_pos: QPoint) -> None:
        delta = global_pos - self._drag_start_global
        start = self._geom_start
        min_w = self.cfg.min_capture_width
        min_h = self.cfg.min_capture_height
        chrome_h = self._chrome_height()
        start_capture_h = start.height() - chrome_h

        g = QRect(start)
        if self._resize_mode & ResizeMode.LEFT:
            g.setLeft(min(start.left() + delta.x(), start.right() - min_w + 1))
        if self._resize_mode & ResizeMode.RIGHT:
            g.setRight(max(start.right() + delta.x(), start.left() + min_w - 1))
        if self._resize_mode & ResizeMode.TOP:
            g.setTop(min(start.top() + delta.y(), start.bottom() - chrome_h - min_h + 1))
        if self._resize_mode & ResizeMode.BOTTOM:
            capture_h = max(min_h, start_capture_h + delta.y())
            g.setHeight(capture_h + chrome_h)

        self._capture_h = g.height() - chrome_h
        self._capture_w = g.width()
        g = self._clamp_geometry(g)
        self._capture_h = g.height() - chrome_h
        self._capture_w = g.width()

        self.setGeometry(g)
        w = g.width()
        self._capture_spacer.setFixedHeight(self._capture_h)
        for panel in (self._match_label, self._settings_panel, self._result_label):
            if panel is not None and panel.width() != w:
                panel.setFixedWidth(w)

    def _grab_capture_image(self) -> np.ndarray | None:
        capture = self.ocr_hole_rect()
        if capture.isEmpty() or capture.width() < 2 or capture.height() < 2:
            return None

        bgr, region = grab_capture_bgr(self, capture)
        if region is not None:
            self._last_capture_region = region
            if self._debug_mode_active():
                self._update_settings_debug_text()

        if bgr is None:
            return None

        if self._runtime_save_capture and region is not None:
            save_debug_image(bgr, self.cfg.debug_capture_path)
            meta_path = self.cfg.debug_capture_path.with_suffix(".txt")
            save_debug_metadata(region, meta_path)
        return bgr

    def _release_worker(self) -> None:
        self._ocr_busy = False
        worker = self._worker
        self._worker = None
        if worker is None:
            return
        if worker.isRunning():
            worker.wait(500)
        worker.deleteLater()

    def shutdown_ocr(self) -> None:
        """停止定时器并等待 OCR 线程结束（关闭窗口 / 退出应用时调用）。"""
        if hasattr(self, "_ocr_tick_timer"):
            self._ocr_tick_timer.stop()
        if hasattr(self, "_debug_timer"):
            self._debug_timer.stop()
        worker = self._worker
        self._worker = None
        self._ocr_busy = False
        if worker is None:
            return
        if worker.isRunning():
            worker.wait(15_000)
        worker.deleteLater()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_window_geometry()
        self.shutdown_ocr()
        if self._icon_panel is not None:
            self._icon_panel.close()
        super().closeEvent(event)

    def _on_ocr_tick(self) -> None:
        if self._interacting or self._ocr_busy:
            self._schedule_ocr_tick(self.cfg.interval_ms)
            return
        if self._worker is not None and self._worker.isRunning():
            self._schedule_ocr_tick(self.cfg.interval_ms)
            return
        image = self._grab_capture_image()
        if image is None:
            self._update_icon_panel(None)
            if self._match_label is not None and self._result_label is not None:
                self._match_label.setText("（截屏失败）")
                self._result_label.setText("（截屏失败：请安装 mss 或检查显示缩放）")
            self._schedule_ocr_tick(self.cfg.interval_ms)
            return
        self._ocr_busy = True
        worker = OcrWorker(self, self._engine, image)
        self._worker = worker
        worker.finished.connect(self._on_ocr_done)
        worker.failed.connect(self._on_ocr_error)
        worker.start()

    def _update_icon_panel(self, result: MatchResult | None) -> None:
        if self._icon_panel is None:
            return
        self._icon_panel.update_match_result(result)

    def _apply_match_and_ocr_panels(self, text: str) -> None:
        raw = text.strip()
        if self._matcher is None:
            self._update_icon_panel(None)
            if self._match_label is not None and self._result_label is not None:
                self._match_label.setText("")
                self._result_label.setText(raw if raw else "（未识别到文字）")
            return

        result = self._matcher.match(raw)
        self._last_match_result = result
        self._update_icon_panel(result)
        if self._match_label is not None and self._result_label is not None:
            self._match_label.setText(result.format_match_list())
            if self.cfg.match_show_ocr:
                self._result_label.setText(result.format_ocr_text())
            else:
                self._result_label.setText("")

    def _finish_ocr_cycle(self) -> None:
        if not self._interacting:
            self._schedule_ocr_tick(self.cfg.interval_ms)

    def _on_ocr_done(self, text: str) -> None:
        try:
            raw = text.strip()
            if raw:
                self._apply_match_and_ocr_panels(text)
            else:
                self._update_icon_panel(None)
                if self._match_label is not None and self._result_label is not None:
                    if self._runtime_save_capture:
                        self._match_label.setText("（未匹配）")
                        self._result_label.setText("（未识别到文字，请查看 debug_last_capture.png）")
                    else:
                        self._match_label.setText("（未匹配）")
                        self._result_label.setText("（未识别到文字）")
        finally:
            self._release_worker()
            self._finish_ocr_cycle()

    def _on_ocr_error(self, message: str) -> None:
        try:
            self._update_icon_panel(None)
            if self._result_label is not None:
                self._result_label.setText(f"识别失败: {message}")
        finally:
            self._release_worker()
            self._finish_ocr_cycle()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() == Qt.Key.Key_Q and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            QApplication.quit()


def run(config_path: Path | None = None) -> int:
    try:
        import mss  # noqa: F401 — 须在 QApplication 之前
    except ImportError:
        pass

    cfg = load_config(config_path) if config_path else load_config()
    app = QApplication(sys.argv)
    window = OcrOverlayWindow(cfg)
    lock_targets: list = [window]
    if window._icon_panel is not None:
        lock_targets.append(window._icon_panel)
    window._alt_monitor = install_alt_key_monitor(app, lock_targets, window._lock_hub)
    window._lock_hub.interaction_changed.connect(window._update_frame_lock_state)
    if window._icon_panel is not None:
        window._lock_hub.interaction_changed.connect(
            window._icon_panel._update_frame_lock_state
        )
    app.aboutToQuit.connect(window._save_window_geometry)
    app.aboutToQuit.connect(window.shutdown_ocr)
    if window._icon_panel is not None:
        app.aboutToQuit.connect(window._icon_panel.close)
    window.show()
    return app.exec()
