from __future__ import annotations

from enum import IntFlag, auto
from pathlib import Path

from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from rw_tool.config import AppConfig
from rw_tool.frame_lock import (
    apply_lock_mouse_policy,
    frame_interaction_allowed,
    frame_locked,
    set_window_input_passthrough,
)
from rw_tool.pet_icons import resolve_pet_icon_path
from rw_tool.pet_matcher import MatchCandidate, MatchResult

_SCALE_MIN = 0.45
_SCALE_MAX = 2.5
_ROW_MARGIN_H = 6
_ROW_MARGIN_B = 4


class ResizeMode(IntFlag):
    NONE = 0
    LEFT = auto()
    RIGHT = auto()
    TOP = auto()
    BOTTOM = auto()


class _PetIconCell(QWidget):
    """单个候选：图标 + 名称 + 概率（随窗口缩放）。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._icon_size = 56
        self._cell_width = 84
        self._font_score = 16
        self._icon_path: Path | None = None
        self._candidate: MatchCandidate | None = None
        self._rank = 0
        self._confident = False
        self._show_score = False

        self._icon = QLabel()
        self._icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon.setStyleSheet(
            "background: rgba(40, 44, 56, 120); border-radius: 6px;"
        )

        self._name = QLabel()
        self._name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name.setStyleSheet("color: #e8e8ec; font-weight: bold; background: transparent;")

        self._score = QLabel()
        self._score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._score.setStyleSheet("color: #ffdc78; background: transparent;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)
        layout.addWidget(self._icon, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._name)
        layout.addWidget(self._score)
        self._score.setVisible(False)

    def apply_metrics(
        self,
        icon_size: int,
        cell_width: int,
        font_score: int,
        *,
        show_score: bool = False,
    ) -> None:
        self._icon_size = icon_size
        self._cell_width = cell_width
        self._font_score = font_score
        self._show_score = show_score
        self.setFixedWidth(cell_width)
        self._icon.setFixedSize(icon_size, icon_size)
        radius = max(3, int(6 * icon_size / 56))
        self._icon.setStyleSheet(
            f"background: rgba(40, 44, 56, 120); border-radius: {radius}px;"
        )
        name_px = max(8, font_score - 1)
        self._name.setStyleSheet(
            f"color: #e8e8ec; font-weight: bold; font-size: {name_px}px;"
            " background: transparent;"
        )
        self._score.setStyleSheet(
            f"color: #ffdc78; font-size: {font_score}px; background: transparent;"
        )
        self._score.setVisible(show_score)
        if self._candidate is not None:
            self._refresh_score_text()
            self._refresh_icon()

    def set_candidate(
        self,
        candidate: MatchCandidate,
        *,
        icon_path: Path | None,
        rank: int,
        confident: bool,
    ) -> None:
        self._candidate = candidate
        self._icon_path = icon_path
        self._rank = rank
        self._confident = confident

        name = candidate.name
        if rank == 0 and not confident:
            title = f"? {name}"
        else:
            title = name
        self._name.setText(title)
        self._refresh_score_text()
        self._refresh_icon()

    def _refresh_score_text(self) -> None:
        if self._candidate is None or not self._show_score:
            self._score.setText("")
            return
        self._score.setText(f"{self._candidate.score:.0f}%")

    def _refresh_icon(self) -> None:
        if self._candidate is None:
            return
        name = self._candidate.name
        pixmap = QPixmap()
        if self._icon_path is not None and pixmap.load(str(self._icon_path)):
            scaled = pixmap.scaled(
                self._icon_size,
                self._icon_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._icon.setText("")
            self._icon.setPixmap(scaled)
            radius = max(3, int(6 * self._icon_size / 56))
            self._icon.setStyleSheet(
                f"background: rgba(40, 44, 56, 120); border-radius: {radius}px;"
            )
        else:
            self._icon.setPixmap(QPixmap())
            self._icon.setText(name[:1] if name else "?")
            fallback_px = max(10, int(18 * self._icon_size / 56))
            radius = max(3, int(6 * self._icon_size / 56))
            self._icon.setStyleSheet(
                f"background: rgba(40, 44, 56, 120); border-radius: {radius}px;"
                f" color: #8ec8ff; font-size: {fallback_px}px; font-weight: bold;"
            )


class PetIconPanelWindow(QWidget):
    """独立可拖动、可缩放的匹配图标条（置顶，30% 底色透明度）。"""

    _DRAG_BAR_H = 22
    _BORDER_PX = 2

    def __init__(
        self,
        config: AppConfig,
        icon_dir: Path,
    ) -> None:
        super().__init__()
        self.cfg = config
        self._icon_dir = icon_dir
        self._moving = False
        self._resizing = ResizeMode.NONE
        self._drag_start = QPoint()
        self._geom_start = QRect()
        self._scale = 1.0
        self._last_result: MatchResult | None = None

        self._base_window_flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setWindowFlags(self._base_window_flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self._cells: list[_PetIconCell] = []
        self._row = QWidget()
        self._row.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._row_layout = QHBoxLayout(self._row)
        self._row_layout.setContentsMargins(_ROW_MARGIN_H, 0, _ROW_MARGIN_H, _ROW_MARGIN_B)
        self._row_layout.setSpacing(config.icon_cell_spacing)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, self._DRAG_BAR_H, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._row)

        self._apply_panel_size(config.match_top_k)
        self._reset_size_constraints()
        self._clear_icon_content()
        self._update_frame_lock_state()

    def _bg_alpha(self) -> int:
        return max(0, min(255, int(round(255 * self.cfg.icon_panel_bg_opacity))))

    def _content_rect(self) -> QRect:
        return QRect(
            0,
            self._DRAG_BAR_H,
            self.width(),
            max(1, self.height() - self._DRAG_BAR_H),
        )

    def _active_slots(self) -> int:
        if self._cells:
            return len(self._cells)
        return self.cfg.match_top_k

    def _baseline_content_height(self) -> int:
        return max(1, self.cfg.icon_panel_height - self._DRAG_BAR_H)

    def _width_for_slots(self, slots: int, scale: float = 1.0) -> int:
        k = max(1, slots)
        base = self.cfg.icon_panel_width_for(k)
        return max(int(base * scale), int(self.cfg.icon_panel_width_for(1) * _SCALE_MIN))

    def _metrics_at(self, scale: float) -> dict[str, int]:
        s = max(_SCALE_MIN, min(_SCALE_MAX, scale))
        return {
            "icon_size": max(16, int(round(self.cfg.icon_size * s))),
            "cell_width": max(32, int(round(self.cfg.icon_cell_width * s))),
            "font_score": max(8, int(round(self.cfg.font_icon_score * s))),
            "spacing": max(2, int(round(self.cfg.icon_cell_spacing * s))),
            "margin_h": max(2, int(round(_ROW_MARGIN_H * s))),
            "margin_b": max(2, int(round(_ROW_MARGIN_B * s))),
        }

    def _effective_scale(self, resize_hint: ResizeMode = ResizeMode.NONE) -> float:
        slots = self._active_slots()
        base_w = max(1, self.cfg.icon_panel_width_for(slots))
        base_h = self._baseline_content_height()
        scale_w = self.width() / base_w
        scale_h = (max(1, self.height() - self._DRAG_BAR_H)) / base_h

        horiz = bool(resize_hint & (ResizeMode.LEFT | ResizeMode.RIGHT))
        vert = bool(resize_hint & (ResizeMode.TOP | ResizeMode.BOTTOM))
        if horiz and not vert:
            raw = scale_w
        elif vert and not horiz:
            raw = scale_h
        else:
            raw = min(scale_w, scale_h)
        return max(_SCALE_MIN, min(_SCALE_MAX, raw))

    def _sync_content_scale(self, resize_hint: ResizeMode = ResizeMode.NONE) -> None:
        scale = self._effective_scale(resize_hint)
        self._scale = scale
        m = self._metrics_at(scale)
        self._row_layout.setSpacing(m["spacing"])
        self._row_layout.setContentsMargins(m["margin_h"], 0, m["margin_h"], m["margin_b"])

        for cell in self._cells:
            cell.apply_metrics(
                m["icon_size"],
                m["cell_width"],
                m["font_score"],
                show_score=self.cfg.icon_show_score,
            )

    def _apply_panel_size(self, slots: int) -> None:
        w = self._width_for_slots(slots)
        h = self.cfg.icon_panel_height
        self.resize(w, h)
        self._reset_size_constraints()
        self._sync_content_scale()

    def _min_panel_size(self) -> QSize:
        scale = _SCALE_MIN
        slots = self._active_slots()
        w = self._width_for_slots(slots, scale)
        extra = 36 if self.cfg.icon_show_score else 22
        h = max(
            int(self.cfg.icon_panel_min_height * scale),
            self._DRAG_BAR_H + self._metrics_at(scale)["icon_size"] + extra,
        )
        return QSize(w, h)

    def _reset_size_constraints(self) -> None:
        self.setMinimumSize(self._min_panel_size())
        self.setMaximumSize(16777215, 16777215)

    def _clear_icon_content(self) -> None:
        self._clear_cells()

    def _clear_cells(self) -> None:
        while self._row_layout.count():
            item = self._row_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._cells.clear()

    def apply_saved_geometry(self, x: int, y: int, w: int, h: int) -> None:
        self.setGeometry(
            x,
            y,
            max(self._min_panel_size().width(), w),
            max(self.cfg.icon_panel_min_height, h),
        )
        self._reset_size_constraints()
        self._sync_content_scale()
        if self._last_result is not None:
            self.update_match_result(self._last_result)

    def update_match_result(self, result: MatchResult | None) -> None:
        self._last_result = result
        slots = self.cfg.match_top_k

        if result is None or not result.candidates:
            self._clear_icon_content()
            return

        self._clear_cells()
        scale = self._effective_scale()
        m = self._metrics_at(scale)

        for i, cand in enumerate(result.candidates[:slots]):
            cell = _PetIconCell(self._row)
            cell.apply_metrics(
                m["icon_size"],
                m["cell_width"],
                m["font_score"],
                show_score=self.cfg.icon_show_score,
            )
            icon_path = resolve_pet_icon_path(self._icon_dir, cand.name)
            cell.set_candidate(
                cand,
                icon_path=icon_path,
                rank=i,
                confident=result.confident and i == 0,
            )
            self._row_layout.addWidget(cell)
            self._cells.append(cell)

        self._sync_content_scale()
        self._reset_size_constraints()

    def drag_bar_rect(self) -> QRect:
        return QRect(0, 0, self.width(), self._DRAG_BAR_H)

    def _update_frame_lock_state(self) -> None:
        """默认锁定；按住 Alt 时解锁。图标区保持可见。"""
        locked = frame_locked()
        apply_lock_mouse_policy(
            self,
            allowed=(),
            always_pass_through=(self._row,),
            locked=locked,
        )
        set_window_input_passthrough(
            self,
            locked,
            base_flags=self._base_window_flags,
        )
        self.setMouseTracking(not locked)
        self.update()

    def _frame_interactive(self) -> bool:
        return frame_interaction_allowed()

    def move_grip_rect(self) -> QRect:
        bar = self.drag_bar_rect()
        c = self.cfg.corner_handle_px
        gw = min(self.cfg.move_grip_width, max(40, bar.width() - c * 2))
        x = bar.left() + (bar.width() - gw) // 2
        return QRect(x, bar.top(), gw, bar.height())

    def _corner_hit(self, pos: QPoint) -> ResizeMode:
        cr = self._content_rect()
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
        cr = self._content_rect()
        m = self.cfg.resize_border_px
        c = self.cfg.corner_handle_px
        x, y = pos.x(), pos.y()
        mode = ResizeMode.NONE
        if cr.left() + c < x < cr.right() - c and y >= cr.bottom() - m:
            mode |= ResizeMode.BOTTOM
        if y > cr.top() + c and y < cr.bottom() - c and x <= cr.left() + m:
            mode |= ResizeMode.LEFT
        if y > cr.top() + c and y < cr.bottom() - c and x >= cr.right() - m:
            mode |= ResizeMode.RIGHT
        return mode

    def _hit_test(self, pos: QPoint) -> ResizeMode:
        if not self._frame_interactive():
            return ResizeMode.NONE
        if not self._content_rect().contains(pos):
            return ResizeMode.NONE
        corner = self._corner_hit(pos)
        if corner != ResizeMode.NONE:
            return corner
        return self._edge_hit(pos)

    def _apply_resize(self, global_pos: QPoint) -> None:
        delta = global_pos - self._drag_start
        start = self._geom_start
        min_sz = self._min_panel_size()
        g = QRect(start)
        if self._resizing & ResizeMode.LEFT:
            g.setLeft(min(start.left() + delta.x(), start.right() - min_sz.width() + 1))
        if self._resizing & ResizeMode.RIGHT:
            g.setRight(max(start.right() + delta.x(), start.left() + min_sz.width() - 1))
        if self._resizing & ResizeMode.TOP:
            g.setTop(min(start.top() + delta.y(), start.bottom() - min_sz.height() + 1))
        if self._resizing & ResizeMode.BOTTOM:
            g.setBottom(max(start.bottom() + delta.y(), start.top() + min_sz.height() - 1))
        self.setGeometry(g)
        self._sync_content_scale(self._resizing)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        alpha = self._bg_alpha()

        painter.fillRect(self.rect(), QColor(20, 20, 24, alpha))

        # 锁定：保留底色与图标/文字，不画边框与拖动/缩放装饰
        if not self._frame_interactive():
            return

        bar = self.drag_bar_rect()
        interactive = self._frame_interactive()
        painter.fillRect(bar, QColor(25, 35, 50, min(255, alpha + 40)))
        grip = self.move_grip_rect()
        if interactive:
            painter.fillRect(grip, QColor(40, 140, 90, min(255, alpha + 60)))
            painter.setPen(QColor(230, 245, 235, 230))
        else:
            painter.fillRect(grip, QColor(50, 55, 65, min(255, alpha + 20)))
            painter.setPen(QColor(140, 145, 155, 180))
        painter.setFont(QFont("Microsoft YaHei UI", self.cfg.font_drag_grip))
        painter.drawText(grip, Qt.AlignmentFlag.AlignCenter, "拖动")

        cr = self._content_rect()
        pen = QPen(QColor(74, 158, 255, 200))
        pen.setWidth(self._BORDER_PX)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(cr.adjusted(1, 1, -2, -2))

        if self._frame_interactive():
            self._paint_resize_handles(painter, cr)

    def _paint_resize_handles(self, painter: QPainter, rect: QRect) -> None:
        c = self.cfg.corner_handle_px
        arm = min(12, max(6, c // 2))
        thick = 3
        color = QColor(255, 160, 50, 220)
        painter.setPen(Qt.PenStyle.NoPen)
        for x0, y0, dx, dy in (
            (rect.left(), rect.top(), 1, 1),
            (rect.right(), rect.top(), -1, 1),
            (rect.left(), rect.bottom(), 1, -1),
            (rect.right(), rect.bottom(), -1, -1),
        ):
            hx = x0 + (0 if dx > 0 else -arm)
            hy = y0 + (0 if dy > 0 else -arm)
            painter.fillRect(QRect(hx, hy, arm, thick), color)
            painter.fillRect(QRect(hx, hy, thick, arm), color)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if frame_locked():
            self._update_frame_lock_state()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position().toPoint()
        if not self._frame_interactive():
            return
        mode = self._hit_test(pos)
        if mode != ResizeMode.NONE:
            self._resizing = mode
            self._drag_start = event.globalPosition().toPoint()
            self._geom_start = self.geometry()
            self.grabMouse()
            return
        if self.move_grip_rect().contains(pos):
            self._moving = True
            self._drag_start = event.globalPosition().toPoint()
            self._geom_start = self.geometry()
            self.grabMouse()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self._frame_interactive():
            return
        pos = event.position().toPoint()
        if self._resizing != ResizeMode.NONE:
            if self._frame_interactive():
                self._apply_resize(event.globalPosition().toPoint())
            return
        if self._moving:
            delta = event.globalPosition().toPoint() - self._drag_start
            g = QRect(self._geom_start)
            g.moveTopLeft(self._geom_start.topLeft() + delta)
            self.setGeometry(g)
            return
        mode = self._hit_test(pos)
        if mode != ResizeMode.NONE:
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif self.move_grip_rect().contains(pos):
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._resizing != ResizeMode.NONE:
            self._sync_content_scale(self._resizing)
        self._moving = False
        self._resizing = ResizeMode.NONE
        if self.mouseGrabber() is self:
            self.releaseMouse()
        self.setCursor(Qt.CursorShape.ArrowCursor)
