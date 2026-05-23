from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.ini"


@dataclass(frozen=True)
class AppConfig:
    frequency_hz: float
    resize_border_px: int
    corner_handle_px: int
    drag_bar_height: int
    move_grip_width: int
    result_move_grip_width: int
    min_capture_width: int
    min_capture_height: int
    restore_saved_geometry: bool
    result_panel_height: int
    match_panel_height: int
    match_top_k: int
    ocr_backend: str
    preprocess_scale: float
    preprocess_mode: str
    dual_preprocess: bool
    ocr_max_side: int
    ocr_layout: str
    ocr_strip_max_lines: int
    ocr_use_angle_cls: bool
    resume_delay_ms: int
    det_box_thresh: float
    text_score: float
    config_path: Path
    debug_enabled: bool
    debug_save_capture: bool
    debug_capture_path: Path
    settings_panel_height: int
    font_match: int
    font_ocr_result: int
    font_drag_grip: int
    font_settings_title: int
    font_settings_debug: int
    font_settings_checkbox: int
    matcher_enabled: bool
    catalog_path: Path
    match_min_score: float
    match_min_margin: float
    match_min_candidate_score: float
    match_show_ocr: bool
    icon_dir: Path
    icon_panel_enabled: bool
    icon_size: int
    icon_panel_height: int
    font_icon_score: int
    icon_cell_width: int
    icon_cell_spacing: int
    icon_panel_padding: int
    icon_panel_bg_opacity: float
    icon_panel_min_height: int
    icon_show_score: bool

    @property
    def interval_ms(self) -> int:
        hz = max(self.frequency_hz, 0.1)
        return max(1, int(1000 / hz))

    def icon_panel_width_for(self, slots: int | None = None) -> int:
        """按候选槽位数计算图标条宽度（默认 match_top_k）。"""
        k = max(1, slots if slots is not None else self.match_top_k)
        return k * self.icon_cell_width + max(0, k - 1) * self.icon_cell_spacing + self.icon_panel_padding


def load_config(path: Path | None = None) -> AppConfig:
    cfg_path = path or DEFAULT_CONFIG_PATH
    parser = configparser.ConfigParser()
    if not parser.read(cfg_path, encoding="utf-8"):
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

    ocr = parser["ocr"]
    window = parser["window"]
    settings_panel_height = 88
    font_match = 26
    font_ocr_result = 13
    font_drag_grip = 8
    font_settings_title = 11
    font_settings_debug = 10
    font_settings_checkbox = 11
    if parser.has_section("settings"):
        st = parser["settings"]
        settings_panel_height = int(st.get("panel_height", "88"))
        font_match = int(st.get("font_match", "26"))
        font_ocr_result = int(st.get("font_ocr_result", "13"))
        font_drag_grip = int(st.get("font_drag_grip", "8"))
        font_settings_title = int(st.get("font_settings_title", "11"))
        font_settings_debug = int(st.get("font_settings_debug", "10"))
        font_settings_checkbox = int(st.get("font_settings_checkbox", "11"))
    ocr_backend = "rapidocr"
    det_box_thresh = 0.2
    text_score = 0.3
    if parser.has_section("engine"):
        engine = parser["engine"]
        ocr_backend = engine.get("backend", "rapidocr").strip().lower()
        det_box_thresh = float(engine.get("det_box_thresh", "0.2"))
        text_score = float(engine.get("text_score", "0.3"))

    debug_path = Path(ocr.get("debug_capture_path", "debug_last_capture.png"))
    if not debug_path.is_absolute():
        debug_path = cfg_path.parent / debug_path

    matcher_enabled = True
    catalog_path = cfg_path.parent / "desc.json"
    match_min_score = 58.0
    match_min_margin = 8.0
    match_min_candidate_score = 50.0
    match_show_ocr = True
    match_top_k = 5
    icon_dir = cfg_path.parent / "img"
    icon_panel_enabled = True
    icon_size = 56
    icon_panel_height = 110
    font_icon_score = 16
    icon_cell_width = 84
    icon_cell_spacing = 8
    icon_panel_padding = 16
    icon_panel_bg_opacity = 0.3
    icon_panel_min_height = 90
    icon_show_score = False
    if parser.has_section("matcher"):
        m = parser["matcher"]
        matcher_enabled = m.getboolean("enabled", fallback=True)
        catalog_path = Path(m.get("catalog_path", "desc.json"))
        if not catalog_path.is_absolute():
            catalog_path = cfg_path.parent / catalog_path
        match_min_score = float(m.get("min_score", "58"))
        match_min_margin = float(m.get("min_margin", "8"))
        match_min_candidate_score = float(m.get("min_candidate_score", "50"))
        match_show_ocr = m.getboolean("show_ocr_raw", fallback=True)
        match_top_k = int(m.get("match_top_k", "5"))
        icon_dir = Path(m.get("icon_dir", "img"))
        if not icon_dir.is_absolute():
            icon_dir = cfg_path.parent / icon_dir
    if parser.has_section("icon_panel"):
        ip = parser["icon_panel"]
        icon_panel_enabled = ip.getboolean("enabled", fallback=True)
        icon_size = int(ip.get("icon_size", "56"))
        icon_panel_height = int(ip.get("panel_height", "110"))
        font_icon_score = int(ip.get("font_score", "16"))
        icon_cell_width = int(ip.get("cell_width", "84"))
        icon_cell_spacing = int(ip.get("cell_spacing", "8"))
        icon_panel_padding = int(ip.get("panel_padding", "16"))
        icon_panel_bg_opacity = float(ip.get("bg_opacity", "0.3"))
        icon_panel_min_height = int(ip.get("min_height", "90"))
        icon_show_score = ip.getboolean("show_score", fallback=False)

    debug_save = ocr.getboolean("debug_save_capture", fallback=False)
    debug_show = ocr.getboolean("debug_show_coords", fallback=False)
    debug_master = ocr.getboolean("debug", fallback=False)
    debug_enabled = debug_master or debug_save or debug_show

    return AppConfig(
        config_path=cfg_path,
        debug_enabled=debug_enabled,
        frequency_hz=float(ocr.get("frequency_hz", "1.0")),
        resize_border_px=int(window.get("resize_border_px", "16")),
        corner_handle_px=int(window.get("corner_handle_px", "22")),
        drag_bar_height=int(window.get("drag_bar_height", "26")),
        move_grip_width=int(window.get("move_grip_width", "100")),
        result_move_grip_width=int(window.get("result_move_grip_width", "36")),
        min_capture_width=int(window.get("min_capture_width", "120")),
        min_capture_height=int(window.get("min_capture_height", "80")),
        restore_saved_geometry=window.getboolean("restore_saved_geometry", fallback=True),
        result_panel_height=int(window.get("result_panel_height", "100")),
        match_panel_height=int(window.get("match_panel_height", "84")),
        match_top_k=match_top_k,
        ocr_backend=ocr_backend,
        preprocess_scale=float(ocr.get("preprocess_scale", "2.5")),
        preprocess_mode=ocr.get("preprocess_mode", "game_ui").strip(),
        dual_preprocess=ocr.getboolean("dual_preprocess", fallback=False),
        ocr_max_side=int(ocr.get("ocr_max_side", "1280")),
        ocr_layout=ocr.get("ocr_layout", "strip").strip().lower(),
        ocr_strip_max_lines=max(1, min(3, int(ocr.get("ocr_strip_max_lines", "2")))),
        ocr_use_angle_cls=ocr.getboolean("use_angle_cls", fallback=False),
        resume_delay_ms=int(ocr.get("resume_delay_ms", "400")),
        det_box_thresh=det_box_thresh,
        text_score=text_score,
        debug_save_capture=debug_save,
        debug_capture_path=debug_path,
        settings_panel_height=settings_panel_height,
        font_match=font_match,
        font_ocr_result=font_ocr_result,
        font_drag_grip=font_drag_grip,
        font_settings_title=font_settings_title,
        font_settings_debug=font_settings_debug,
        font_settings_checkbox=font_settings_checkbox,
        matcher_enabled=matcher_enabled,
        catalog_path=catalog_path,
        match_min_score=match_min_score,
        match_min_margin=match_min_margin,
        match_min_candidate_score=match_min_candidate_score,
        match_show_ocr=match_show_ocr,
        icon_dir=icon_dir,
        icon_panel_enabled=icon_panel_enabled and matcher_enabled,
        icon_size=icon_size,
        icon_panel_height=icon_panel_height,
        font_icon_score=font_icon_score,
        icon_cell_width=icon_cell_width,
        icon_cell_spacing=icon_cell_spacing,
        icon_panel_padding=icon_panel_padding,
        icon_panel_bg_opacity=icon_panel_bg_opacity,
        icon_panel_min_height=icon_panel_min_height,
        icon_show_score=icon_show_score,
    )
