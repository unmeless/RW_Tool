from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict


class OverlayState(TypedDict, total=False):
    x: int
    y: int
    w: int
    h: int
    capture_w: int
    capture_h: int


class IconPanelState(TypedDict, total=False):
    x: int
    y: int
    w: int
    h: int


class WindowState(TypedDict, total=False):
    overlay: OverlayState
    icon_panel: IconPanelState


def state_path_for_config(config_path: Path) -> Path:
    return config_path.parent / "window_state.json"


def load_window_state(path: Path) -> WindowState | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data  # type: ignore[return-value]


def save_window_state(
    path: Path,
    *,
    overlay: OverlayState | None = None,
    icon_panel: IconPanelState | None = None,
) -> None:
    payload: dict[str, Any] = load_window_state(path) or {}
    if overlay is not None:
        payload["overlay"] = overlay
    if icon_panel is not None:
        payload["icon_panel"] = icon_panel
    if not payload:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
