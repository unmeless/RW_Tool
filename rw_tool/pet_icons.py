from __future__ import annotations

from pathlib import Path

_ICON_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")


def resolve_pet_icon_path(icon_dir: Path, pet_name: str) -> Path | None:
    """
    按小动物名称查找图标。

    支持：
    - img/名称.png
    - img/名称/ 目录下任意一张图
    """
    if not pet_name or not icon_dir.is_dir():
        return None

    for ext in _ICON_EXTS:
        path = icon_dir / f"{pet_name}{ext}"
        if path.is_file():
            return path

    sub = icon_dir / pet_name
    if sub.is_dir():
        for path in sorted(sub.iterdir()):
            if path.is_file() and path.suffix.lower() in _ICON_EXTS:
                return path

    return None
