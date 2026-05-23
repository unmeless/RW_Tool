from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent.parent / "desc.json"


@dataclass(frozen=True)
class PetEntry:
    name: str
    description: str


def _parse_pet_item(raw: Any, *, source: str, index: int) -> PetEntry:
    if not isinstance(raw, dict):
        raise ValueError(f"{source} 第 {index} 项应为对象，含 name 与 description")
    name = raw.get("name")
    description = raw.get("description")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{source} 第 {index} 项缺少非空 name")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{source} 第 {index} 项缺少非空 description")
    return PetEntry(name=name.strip(), description=description.strip())


def _extract_pet_list(data: Any, *, source: str) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("pets"), list):
        return data["pets"]
    raise ValueError(f"{source} 格式错误：应为 {{\"pets\": [...]}} 或 [...]")


def load_catalog(path: Path | None = None) -> list[PetEntry]:
    catalog_path = path or DEFAULT_CATALOG_PATH
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"图鉴 JSON 解析失败: {catalog_path}") from exc

    raw_items = _extract_pet_list(data, source=str(catalog_path))
    entries: list[PetEntry] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_items, start=1):
        entry = _parse_pet_item(raw, source=str(catalog_path), index=i)
        if entry.name in seen:
            raise ValueError(f"{catalog_path} 中存在重复名称: {entry.name}")
        seen.add(entry.name)
        entries.append(entry)

    if not entries:
        raise ValueError(f"图鉴为空: {catalog_path}")
    return entries
