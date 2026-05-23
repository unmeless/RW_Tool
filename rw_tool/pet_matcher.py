from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from rw_tool.pet_catalog import PetEntry, load_catalog

# 常见 OCR 误识别 → 纠正（可按游戏内实际错字继续补充）
_OCR_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("间起来", "闻起来"),
    ("摩擦股", "摩擦的"),
    ("窸率", "窸窣"),
    ("瞬啪", "噼啪"),
    ("鼻出", "冒出"),
    ("朵柔", "朵朵"),
    ("罪近", "一靠近"),
    ("记彼", "起彼"),
    ("喜出着", "点击查看"),
    ("(", "（"),
    (")", "）"),
)

_PUNCT_RE = re.compile(r"[\s\u3000，。！？、；：:\"'（）()\[\]【】—\-·…]+")


def normalize_text(text: str) -> str:
    s = text.strip()
    for old, new in _OCR_REPLACEMENTS:
        s = s.replace(old, new)
    s = _PUNCT_RE.sub("", s)
    return s


def _bigram_jaccard_percent(a: str, b: str) -> float:
    if len(a) < 2 or len(b) < 2:
        return 100.0 if a == b else 0.0
    set_a = {a[i : i + 2] for i in range(len(a) - 1)}
    set_b = {b[i : i + 2] for i in range(len(b) - 1)}
    union = set_a | set_b
    if not union:
        return 0.0
    return 100.0 * len(set_a & set_b) / len(union)


def _sequence_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio() * 100.0


def _fuzz_scores(ocr_norm: str, desc_norm: str) -> tuple[float, float, float]:
    try:
        from rapidfuzz import fuzz
    except ImportError:
        ratio = _sequence_ratio(ocr_norm, desc_norm)
        return ratio, ratio, ratio
    return (
        float(fuzz.ratio(ocr_norm, desc_norm)),
        float(fuzz.partial_ratio(ocr_norm, desc_norm)),
        float(fuzz.token_set_ratio(ocr_norm, desc_norm)),
    )


def score_pair(ocr_norm: str, desc_norm: str) -> float:
    if not ocr_norm or not desc_norm:
        return 0.0
    ratio, partial, token_set = _fuzz_scores(ocr_norm, desc_norm)
    bigram = _bigram_jaccard_percent(ocr_norm, desc_norm)
    # 描述较长、OCR 为片段时 partial 权重更高
    if len(ocr_norm) < len(desc_norm) * 0.85:
        return 0.15 * ratio + 0.45 * partial + 0.25 * token_set + 0.15 * bigram
    return 0.25 * ratio + 0.25 * partial + 0.35 * token_set + 0.15 * bigram


@dataclass(frozen=True)
class MatchCandidate:
    name: str
    score: float
    description: str


@dataclass(frozen=True)
class MatchResult:
    best: MatchCandidate | None
    candidates: tuple[MatchCandidate, ...]
    ocr_raw: str
    confident: bool

    def format_match_list(self) -> str:
        """匹配候选列表（供扫描框上方展示）。"""
        if not self.candidates:
            return "（未匹配到小动物）"
        lines: list[str] = []
        for i, c in enumerate(self.candidates):
            if i == 0:
                if self.confident:
                    lines.append(f"【{c.name}】  {c.score:.0f}%")
                else:
                    lines.append(f"【? {c.name}】  {c.score:.0f}%")
            else:
                lines.append(f"{c.name}  {c.score:.0f}%")
        return "\n".join(lines)

    def format_ocr_text(self) -> str:
        raw = self.ocr_raw.strip()
        return raw if raw else "（未识别到文字）"

    def format_display(self, *, show_ocr: bool = True) -> str:
        lines: list[str] = [self.format_match_list()]
        if show_ocr and self.ocr_raw.strip():
            lines.append("——")
            lines.append(self.ocr_raw.strip())
        return "\n".join(lines)


class PetMatcher:
    def __init__(
        self,
        entries: list[PetEntry],
        *,
        min_score: float = 58.0,
        min_margin: float = 8.0,
        min_candidate_score: float = 10.0,
        top_k: int = 3,
    ) -> None:
        self._entries = entries
        self._norm_desc = [(e, normalize_text(e.description)) for e in entries]
        self.min_score = min_score
        self.min_margin = min_margin
        self.min_candidate_score = min_candidate_score
        self.top_k = top_k

    @classmethod
    def from_path(
        cls,
        catalog_path: Path,
        *,
        min_score: float = 58.0,
        min_margin: float = 8.0,
        min_candidate_score: float = 10.0,
        top_k: int = 3,
    ) -> PetMatcher:
        return cls(
            load_catalog(catalog_path),
            min_score=min_score,
            min_margin=min_margin,
            min_candidate_score=min_candidate_score,
            top_k=top_k,
        )

    def match(self, ocr_text: str) -> MatchResult:
        raw = ocr_text or ""
        ocr_norm = normalize_text(raw)
        if not ocr_norm:
            return MatchResult(best=None, candidates=(), ocr_raw=raw, confident=False)

        ranked: list[MatchCandidate] = []
        for entry, desc_norm in self._norm_desc:
            s = score_pair(ocr_norm, desc_norm)
            ranked.append(MatchCandidate(entry.name, s, entry.description))

        ranked.sort(key=lambda c: c.score, reverse=True)
        floor = self.min_candidate_score
        qualified = [c for c in ranked if c.score >= floor]
        top = qualified[: max(1, self.top_k)]

        best = top[0] if top else None
        confident = False
        if best and best.score >= self.min_score:
            if len(top) < 2:
                confident = True
            else:
                confident = (best.score - top[1].score) >= self.min_margin

        return MatchResult(
            best=best,
            candidates=tuple(top),
            ocr_raw=raw,
            confident=confident,
        )
