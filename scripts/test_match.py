"""本地测试：用一段（可含错字）描述文本匹配小动物。用法: python scripts/test_match.py "靠近盒子就听见噼啪声" """

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rw_tool.pet_matcher import PetMatcher  # noqa: E402


def main() -> None:
    sample = " ".join(sys.argv[1:]) or "一靠近盒子就听见热闹的噼啪声此起彼伏，接着盒顶冒出了一朵朵小烟花"
    matcher = PetMatcher.from_path(ROOT / "desc.json")
    result = matcher.match(sample)
    print(result.format_display())


if __name__ == "__main__":
    main()
