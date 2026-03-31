from __future__ import annotations

import re
from typing import List, Sequence


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def dedupe_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        normalized = normalize_text(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def build_doc_title_bundle(titles: Sequence[str]) -> str:
    normalized_titles = dedupe_keep_order(titles)
    return "\n".join(
        f"- DOC{index}: {title}"
        for index, title in enumerate(normalized_titles, start=1)
    )


def build_doc_slot_bundle(doc_count: int) -> str:
    if doc_count <= 0:
        return ""

    lines: List[str] = []
    for index in range(1, doc_count + 1):
        lines.append(f"- DOC{index}:")
        lines.append("  status: unscanned")
        lines.append("  slots: {}")
    return "\n".join(lines)
