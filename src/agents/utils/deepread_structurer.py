from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def normalize_plaintext_to_markdown(
    text: str,
    *,
    title: str = "",
) -> str:
    """Convert raw document text into a Markdown-like form that preserves structure.

    This is intentionally lightweight and format-oriented:
    - preserve original lines and tables
    - promote heading-like lines into markdown headings
    - split common inline heading suffixes that are fused into table rows
    """

    raw_lines = str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = _split_inline_headings(raw_lines)

    normalized: List[str] = []
    saw_title = False

    if title.strip():
        normalized.append(f"# {title.strip()}")
        normalized.append("")
        saw_title = True

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if normalized and normalized[-1] != "":
                normalized.append("")
            continue

        if stripped.startswith("#"):
            normalized.append(stripped)
            continue

        heading = _normalize_heading_line(stripped)
        if heading is not None:
            if not saw_title and heading["level"] == 1:
                saw_title = True
            normalized.append(f"{'#' * heading['level']} {heading['text']}")
            normalized.append("")
            continue

        normalized.append(line.rstrip())

    while normalized and normalized[-1] == "":
        normalized.pop()

    return "\n".join(normalized)


def build_structured_corpus(
    text: str,
    *,
    title: str = "",
) -> Dict[str, Any]:
    markdown_text = normalize_plaintext_to_markdown(text, title=title)
    return parse_markdown_like_text_to_corpus(markdown_text)


def parse_markdown_like_text_to_corpus(markdown_text: str) -> Dict[str, Any]:
    """Adapted from DeepRead's markdown parser for local structured corpora."""

    lines = markdown_text.splitlines()

    heading_levels: List[int] = []
    for line in lines:
        info = _is_heading(line)
        if info:
            heading_levels.append(info["level"])
    min_level = min(heading_levels) if heading_levels else 1
    level_offset = min_level - 1

    nodes: List[Dict[str, Any]] = []
    node_map: Dict[str, Dict[str, Any]] = {}
    stack: List[Dict[str, Any]] = []
    next_id = 0
    front_matter_id: Optional[str] = None

    def alloc_id() -> str:
        nonlocal next_id
        next_id += 1
        return str(next_id)

    def ensure_front_matter(paragraph: Any) -> None:
        nonlocal front_matter_id, stack
        if front_matter_id is None:
            front_matter_id = alloc_id()
            node = {"id": front_matter_id, "title": "前言", "paragraphs": [], "children": []}
            nodes.append(node)
            node_map[front_matter_id] = node
            stack = [{"id": front_matter_id, "level": 1}]
        node_map[front_matter_id]["paragraphs"].append(paragraph)

    def new_node(level: int, title_text: str) -> Dict[str, Any]:
        nonlocal stack
        if level == 1 or not stack:
            node_id = alloc_id()
            node = {"id": node_id, "title": title_text.strip(), "paragraphs": [], "children": []}
            nodes.append(node)
            node_map[node_id] = node
            stack = [{"id": node_id, "level": 1}]
            return node

        while stack and stack[-1]["level"] >= level:
            stack.pop()

        if not stack:
            node_id = alloc_id()
            node = {"id": node_id, "title": title_text.strip(), "paragraphs": [], "children": []}
            nodes.append(node)
            node_map[node_id] = node
            stack = [{"id": node_id, "level": 1}]
            return node

        parent_id = stack[-1]["id"]
        node_id = alloc_id()
        node = {"id": node_id, "title": title_text.strip(), "paragraphs": [], "children": []}
        nodes.append(node)
        node_map[node_id] = node
        node_map[parent_id]["children"].append(node_id)
        stack.append({"id": node_id, "level": level})
        return node

    def append_paragraph(node: Optional[Dict[str, Any]], paragraph: Any) -> None:
        if node is None:
            ensure_front_matter(paragraph)
            return
        node["paragraphs"].append(paragraph)

    current_node: Optional[Dict[str, Any]] = None
    i = 0
    while i < len(lines):
        line = lines[i]
        heading = _is_heading(line)
        if heading:
            normalized_level = max(1, heading["level"] - level_offset)
            if stack:
                parent_level = stack[-1]["level"]
                level = min(normalized_level, parent_level + 1)
            else:
                level = normalized_level
            current_node = new_node(level, heading["title"])
            i += 1
            continue

        if "<table" in line.lower():
            block, nxt = _extract_html_table(lines, i)
            append_paragraph(current_node, block)
            i = nxt
            continue

        if re.match(r"^\s*\|.*\|\s*$", line):
            block, nxt = _extract_md_table(lines, i)
            append_paragraph(current_node, block)
            i = nxt
            continue

        if line.strip() == "":
            i += 1
            continue

        buf = [line]
        i += 1
        while i < len(lines):
            peek = lines[i]
            if peek.strip() == "":
                i += 1
                break
            if _is_heading(peek) or "<table" in peek.lower() or re.match(r"^\s*\|.*\|\s*$", peek):
                break
            buf.append(peek)
            i += 1
        append_paragraph(current_node, "\n".join(buf))

    return {"nodes": nodes}


def _is_heading(line: str) -> Optional[Dict[str, Any]]:
    match = re.match(r"^\s*(#{1,6})\s+(.*)$", line)
    if not match:
        return None
    return {"level": len(match.group(1)), "title": match.group(2).strip()}


def _extract_html_table(lines: List[str], start_idx: int) -> Tuple[str, int]:
    buf = [lines[start_idx]]
    if "</table>" in lines[start_idx].lower():
        return "\n".join(buf), start_idx + 1
    i = start_idx + 1
    while i < len(lines):
        buf.append(lines[i])
        if "</table>" in lines[i].lower():
            i += 1
            break
        i += 1
    return "\n".join(buf), i


def _extract_md_table(lines: List[str], start_idx: int) -> Tuple[str, int]:
    buf: List[str] = []
    i = start_idx
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*\|.*\|\s*$", line) or re.match(
            r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$",
            line,
        ):
            buf.append(line)
            i += 1
        else:
            break
    return "\n".join(buf), i


def _split_inline_headings(lines: List[str]) -> List[str]:
    """Split common heading suffixes accidentally fused into previous lines."""

    out: List[str] = []
    inline_heading_patterns = [
        r"(（[一二三四五六七八九十百]+）[^|]+)$",
        r"(\([ivxIVX]+\)\s+.+)$",
        r"([一二三四五六七八九十百]+、[^|]+)$",
    ]

    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            out.append(line)
            continue
        split_done = False
        for pattern in inline_heading_patterns:
            match = re.search(pattern, stripped)
            if not match:
                continue
            suffix = match.group(1).strip()
            prefix = stripped[: match.start(1)].rstrip()
            if not prefix or len(suffix) > 80:
                continue
            if "|" in suffix:
                continue
            out.append(prefix)
            out.append("")
            out.append(suffix)
            split_done = True
            break
        if not split_done:
            out.append(line)
    return out


def _normalize_heading_line(line: str) -> Optional[Dict[str, Any]]:
    stripped = line.strip()
    if not stripped:
        return None

    if re.match(r"^(abstract|references|bibliography|appendix)\b", stripped, flags=re.IGNORECASE):
        return {"level": 2, "text": stripped}

    if re.match(r"^参考文献$", stripped):
        return {"level": 2, "text": stripped}

    if re.match(r"^\d+(\.\d+)*\s+\S+", stripped):
        level = min(2 + stripped.split()[0].count("."), 4)
        return {"level": level, "text": stripped}

    if re.match(r"^[一二三四五六七八九十百]+、\S+", stripped):
        return {"level": 2, "text": stripped}

    if re.match(r"^[（(][一二三四五六七八九十百]+[）)]\S+", stripped):
        return {"level": 3, "text": stripped}

    if re.match(r"^\([ivxIVX]+\)\s+\S+", stripped):
        return {"level": 3, "text": stripped}

    if re.match(r"^第[一二三四五六七八九十百]+[章节部分编]\S*", stripped):
        return {"level": 2, "text": stripped}

    legal_markers_level2 = [
        "本院认为",
        "本院赔偿委员会认为",
        "赔偿委员会认为",
        "经审理查明",
        "审理查明",
        "另查明",
        "再审申请人称",
        "申请人称",
        "申诉人称",
        "答辩称",
        "辩称",
        "本院查明",
    ]
    if any(stripped.startswith(marker) for marker in legal_markers_level2):
        return {"level": 2, "text": stripped}

    legal_markers_level3 = [
        "判决如下",
        "裁定如下",
        "决定如下",
        "请求如下",
        "申请事项",
        "赔偿请求",
    ]
    if any(stripped.startswith(marker) for marker in legal_markers_level3):
        return {"level": 3, "text": stripped}

    # Promote common report titles when they appear standalone.
    if len(stripped) <= 80 and re.search(r"(年度报告|季度报告|半年度报告|年报|中报)$", stripped):
        return {"level": 1, "text": stripped}

    return None


__all__ = [
    "normalize_plaintext_to_markdown",
    "build_structured_corpus",
    "parse_markdown_like_text_to_corpus",
]
