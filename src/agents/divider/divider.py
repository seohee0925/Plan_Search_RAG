from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

try:
    from src.agents.planner.loong.planner import (
        OpenAIChatLLM,
        Planner as BasePlanner,
        PlannerLLM,
    )
except ModuleNotFoundError:  # pragma: no cover
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.agents.planner.loong.planner import (  # type: ignore
        OpenAIChatLLM,
        Planner as BasePlanner,
        PlannerLLM,
    )


REQUIRED_HEADERS = [
    "Document Anchor",
    "Region Roles",
    "Divider Notes",
]


@dataclass
class DividerPacket:
    packet_id: str
    packet_type: str
    section_path: str
    order_index: int
    text: str
    preview: str
    prev_packet_id: str = ""
    next_packet_id: str = ""
    search_tags: List[str] = field(default_factory=list)


@dataclass
class DividerRegion:
    region_id: str
    section_path: str
    packet_ids: List[str]
    start_packet_id: str
    end_packet_id: str
    region_type: str
    preview: str
    packet_count: int


@dataclass
class DividerInput:
    instruction: str
    question: str
    doc_id: str
    display_title: str
    document_text: str
    packet_inventory: str
    region_inventory: str
    sample_id: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DividerOutput:
    raw_text: str
    doc_id: str
    display_title: str
    doc_anchor: str
    region_roles: List[Dict[str, str]]
    divider_notes: str
    doc_map_summary: str
    packet_store: List[Dict[str, Any]]
    region_store: List[Dict[str, Any]]
    search_views: Dict[str, Any]
    packet_count: int
    region_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DividerDocRun:
    doc_id: str
    trace_path: str
    divider_input: DividerInput
    divider_output: DividerOutput

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "trace_path": self.trace_path,
            "input": asdict(self.divider_input),
            "output": self.divider_output.to_dict(),
        }


class DividerParseError(ValueError):
    pass


class Divider:
    def __init__(
        self,
        llm: PlannerLLM,
        project_root: Optional[Path] = None,
        prompt_dir: str = "prompts",
        trace_dir: str = "trace_logs/divider",
    ) -> None:
        self.llm = llm
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[3]
        self.prompt_dir = self.project_root / prompt_dir
        self.trace_dir = self.project_root / trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.system_prompt = (self.prompt_dir / "divider_sys.txt").read_text(encoding="utf-8").strip()
        self.user_prompt_template = (self.prompt_dir / "divider_user.txt").read_text(encoding="utf-8").strip()

    @staticmethod
    def _safe_filename(name: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._")
        return safe or "unknown"

    @staticmethod
    def _preview(text: str, limit: int = 180) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    @staticmethod
    def _split_long_paragraph(text: str, target_chars: int = 1200) -> List[str]:
        compact = text.strip()
        if not compact:
            return []
        if len(compact) <= target_chars:
            return [compact]

        parts = re.split(r"(?<=[。！？；;.!?])\s+", compact)
        parts = [part.strip() for part in parts if part.strip()]
        if len(parts) <= 1:
            hard_chunks: List[str] = []
            for start in range(0, len(compact), target_chars):
                chunk = compact[start : start + target_chars].strip()
                if chunk:
                    hard_chunks.append(chunk)
            return hard_chunks or [compact]

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for part in parts:
            part_len = len(part)
            if current and current_len + part_len > target_chars:
                chunks.append(" ".join(current).strip())
                current = [part]
                current_len = part_len
                continue
            current.append(part)
            current_len += part_len + 1
        if current:
            chunks.append(" ".join(current).strip())
        return chunks or [compact]

    @classmethod
    def _append_packet(
        cls,
        packets: List[DividerPacket],
        packet_type: str,
        section_path: str,
        text: str,
    ) -> None:
        normalized_text = text.strip()
        if not normalized_text:
            return
        packet_id = f"P{len(packets) + 1}"
        packets.append(
            DividerPacket(
                packet_id=packet_id,
                packet_type=packet_type,
                section_path=section_path,
                order_index=len(packets),
                text=normalized_text,
                preview=cls._preview(normalized_text),
                search_tags=cls._search_tags_for_packet(
                    text=normalized_text,
                    packet_type=packet_type,
                    section_path=section_path,
                ),
            )
        )

    @staticmethod
    def _packet_type_for_line(line: str, in_references: bool) -> str:
        stripped = line.strip()
        if not stripped:
            return "blank"
        if re.match(r"^#{1,6}\s+", stripped):
            return "heading"
        if re.match(r"^(part\s+[ivx]+|item\s+\d+\.?)", stripped, flags=re.IGNORECASE):
            return "heading"
        if re.match(r"^(第[一二三四五六七八九十百]+|[一二三四五六七八九十]+[、.])", stripped):
            return "heading"
        if stripped.count("|") >= 2:
            return "table_row"
        if in_references:
            return "reference_entry"
        if re.match(r"^[-*•]\s+", stripped):
            return "list_item"
        if re.match(r"^\d+[.)、]\s+", stripped):
            return "list_item"
        if re.match(r"^[（(]?[一二三四五六七八九十0-9]+[）)]", stripped):
            return "clause"
        return "paragraph"

    @classmethod
    def _packetize_document(cls, document_text: str) -> Tuple[List[DividerPacket], List[DividerRegion], Dict[str, Any]]:
        lines = document_text.splitlines()
        packets: List[DividerPacket] = []
        current_text: List[str] = []
        current_type = "paragraph"
        heading_stack: List[str] = []
        current_section_path = ""
        in_references = False

        def flush_current() -> None:
            nonlocal current_text, current_type, current_section_path
            text = "\n".join(current_text).strip()
            current_text = []
            if not text:
                return
            if current_type == "paragraph" and len(text) > 1600:
                for chunk in cls._split_long_paragraph(text):
                    cls._append_packet(
                        packets=packets,
                        packet_type=current_type,
                        section_path=current_section_path,
                        text=chunk,
                    )
                return
            cls._append_packet(
                packets=packets,
                packet_type=current_type,
                section_path=current_section_path,
                text=text,
            )

        for raw_line in lines:
            line = raw_line.rstrip()
            stripped = line.strip()

            if stripped:
                lowered = stripped.casefold()
                if re.match(r"^#{1,6}\s+", stripped):
                    heading_text = re.sub(r"^#{1,6}\s+", "", stripped).strip()
                    if heading_text:
                        in_references = bool(re.search(r"(references|bibliography|参考文献)", heading_text, flags=re.IGNORECASE))
                elif re.match(r"^(references|bibliography|参考文献)\b", stripped, flags=re.IGNORECASE):
                    in_references = True

            packet_type = cls._packet_type_for_line(line, in_references=in_references)

            if packet_type == "blank":
                flush_current()
                continue

            if packet_type == "heading":
                flush_current()
                heading_text = re.sub(r"^#{1,6}\s+", "", stripped).strip()
                if not heading_text:
                    heading_text = stripped
                depth = 1
                markdown_match = re.match(r"^(#{1,6})\s+", stripped)
                if markdown_match:
                    depth = len(markdown_match.group(1))
                elif re.match(r"^(part\s+[ivx]+|item\s+\d+\.?)", stripped, flags=re.IGNORECASE):
                    depth = 2
                elif re.match(r"^第[一二三四五六七八九十百]+", stripped):
                    depth = 2
                elif re.match(r"^[一二三四五六七八九十]+[、.]", stripped):
                    depth = 3

                while len(heading_stack) >= depth:
                    heading_stack.pop()
                heading_stack.append(heading_text)
                current_section_path = " > ".join(heading_stack)

                cls._append_packet(
                    packets=packets,
                    packet_type="heading",
                    section_path=current_section_path,
                    text=heading_text,
                )
                continue

            # Start a fresh packet whenever the packet kind changes into a more structural type.
            if current_text and packet_type != current_type and (
                packet_type in {"table_row", "reference_entry", "list_item", "clause"}
                or current_type in {"table_row", "reference_entry", "list_item", "clause"}
            ):
                flush_current()

            if not current_text:
                current_type = packet_type
            current_section_path = " > ".join(heading_stack)
            current_text.append(stripped)

            if packet_type in {"table_row", "reference_entry", "list_item", "clause"}:
                flush_current()

        flush_current()

        for idx, packet in enumerate(packets):
            if idx > 0:
                packet.prev_packet_id = packets[idx - 1].packet_id
            if idx + 1 < len(packets):
                packet.next_packet_id = packets[idx + 1].packet_id

        regions: List[DividerRegion] = []
        current_region_packets: List[DividerPacket] = []
        current_region_path = ""
        current_region_chars = 0
        for packet in packets:
            if not current_region_packets:
                current_region_packets = [packet]
                current_region_path = packet.section_path
                current_region_chars = len(packet.text)
                continue
            same_path = packet.section_path == current_region_path
            current_packet_types = {item.packet_type for item in current_region_packets}
            combined_packet_types = current_packet_types | {packet.packet_type}
            non_heading_types = {packet_type for packet_type in combined_packet_types if packet_type != "heading"}
            structural_break = packet.packet_type == "heading"
            table_like_run = bool(non_heading_types) and non_heading_types <= {"table_row"}
            reference_like_run = bool(non_heading_types) and non_heading_types <= {"reference_entry"}
            clause_like_run = bool(non_heading_types) and non_heading_types <= {"clause", "list_item"}

            if table_like_run or reference_like_run:
                max_packets = 64
                max_chars = 8000
            elif clause_like_run:
                max_packets = 12
                max_chars = 2400
            else:
                max_packets = 4
                max_chars = 1600

            region_too_large = (
                len(current_region_packets) >= max_packets
                or (current_region_chars + len(packet.text)) >= max_chars
            )

            if same_path and not structural_break and not region_too_large:
                current_region_packets.append(packet)
                current_region_chars += len(packet.text)
                continue

            region_id = f"R{len(regions) + 1}"
            regions.append(cls._build_region(region_id, current_region_packets))
            current_region_packets = [packet]
            current_region_path = packet.section_path
            current_region_chars = len(packet.text)

        if current_region_packets:
            region_id = f"R{len(regions) + 1}"
            regions.append(cls._build_region(region_id, current_region_packets))

        search_views = cls._build_search_views(packets, regions)
        return packets, regions, search_views

    @classmethod
    def _build_region(cls, region_id: str, packets: Sequence[DividerPacket]) -> DividerRegion:
        packet_ids = [packet.packet_id for packet in packets]
        section_path = packets[0].section_path if packets else ""
        region_type = cls._infer_region_type(packets)
        preview = cls._preview(" ".join(packet.preview for packet in packets[:3]), limit=220)
        return DividerRegion(
            region_id=region_id,
            section_path=section_path,
            packet_ids=packet_ids,
            start_packet_id=packet_ids[0],
            end_packet_id=packet_ids[-1],
            region_type=region_type,
            preview=preview,
            packet_count=len(packet_ids),
        )

    @staticmethod
    def _infer_region_type(packets: Sequence[DividerPacket]) -> str:
        packet_types = {packet.packet_type for packet in packets}
        section_path = (packets[0].section_path if packets else "").casefold()
        if "reference_entry" in packet_types or re.search(r"(references|bibliography|参考文献)", section_path):
            return "attribution_region"
        if "table_row" in packet_types:
            return "table_region"
        if "heading" in packet_types and len(packet_types) == 1:
            return "section_header"
        if "clause" in packet_types or "list_item" in packet_types:
            return "clause_region"
        return "paragraph_region"

    @staticmethod
    def _search_tags_for_packet(text: str, packet_type: str, section_path: str) -> List[str]:
        tags = {packet_type}
        lowered = text.casefold()
        if re.search(r"\d", text):
            tags.add("numeric")
        if re.search(r"(references|bibliography|参考文献)", section_path, flags=re.IGNORECASE):
            tags.add("attribution")
        if re.search(r"(table|资产负债表|financial statements|balance sheets|notes to)", text, flags=re.IGNORECASE):
            tags.add("structured_financial")
        if re.search(r"(判决|裁定|本院认为|经审理查明|claim|诉称|申请)", text):
            tags.add("legal_reasoning")
        if re.search(r"(abstract|introduction|conclusion|references)", lowered):
            tags.add("paper_section")
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,6}", text)
        for token in tokens[:6]:
            tags.add(token)
        return sorted(tags)

    @classmethod
    def _build_search_views(cls, packets: Sequence[DividerPacket], regions: Sequence[DividerRegion]) -> Dict[str, Any]:
        return {
            "packet_type_counts": {
                packet_type: sum(1 for packet in packets if packet.packet_type == packet_type)
                for packet_type in sorted({packet.packet_type for packet in packets})
            },
            "region_type_counts": {
                region_type: sum(1 for region in regions if region.region_type == region_type)
                for region_type in sorted({region.region_type for region in regions})
            },
            "toc": [
                {
                    "packet_id": packet.packet_id,
                    "section_path": packet.section_path,
                    "text": packet.text,
                }
                for packet in packets
                if packet.packet_type == "heading"
            ],
            "region_inventory": [
                {
                    "region_id": region.region_id,
                    "region_type": region.region_type,
                    "section_path": region.section_path,
                    "packet_span": f"{region.start_packet_id}..{region.end_packet_id}",
                    "preview": region.preview,
                }
                for region in regions
            ],
        }

    @staticmethod
    def _format_packet_inventory(packets: Sequence[DividerPacket], limit: int = 48) -> str:
        lines: List[str] = []
        for packet in packets[:limit]:
            lines.append(
                f"- [{packet.packet_id}] type={packet.packet_type} | section_path={packet.section_path or 'ROOT'} | preview={packet.preview}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_region_inventory(regions: Sequence[DividerRegion], limit: int = 20) -> str:
        lines: List[str] = []
        for region in regions[:limit]:
            lines.append(
                f"- [{region.region_id}] region_type={region.region_type} | section_path={region.section_path or 'ROOT'} | packet_span={region.start_packet_id}..{region.end_packet_id} | preview={region.preview}"
            )
        return "\n".join(lines)

    @classmethod
    def build_inputs_from_loong(
        cls,
        record: Dict[str, Any],
        sample_id_prefix: Optional[str] = None,
    ) -> List[DividerInput]:
        raw_docs = str(record.get("docs", "") or "")
        contents, titles = BasePlanner.split_loong_docs(raw_docs)

        if not titles:
            fallback_titles = record.get("doc", [])
            if isinstance(fallback_titles, list):
                titles = [str(item).strip() for item in fallback_titles if str(item).strip()]

        inputs: List[DividerInput] = []
        prefix = sample_id_prefix or str(record.get("id", "unknown"))
        for idx, title in enumerate(titles):
            doc_id = f"DOC{idx + 1}"
            text = contents[idx] if idx < len(contents) else ""
            packets, regions, _ = cls._packetize_document(text)
            inputs.append(
                DividerInput(
                    instruction=str(record.get("instruction", "")).strip(),
                    question=str(record.get("question", "")).strip(),
                    doc_id=doc_id,
                    display_title=title,
                    document_text=text,
                    packet_inventory=cls._format_packet_inventory(packets),
                    region_inventory=cls._format_region_inventory(regions),
                    sample_id=f"{prefix}_{doc_id}",
                    metadata={
                        "record_id": record.get("id"),
                        "level": record.get("level"),
                        "set": record.get("set"),
                        "type": record.get("type"),
                        "language": record.get("language"),
                        "packet_count": len(packets),
                        "region_count": len(regions),
                    },
                )
            )
        return inputs

    def render_user_prompt(self, divider_input: DividerInput) -> str:
        return self.user_prompt_template.format(
            instruction=divider_input.instruction,
            question=divider_input.question,
            doc_id=divider_input.doc_id,
            display_title=divider_input.display_title,
            packet_inventory=divider_input.packet_inventory,
            region_inventory=divider_input.region_inventory,
        )

    def divide_document(self, divider_input: DividerInput, save_trace: bool = True) -> DividerOutput:
        packets, regions, search_views = self._packetize_document(divider_input.document_text)
        user_prompt = self.render_user_prompt(divider_input)
        raw_text = self.llm.generate(self.system_prompt, user_prompt)
        parsed = self.parse_response(raw_text)
        region_roles = self._parse_kv_bullets(parsed["Region Roles"])
        output = DividerOutput(
            raw_text=raw_text,
            doc_id=divider_input.doc_id,
            display_title=divider_input.display_title,
            doc_anchor=parsed["Document Anchor"].strip(),
            region_roles=region_roles,
            divider_notes=parsed["Divider Notes"].strip(),
            doc_map_summary=self._build_doc_map_summary(divider_input, packets, regions, region_roles),
            packet_store=[asdict(packet) for packet in packets],
            region_store=[asdict(region) for region in regions],
            search_views=search_views,
            packet_count=len(packets),
            region_count=len(regions),
        )
        if save_trace:
            self._save_trace(divider_input, user_prompt, output)
        return output

    def run_loong_record(
        self,
        record: Dict[str, Any],
        sample_id_prefix: Optional[str] = None,
        save_trace: bool = True,
    ) -> List[DividerDocRun]:
        runs: List[DividerDocRun] = []
        for divider_input in self.build_inputs_from_loong(record, sample_id_prefix=sample_id_prefix):
            output = self.divide_document(divider_input, save_trace=save_trace)
            runs.append(
                DividerDocRun(
                    doc_id=divider_input.doc_id,
                    trace_path=str(self.trace_path_for_sample_id(divider_input.sample_id)),
                    divider_input=divider_input,
                    divider_output=output,
                )
            )
        return runs

    def trace_path_for_sample_id(self, sample_id: str) -> Path:
        return self.trace_dir / f"{self._safe_filename(sample_id)}.json"

    def _save_trace(self, divider_input: DividerInput, user_prompt: str, output: DividerOutput) -> Path:
        path = self.trace_path_for_sample_id(divider_input.sample_id)
        payload = {
            "input": asdict(divider_input),
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt,
            "output": output.to_dict(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def parse_response(raw_text: str) -> Dict[str, str]:
        matches = list(re.finditer(r"^###\s+(.+?)\s*$", raw_text, flags=re.MULTILINE))
        if not matches:
            raise DividerParseError("No markdown section headers were found in divider output.")
        sections: Dict[str, str] = {}
        for idx, match in enumerate(matches):
            header = match.group(1).strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
            sections[header] = raw_text[start:end].strip()
        missing = [header for header in REQUIRED_HEADERS if header not in sections]
        if missing:
            raise DividerParseError(f"Divider output is missing required sections: {missing}")
        return sections

    @staticmethod
    def _parse_kv_bullets(section_text: str) -> List[Dict[str, str]]:
        items: List[Dict[str, str]] = []
        for raw_line in section_text.splitlines():
            line = raw_line.strip()
            if not line.startswith("- "):
                continue
            body = line[2:].strip()
            item: Dict[str, str] = {}
            bracket_match = re.match(r"\[(.+?)\]\s*(.*)", body)
            remainder = bracket_match.group(2).strip() if bracket_match else body
            if bracket_match:
                item["id"] = bracket_match.group(1).strip()
            for part in remainder.split("|"):
                segment = part.strip()
                if "=" in segment:
                    key, value = segment.split("=", 1)
                    item[key.strip()] = value.strip()
            if item:
                items.append(item)
        return items

    @classmethod
    def _build_doc_map_summary(
        cls,
        divider_input: DividerInput,
        packets: Sequence[DividerPacket],
        regions: Sequence[DividerRegion],
        region_roles: Sequence[Dict[str, str]],
    ) -> str:
        lines = [
            f"doc={divider_input.doc_id}",
            f"title={divider_input.display_title}",
            f"packet_count={len(packets)}",
            f"region_count={len(regions)}",
        ]
        for role in region_roles[:8]:
            region_id = str(role.get("region_id", "")).strip()
            role_name = str(role.get("role", "")).strip()
            note = str(role.get("why", role.get("rationale", ""))).strip()
            lines.append(f"{region_id}: {role_name} ({note})")
        if not region_roles:
            for region in regions[:6]:
                lines.append(
                    f"{region.region_id}: {region.region_type} @ {region.section_path or 'ROOT'} -> {region.preview}"
                )
        return "\n".join(lines)


__all__ = [
    "Divider",
    "DividerDocRun",
    "DividerInput",
    "DividerOutput",
    "DividerPacket",
    "DividerRegion",
    "DividerParseError",
    "OpenAIChatLLM",
]
