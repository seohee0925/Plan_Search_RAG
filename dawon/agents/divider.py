from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

try:
    from dawon.agents.planner import (
        OpenAIChatLLM,
        Planner as BasePlanner,
        PlannerLLM,
    )
except ModuleNotFoundError:  # pragma: no cover
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from dawon.agents.planner import (  # type: ignore
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
    granularity: str = "meso"
    unit_family: str = ""


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
    divider_mode: str
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
        divider_mode: str = "default",
    ) -> None:
        self.llm = llm
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        self.prompt_dir = self.project_root / prompt_dir
        self.trace_dir = self.project_root / trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.system_prompt = (self.prompt_dir / "divider_sys.txt").read_text(encoding="utf-8").strip()
        self.user_prompt_template = (self.prompt_dir / "divider_user.txt").read_text(encoding="utf-8").strip()
        self.divider_mode = self._normalize_divider_mode(divider_mode)

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
    def _normalize_divider_mode(divider_mode: str) -> str:
        normalized = str(divider_mode or "default").strip().lower()
        if normalized in {"", "base", "baseline"}:
            return "default"
        if normalized not in {"default", "granv1", "granv2"}:
            raise ValueError(f"Unsupported divider_mode={divider_mode!r}. Expected 'default', 'granv1', or 'granv2'.")
        return normalized

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
    def _granularity_sequence(cls, divider_mode: str) -> List[str]:
        normalized = cls._normalize_divider_mode(divider_mode)
        if normalized == "granv1":
            return ["meso", "macro"]
        if normalized == "granv2":
            return ["meso", "semantic"]
        return ["meso"]

    @staticmethod
    def _region_id_prefix(granularity: str, divider_mode: str) -> str:
        if divider_mode == "default" and granularity == "meso":
            return "R"
        if granularity == "micro":
            return "MICRO_R"
        if granularity == "macro":
            return "MACRO_R"
        if granularity == "semantic":
            return "SEM_R"
        return "R"

    @staticmethod
    def _region_limit_profile(run_family: str, granularity: str) -> tuple[int, int]:
        profiles = {
            "micro": {
                "table": (1, 700),
                "reference": (1, 700),
                "clause": (3, 900),
                "default": (2, 900),
            },
            "meso": {
                "table": (64, 8000),
                "reference": (64, 8000),
                "clause": (12, 2400),
                "default": (4, 1600),
            },
            "macro": {
                "table": (96, 12000),
                "reference": (96, 12000),
                "clause": (24, 4800),
                "default": (8, 3200),
            },
        }
        profile = profiles.get(granularity, profiles["meso"])
        return profile.get(run_family, profile["default"])

    @staticmethod
    def _run_family_for_region_packets(packets: Sequence[DividerPacket]) -> str:
        packet_types = {item.packet_type for item in packets}
        non_heading_types = {packet_type for packet_type in packet_types if packet_type != "heading"}
        if bool(non_heading_types) and non_heading_types <= {"table_row"}:
            return "table"
        if bool(non_heading_types) and non_heading_types <= {"reference_entry"}:
            return "reference"
        if bool(non_heading_types) and non_heading_types <= {"clause", "list_item"}:
            return "clause"
        return "default"

    @classmethod
    def _build_regions_for_granularity(
        cls,
        packets: Sequence[DividerPacket],
        granularity: str,
        divider_mode: str,
    ) -> List[DividerRegion]:
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
            structural_break = packet.packet_type == "heading"
            run_family = cls._run_family_for_region_packets([*current_region_packets, packet])
            max_packets, max_chars = cls._region_limit_profile(run_family, granularity)
            region_too_large = (
                len(current_region_packets) >= max_packets
                or (current_region_chars + len(packet.text)) >= max_chars
            )

            if same_path and not structural_break and not region_too_large:
                current_region_packets.append(packet)
                current_region_chars += len(packet.text)
                continue

            region_id = f"{cls._region_id_prefix(granularity, divider_mode)}{len(regions) + 1}"
            regions.append(cls._build_region(region_id, current_region_packets, granularity=granularity))
            current_region_packets = [packet]
            current_region_path = packet.section_path
            current_region_chars = len(packet.text)

        if current_region_packets:
            region_id = f"{cls._region_id_prefix(granularity, divider_mode)}{len(regions) + 1}"
            regions.append(cls._build_region(region_id, current_region_packets, granularity=granularity))
        return regions

    @staticmethod
    def _is_note_like_packet(packet: DividerPacket) -> bool:
        if packet.packet_type not in {"paragraph", "clause", "list_item"}:
            return False
        combined = f"{packet.section_path} {packet.text}"
        return bool(
            re.search(
                r"(notes?|注[:：]?|说明|会计政策|公允价值|以公允价值计量|交易性金融资产|资产负债表|financial|fair value|measured)",
                combined,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _is_value_bearing_table_row(packet: DividerPacket) -> bool:
        if packet.packet_type != "table_row":
            return False
        text = packet.text
        return bool(re.search(r"\d", text) and re.search(r"[A-Za-z\u4e00-\u9fff]{2,}", text))

    @staticmethod
    def _is_reference_packet(packet: DividerPacket) -> bool:
        return packet.packet_type == "reference_entry"

    @staticmethod
    def _is_citation_context_packet(packet: DividerPacket) -> bool:
        if packet.packet_type not in {"paragraph", "clause", "list_item"}:
            return False
        text = packet.text
        patterns = [
            r"\[[0-9]{1,3}\]",
            r"\([A-Z][A-Za-z]+(?:\s+et al\.)?,?\s*\d{4}[a-z]?\)",
            r"\bet al\.,?\s*\d{4}\b",
            r"[A-Z][A-Za-z-]+\s+and\s+[A-Z][A-Za-z-]+,\s*\d{4}",
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def _extract_query_focus_terms(task_text: str) -> List[str]:
        text = str(task_text or "").strip()
        if not text:
            return []

        candidates: List[str] = []
        quote_patterns = [
            r"'([^']{2,80})'",
            r"\"([^\"]{2,80})\"",
            r"“([^”]{2,80})”",
            r"‘([^’]{2,80})’",
            r"《([^》]{2,80})》",
        ]
        for pattern in quote_patterns:
            candidates.extend(match.strip() for match in re.findall(pattern, text) if match.strip())

        candidates.extend(
            token.strip()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9._:+-]{2,}|[\u4e00-\u9fff]{2,16}", text)
            if token.strip()
        )

        stopwords = {
            "请回答",
            "问题",
            "公司",
            "文书",
            "标题",
            "哪些",
            "哪个",
            "回答",
            "以下",
            "只需要",
            "根据",
            "provided",
            "question",
            "answer",
            "which",
            "what",
            "title",
            "titles",
            "paper",
            "papers",
            "document",
            "documents",
        }
        filtered: List[str] = []
        seen: set[str] = set()
        for candidate in sorted(candidates, key=lambda item: (-len(item), item.casefold())):
            normalized = candidate.casefold()
            if normalized in seen:
                continue
            if candidate in stopwords or normalized in stopwords:
                continue
            if len(candidate) < 3 and not re.search(r"[\u4e00-\u9fff]", candidate):
                continue
            seen.add(normalized)
            filtered.append(candidate)
        return filtered[:8]

    @staticmethod
    def _packet_matches_focus(packet: DividerPacket, focus_terms: Sequence[str]) -> bool:
        if not focus_terms:
            return False
        haystack = f"{packet.section_path} {packet.text}".casefold()
        return any(term.casefold() in haystack for term in focus_terms)

    @staticmethod
    def _is_legal_case_profile_packet(packet: DividerPacket, order_index: int) -> bool:
        if packet.packet_type not in {"paragraph", "clause", "list_item"}:
            return False
        text = packet.text
        return order_index <= 12 and bool(
            re.search(
                r"(一案|原告|被告|上诉人|被上诉人|申请人|被申请人|诉称|请求|行政机关|资格认定|工伤保险|养老保险|行政给付|不予支付)",
                text,
            )
        )

    @staticmethod
    def _is_legal_reasoning_packet(packet: DividerPacket) -> bool:
        if packet.packet_type not in {"paragraph", "clause", "list_item"}:
            return False
        return bool(re.search(r"(本院认为|法院认为|经审理查明|审理查明|争议焦点|查明如下)", packet.text))

    @staticmethod
    def _is_legal_decision_packet(packet: DividerPacket) -> bool:
        if packet.packet_type not in {"paragraph", "clause", "list_item"}:
            return False
        return bool(
            re.search(
                r"(判决如下|裁定如下|决定如下|驳回上诉|维持原判|撤销|责令|确认违法|不予支付|给付|支付|准予|不予受理)",
                packet.text,
            )
        )

    @staticmethod
    def _prepend_heading_index(packets: Sequence[DividerPacket], indices: List[int]) -> List[int]:
        if not indices:
            return []
        first_index = indices[0]
        if first_index <= 0:
            return indices
        previous = packets[first_index - 1]
        first_packet = packets[first_index]
        if previous.packet_type == "heading" and previous.section_path == first_packet.section_path:
            return [first_index - 1, *indices]
        return indices

    @classmethod
    def _extend_forward_bundle(
        cls,
        packets: Sequence[DividerPacket],
        start_index: int,
        max_extra: int,
        allowed_types: Sequence[str],
    ) -> List[int]:
        indices = [start_index]
        base_packet = packets[start_index]
        extras = 0
        for cursor in range(start_index + 1, len(packets)):
            candidate = packets[cursor]
            if candidate.packet_type == "heading":
                break
            if candidate.section_path != base_packet.section_path:
                break
            if candidate.packet_type not in allowed_types:
                break
            indices.append(cursor)
            extras += 1
            if extras >= max_extra:
                break
        return indices

    @classmethod
    def _find_table_run(cls, packets: Sequence[DividerPacket], start_index: int) -> tuple[int, int]:
        return cls._find_same_type_run(packets, start_index, packet_type="table_row")

    @classmethod
    def _find_same_type_run(
        cls,
        packets: Sequence[DividerPacket],
        start_index: int,
        packet_type: str,
    ) -> tuple[int, int]:
        run_start = start_index
        end_index = start_index
        base_section = packets[start_index].section_path
        while run_start - 1 >= 0:
            candidate = packets[run_start - 1]
            if candidate.packet_type != packet_type or candidate.section_path != base_section:
                break
            run_start -= 1
        while end_index + 1 < len(packets):
            candidate = packets[end_index + 1]
            if candidate.packet_type != packet_type or candidate.section_path != base_section:
                break
            end_index += 1
        return run_start, end_index

    @classmethod
    def _collect_financial_note_indices(
        cls,
        packets: Sequence[DividerPacket],
        run_end: int,
    ) -> List[int]:
        note_indices: List[int] = []
        seen_note_heading = False
        for cursor in range(run_end + 1, min(run_end + 6, len(packets))):
            candidate = packets[cursor]
            if candidate.packet_type == "heading":
                if re.search(r"(notes?|注[:：]?|说明)", candidate.text, flags=re.IGNORECASE):
                    note_indices.append(cursor)
                    seen_note_heading = True
                    continue
                break
            if cls._is_note_like_packet(candidate):
                note_indices.append(cursor)
                if not seen_note_heading:
                    break
                if len(note_indices) >= 3:
                    break
                continue
            if seen_note_heading:
                break
        return note_indices

    @classmethod
    def _build_semantic_regions(
        cls,
        packets: Sequence[DividerPacket],
        divider_mode: str,
        task_text: str = "",
    ) -> List[DividerRegion]:
        if divider_mode != "granv2":
            return []

        focus_terms = cls._extract_query_focus_terms(task_text)
        task_lower = str(task_text or "").casefold()
        wants_financial = bool(
            re.search(r"(财务|资产|金额|数值|financial|asset|assets|report|highest|lowest|largest|交易性金融资产)", task_lower)
        )
        wants_legal = bool(
            re.search(r"(案由|判决|行政|给付|案件|judgment|case|title_list|administrative)", task_lower)
        )
        wants_paper = bool(
            re.search(r"(citation|reference|cite|paper|论文|OpenMoE|TinyLlama|Lory|references|bibliography)", task_text, flags=re.IGNORECASE)
        )
        if not any((wants_financial, wants_legal, wants_paper)):
            wants_financial = wants_legal = wants_paper = True

        semantic_specs: List[tuple[str, str, List[int]]] = []
        if wants_financial:
            seen_financial_runs: set[tuple[int, int]] = set()
            for index, packet in enumerate(packets):
                if not cls._is_value_bearing_table_row(packet):
                    continue
                if focus_terms and not cls._packet_matches_focus(packet, focus_terms):
                    continue
                run_start, run_end = cls._find_table_run(packets, index)
                run_key = (run_start, run_end)
                if run_key in seen_financial_runs:
                    continue
                seen_financial_runs.add(run_key)

                header_indices = list(range(run_start, min(run_start + 2, run_end + 1)))
                bundle_indices = cls._prepend_heading_index(packets, sorted(set(header_indices + [index])))
                note_indices = cls._collect_financial_note_indices(packets, run_end)[:2]
                semantic_specs.append(
                    (
                        "table_row_bundle_region",
                        "financial_row_bundle",
                        sorted(set(bundle_indices + note_indices)),
                    )
                )
                if note_indices:
                    semantic_specs.append(
                        (
                            "table_note_bundle_region",
                            "financial_table_note_bundle",
                            sorted(set(bundle_indices + note_indices)),
                        )
                    )

        if wants_paper:
            seen_reference_runs: set[tuple[int, int]] = set()
            citation_context_count = 0
            for index, packet in enumerate(packets):
                if cls._is_reference_packet(packet):
                    run_start, run_end = cls._find_same_type_run(
                        packets,
                        index,
                        packet_type="reference_entry",
                    )
                    run_key = (run_start, run_end)
                    if run_key not in seen_reference_runs:
                        seen_reference_runs.add(run_key)
                        run_indices = cls._prepend_heading_index(packets, list(range(run_start, run_end + 1)))
                        semantic_specs.append(
                            (
                                "reference_section_region",
                                "paper_reference_section_bundle",
                                run_indices,
                            )
                        )
                    continue

                if citation_context_count >= 2:
                    continue
                if cls._is_citation_context_packet(packet):
                    if focus_terms and not cls._packet_matches_focus(packet, focus_terms):
                        continue
                    bundle_indices = [index]
                    if index > 0:
                        previous = packets[index - 1]
                        if previous.packet_type in {"paragraph", "clause", "list_item"} and previous.section_path == packet.section_path:
                            bundle_indices.insert(0, index - 1)
                    if index + 1 < len(packets):
                        following = packets[index + 1]
                        if following.packet_type in {"paragraph", "clause", "list_item"} and following.section_path == packet.section_path:
                            bundle_indices.append(index + 1)
                    semantic_specs.append(
                        (
                            "citation_context_region",
                            "paper_citation_context",
                            cls._prepend_heading_index(packets, bundle_indices),
                        )
                    )
                    citation_context_count += 1

        if wants_legal:
            family_counts = {
                "legal_case_profile": 0,
                "legal_reasoning_bundle": 0,
                "legal_decision_bundle": 0,
            }
            for index, packet in enumerate(packets):
                if family_counts["legal_case_profile"] < 2 and cls._is_legal_case_profile_packet(packet, index):
                    bundle_indices = cls._extend_forward_bundle(
                        packets,
                        index,
                        max_extra=2,
                        allowed_types=("paragraph", "clause", "list_item"),
                    )
                    semantic_specs.append(
                        ("case_profile_region", "legal_case_profile", cls._prepend_heading_index(packets, bundle_indices))
                    )
                    family_counts["legal_case_profile"] += 1

                if family_counts["legal_reasoning_bundle"] < 1 and cls._is_legal_reasoning_packet(packet):
                    bundle_indices = cls._extend_forward_bundle(
                        packets,
                        index,
                        max_extra=2,
                        allowed_types=("paragraph", "clause", "list_item"),
                    )
                    semantic_specs.append(
                        ("reasoning_bundle_region", "legal_reasoning_bundle", cls._prepend_heading_index(packets, bundle_indices))
                    )
                    family_counts["legal_reasoning_bundle"] += 1

                if family_counts["legal_decision_bundle"] < 1 and cls._is_legal_decision_packet(packet):
                    bundle_indices = cls._extend_forward_bundle(
                        packets,
                        index,
                        max_extra=3,
                        allowed_types=("paragraph", "clause", "list_item"),
                    )
                    semantic_specs.append(
                        ("decision_bundle_region", "legal_decision_bundle", cls._prepend_heading_index(packets, bundle_indices))
                    )
                    family_counts["legal_decision_bundle"] += 1

        unique_specs: List[tuple[str, str, List[int]]] = []
        seen_specs: set[tuple[str, tuple[int, ...]]] = set()
        for region_type, unit_family, indices in semantic_specs:
            normalized_indices = sorted(dict.fromkeys(index for index in indices if 0 <= index < len(packets)))
            if not normalized_indices:
                continue
            key = (region_type, tuple(normalized_indices))
            if key in seen_specs:
                continue
            seen_specs.add(key)
            unique_specs.append((region_type, unit_family, normalized_indices))

        unique_specs.sort(key=lambda item: (item[2][0], len(item[2]), item[0]))

        family_limits = {
            "financial_row_bundle": 2,
            "financial_table_note_bundle": 2,
            "paper_reference_section_bundle": 2,
            "paper_citation_context": 2,
            "legal_case_profile": 2,
            "legal_reasoning_bundle": 1,
            "legal_decision_bundle": 1,
        }

        regions: List[DividerRegion] = []
        emitted_by_family: Dict[str, int] = {}
        for region_index, (region_type, unit_family, packet_indices) in enumerate(unique_specs, start=1):
            limit = family_limits.get(unit_family)
            current = emitted_by_family.get(unit_family, 0)
            if limit is not None and current >= limit:
                continue
            emitted_by_family[unit_family] = current + 1
            region_id = f"{cls._region_id_prefix('semantic', divider_mode)}{region_index}"
            region_packets = [packets[index] for index in packet_indices]
            regions.append(
                cls._build_region(
                    region_id,
                    region_packets,
                    granularity="semantic",
                    region_type_override=region_type,
                    unit_family=unit_family,
                )
            )
        return regions

    @staticmethod
    def _select_prompt_regions(regions: Sequence[DividerRegion]) -> List[DividerRegion]:
        meso_regions = [region for region in regions if region.granularity == "meso"]
        return meso_regions or list(regions)

    @classmethod
    def _packetize_document(
        cls,
        document_text: str,
        divider_mode: str = "default",
        task_text: str = "",
    ) -> Tuple[List[DividerPacket], List[DividerRegion], Dict[str, Any]]:
        divider_mode = cls._normalize_divider_mode(divider_mode)
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

        granularity_regions: List[DividerRegion] = []
        for granularity in cls._granularity_sequence(divider_mode):
            if granularity == "semantic":
                granularity_regions.extend(
                    cls._build_semantic_regions(
                        packets=packets,
                        divider_mode=divider_mode,
                        task_text=task_text,
                    )
                )
                continue
            granularity_regions.extend(
                cls._build_regions_for_granularity(
                    packets=packets,
                    granularity=granularity,
                    divider_mode=divider_mode,
                )
            )

        search_views = cls._build_search_views(
            packets,
            granularity_regions,
            divider_mode=divider_mode,
            prompt_regions=cls._select_prompt_regions(granularity_regions),
        )
        return packets, granularity_regions, search_views

    @classmethod
    def _build_region(
        cls,
        region_id: str,
        packets: Sequence[DividerPacket],
        granularity: str = "meso",
        region_type_override: str = "",
        unit_family: str = "",
    ) -> DividerRegion:
        packet_ids = [packet.packet_id for packet in packets]
        section_path = packets[0].section_path if packets else ""
        region_type = region_type_override or cls._infer_region_type(packets)
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
            granularity=granularity,
            unit_family=unit_family,
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
    def _build_search_views(
        cls,
        packets: Sequence[DividerPacket],
        regions: Sequence[DividerRegion],
        divider_mode: str,
        prompt_regions: Sequence[DividerRegion],
    ) -> Dict[str, Any]:
        granularity_counts = {
            granularity: sum(1 for region in regions if region.granularity == granularity)
            for granularity in sorted({region.granularity for region in regions})
        }
        unit_family_counts = {
            unit_family: sum(1 for region in regions if region.unit_family == unit_family)
            for unit_family in sorted({region.unit_family for region in regions if region.unit_family})
        }
        return {
            "divider_mode": divider_mode,
            "packet_type_counts": {
                packet_type: sum(1 for packet in packets if packet.packet_type == packet_type)
                for packet_type in sorted({packet.packet_type for packet in packets})
            },
            "region_type_counts": {
                region_type: sum(1 for region in regions if region.region_type == region_type)
                for region_type in sorted({region.region_type for region in regions})
            },
            "granularity_counts": granularity_counts,
            "unit_family_counts": unit_family_counts,
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
                    "granularity": region.granularity,
                    "unit_family": region.unit_family,
                    "section_path": region.section_path,
                    "packet_span": f"{region.start_packet_id}..{region.end_packet_id}",
                    "preview": region.preview,
                }
                for region in prompt_regions
            ],
            "all_region_inventory": [
                {
                    "region_id": region.region_id,
                    "region_type": region.region_type,
                    "granularity": region.granularity,
                    "unit_family": region.unit_family,
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
                f"- [{region.region_id}] granularity={region.granularity} | region_type={region.region_type} | unit_family={region.unit_family or 'none'} | section_path={region.section_path or 'ROOT'} | packet_span={region.start_packet_id}..{region.end_packet_id} | preview={region.preview}"
            )
        return "\n".join(lines)

    @classmethod
    def build_inputs_from_loong(
        cls,
        record: Dict[str, Any],
        sample_id_prefix: Optional[str] = None,
        divider_mode: str = "default",
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
            task_text = " ".join(
                [str(record.get("instruction", "")).strip(), str(record.get("question", "")).strip()]
            ).strip()
            packets, regions, search_views = cls._packetize_document(
                text,
                divider_mode=divider_mode,
                task_text=task_text,
            )
            prompt_regions = cls._select_prompt_regions(regions)
            inputs.append(
                DividerInput(
                    instruction=str(record.get("instruction", "")).strip(),
                    question=str(record.get("question", "")).strip(),
                    doc_id=doc_id,
                    display_title=title,
                    document_text=text,
                    packet_inventory=cls._format_packet_inventory(packets),
                    region_inventory=cls._format_region_inventory(prompt_regions),
                    sample_id=f"{prefix}_{doc_id}",
                    metadata={
                        "record_id": record.get("id"),
                        "level": record.get("level"),
                        "set": record.get("set"),
                        "type": record.get("type"),
                        "language": record.get("language"),
                        "divider_mode": divider_mode,
                        "packet_count": len(packets),
                        "region_count": len(regions),
                        "region_count_by_granularity": search_views.get("granularity_counts", {}),
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
        packets, regions, search_views = self._packetize_document(
            divider_input.document_text,
            divider_mode=self.divider_mode,
            task_text=" ".join([divider_input.instruction, divider_input.question]).strip(),
        )
        user_prompt = self.render_user_prompt(divider_input)
        raw_text = self.llm.generate(self.system_prompt, user_prompt)
        parsed = self.parse_response(raw_text)
        region_roles = self._parse_kv_bullets(parsed["Region Roles"])
        output = DividerOutput(
            raw_text=raw_text,
            doc_id=divider_input.doc_id,
            display_title=divider_input.display_title,
            divider_mode=self.divider_mode,
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
        for divider_input in self.build_inputs_from_loong(
            record,
            sample_id_prefix=sample_id_prefix,
            divider_mode=self.divider_mode,
        ):
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
            f"divider_mode={divider_input.metadata.get('divider_mode', 'default')}",
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
