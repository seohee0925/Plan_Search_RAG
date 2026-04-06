from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from dawon.agents.divider import DividerDocRun
    from dawon.agents.planner import OpenAIChatLLM, PlannerLLM
except ModuleNotFoundError:  # pragma: no cover
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from dawon.agents.divider import DividerDocRun  # type: ignore
    from dawon.agents.planner import OpenAIChatLLM, PlannerLLM  # type: ignore


REQUIRED_HEADERS = [
    "Search Trace",
    "Evidence Units",
    "Search Status",
]


@dataclass
class GoldenRetrieverInput:
    instruction: str
    question: str
    task_goal: str
    task_model: str
    answer_schema: str
    merge_policy: str
    search_target_name: str
    search_target_ask: str
    search_target_success_condition: str
    doc_id: str
    display_title: str
    doc_anchor: str
    focus_regions: str
    scoped_region_map: str
    scoped_packet_view: str
    sample_id: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GoldenRetrieverOutput:
    raw_text: str
    search_trace: List[Dict[str, str]]
    evidence_units: List[Dict[str, str]]
    search_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GoldenRetrieverStepRun:
    step_id: str
    doc_id: str
    trace_path: str
    retriever_input: GoldenRetrieverInput
    retriever_output: GoldenRetrieverOutput

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "doc_id": self.doc_id,
            "trace_path": self.trace_path,
            "input": asdict(self.retriever_input),
            "output": self.retriever_output.to_dict(),
        }


class GoldenRetrieverParseError(ValueError):
    pass


class GoldenRetriever:
    def __init__(
        self,
        llm: PlannerLLM,
        project_root: Optional[Path] = None,
        prompt_dir: str = "prompts",
        trace_dir: str = "trace_logs/golden_retriever",
    ) -> None:
        self.llm = llm
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        self.prompt_dir = self.project_root / prompt_dir
        self.trace_dir = self.project_root / trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.system_prompt = (self.prompt_dir / "golden_retriever_sys.txt").read_text(encoding="utf-8").strip()
        self.user_prompt_template = (self.prompt_dir / "golden_retriever_user.txt").read_text(encoding="utf-8").strip()

    @staticmethod
    def _safe_filename(name: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._")
        return safe or "unknown"

    @staticmethod
    def _format_bullets(items: Sequence[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for item in items:
            item_id = str(item.get("id", "")).strip()
            body = []
            for key, value in item.items():
                if key == "id":
                    continue
                body.append(f"{key}={value}")
            prefix = f"- [{item_id}] " if item_id else "- "
            lines.append(prefix + " | ".join(body))
        return "\n".join(lines)

    @staticmethod
    def _parse_bullets(section_text: str) -> List[Dict[str, str]]:
        if section_text.strip().lower() == "none":
            return []
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

    @staticmethod
    def _extract_sections(raw_text: str) -> Dict[str, str]:
        matches = list(re.finditer(r"^###\s+(.+?)\s*$", raw_text, flags=re.MULTILINE))
        if not matches:
            raise GoldenRetrieverParseError("No markdown section headers were found in retriever output.")
        sections: Dict[str, str] = {}
        for idx, match in enumerate(matches):
            header = match.group(1).strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
            sections[header] = raw_text[start:end].strip()
        missing = [header for header in REQUIRED_HEADERS if header not in sections]
        if missing:
            raise GoldenRetrieverParseError(f"Retriever output is missing required sections: {missing}")
        return sections

    @staticmethod
    def _scoped_packet_ids(divider_output: Dict[str, Any], focus_region_ids: Sequence[str]) -> List[str]:
        region_map = {
            str(region.get("region_id", "")).strip(): region
            for region in divider_output.get("region_store", [])
            if str(region.get("region_id", "")).strip()
        }
        packet_ids: List[str] = []
        for region_id in focus_region_ids:
            region = region_map.get(region_id)
            if not region:
                continue
            packet_ids.extend([str(packet_id).strip() for packet_id in region.get("packet_ids", []) if str(packet_id).strip()])
        return list(dict.fromkeys(packet_ids))

    @staticmethod
    def _expand_neighbor_packet_ids(divider_output: Dict[str, Any], packet_ids: Sequence[str]) -> List[str]:
        packet_map = {
            str(packet.get("packet_id", "")).strip(): packet
            for packet in divider_output.get("packet_store", [])
            if str(packet.get("packet_id", "")).strip()
        }
        expanded: List[str] = []
        for packet_id in packet_ids:
            packet = packet_map.get(packet_id)
            if not packet:
                continue
            prev_packet_id = str(packet.get("prev_packet_id", "")).strip()
            next_packet_id = str(packet.get("next_packet_id", "")).strip()
            if prev_packet_id:
                expanded.append(prev_packet_id)
            expanded.append(packet_id)
            if next_packet_id:
                expanded.append(next_packet_id)
        return list(dict.fromkeys(expanded))

    @classmethod
    def _build_scoped_views(
        cls,
        divider_output: Dict[str, Any],
        focus_region_ids: Sequence[str],
    ) -> tuple[str, str]:
        region_store = divider_output.get("region_store", [])
        packet_store = divider_output.get("packet_store", [])
        region_map = {
            str(region.get("region_id", "")).strip(): region
            for region in region_store
            if str(region.get("region_id", "")).strip()
        }
        packet_map = {
            str(packet.get("packet_id", "")).strip(): packet
            for packet in packet_store
            if str(packet.get("packet_id", "")).strip()
        }

        selected_region_ids = [region_id for region_id in focus_region_ids if region_id in region_map]
        if not selected_region_ids:
            selected_region_ids = [str(region.get("region_id", "")).strip() for region in region_store[:3] if str(region.get("region_id", "")).strip()]

        scoped_region_lines: List[str] = []
        for region_id in selected_region_ids:
            region = region_map[region_id]
            scoped_region_lines.append(
                f"- [{region_id}] region_type={region.get('region_type', '')} | granularity={region.get('granularity', '')} | unit_family={region.get('unit_family', '') or 'none'} | section_path={region.get('section_path', '') or 'ROOT'} | "
                f"packet_span={region.get('start_packet_id', '')}..{region.get('end_packet_id', '')} | preview={region.get('preview', '')}"
            )

        packet_ids = cls._scoped_packet_ids(divider_output, selected_region_ids)
        expanded_packet_ids = cls._expand_neighbor_packet_ids(divider_output, packet_ids)
        scoped_packet_lines: List[str] = []
        for packet_id in expanded_packet_ids:
            packet = packet_map.get(packet_id)
            if not packet:
                continue
            scoped_packet_lines.append(
                f"- [{packet_id}] type={packet.get('packet_type', '')} | section_path={packet.get('section_path', '') or 'ROOT'} | text={packet.get('text', '')}"
            )

        return "\n".join(scoped_region_lines), "\n".join(scoped_packet_lines)

    @classmethod
    def build_inputs_from_execution_graph(
        cls,
        record: Dict[str, Any],
        planner_output: Dict[str, Any],
        divider_runs: Sequence[DividerDocRun],
        sample_id_prefix: Optional[str] = None,
    ) -> List[tuple[str, GoldenRetrieverInput]]:
        target_by_name = {
            str(item.get("name", "")).strip(): item
            for item in planner_output.get("search_targets", [])
            if str(item.get("name", "")).strip()
        }
        divider_by_doc = {
            run.doc_id: run.divider_output.to_dict()
            for run in divider_runs
        }
        prefix = sample_id_prefix or str(record.get("id", "unknown"))

        inputs: List[tuple[str, GoldenRetrieverInput]] = []
        for step in planner_output.get("doc_execution_graph", []):
            step_id = str(step.get("id", "")).strip()
            doc_id = str(step.get("doc", "")).strip()
            search_target_name = str(step.get("search_target", "")).strip()
            divider_output = divider_by_doc.get(doc_id)
            if not step_id or not doc_id or not search_target_name or not divider_output:
                continue
            target = target_by_name.get(search_target_name, {})
            focus_region_ids = [part.strip() for part in str(step.get("focus_regions", "")).split(",") if part.strip() and part.strip() != "ALL"]
            scoped_region_map, scoped_packet_view = cls._build_scoped_views(divider_output, focus_region_ids)
            input_obj = GoldenRetrieverInput(
                instruction=str(record.get("instruction", "")).strip(),
                question=str(record.get("question", "")).strip(),
                task_goal=str(planner_output.get("task_goal", "")).strip(),
                task_model=cls._format_bullets(planner_output.get("task_model", [])),
                answer_schema=cls._format_bullets(planner_output.get("answer_schema", [])),
                merge_policy=str(planner_output.get("merge_policy", "")).strip(),
                search_target_name=search_target_name,
                search_target_ask=str(target.get("ask", "")).strip(),
                search_target_success_condition=str(target.get("success_condition", "")).strip(),
                doc_id=doc_id,
                display_title=str(divider_output.get("display_title", "")).strip(),
                doc_anchor=str(divider_output.get("doc_anchor", "")).strip(),
                focus_regions=",".join(focus_region_ids) if focus_region_ids else "AUTO",
                scoped_region_map=scoped_region_map,
                scoped_packet_view=scoped_packet_view,
                sample_id=f"{prefix}_{step_id}",
                metadata={
                    "record_id": record.get("id"),
                    "step_id": step_id,
                    "step_anchor": str(step.get("anchor", "")).strip(),
                    "read_strategy": str(step.get("read_strategy", "")).strip(),
                    "stop_when": str(step.get("stop_when", "")).strip(),
                    "focus_region_ids": focus_region_ids,
                    "all_region_ids": [
                        str(region.get("region_id", "")).strip()
                        for region in divider_output.get("region_store", [])
                        if str(region.get("region_id", "")).strip()
                    ],
                },
            )
            inputs.append((step_id, input_obj))
        return inputs

    def render_user_prompt(self, retriever_input: GoldenRetrieverInput) -> str:
        return self.user_prompt_template.format(
            instruction=retriever_input.instruction,
            question=retriever_input.question,
            task_goal=retriever_input.task_goal,
            task_model=retriever_input.task_model,
            answer_schema=retriever_input.answer_schema,
            merge_policy=retriever_input.merge_policy,
            search_target_name=retriever_input.search_target_name,
            search_target_ask=retriever_input.search_target_ask,
            search_target_success_condition=retriever_input.search_target_success_condition,
            doc_id=retriever_input.doc_id,
            display_title=retriever_input.display_title,
            doc_anchor=retriever_input.doc_anchor,
            focus_regions=retriever_input.focus_regions,
            scoped_region_map=retriever_input.scoped_region_map,
            scoped_packet_view=retriever_input.scoped_packet_view,
        )

    def retrieve(self, retriever_input: GoldenRetrieverInput, save_trace: bool = True) -> GoldenRetrieverOutput:
        user_prompt = self.render_user_prompt(retriever_input)
        raw_text = self.llm.generate(self.system_prompt, user_prompt)
        sections = self._extract_sections(raw_text)
        output = GoldenRetrieverOutput(
            raw_text=raw_text,
            search_trace=self._parse_bullets(sections["Search Trace"]),
            evidence_units=self._parse_bullets(sections["Evidence Units"]),
            search_status=sections["Search Status"].strip(),
        )
        if save_trace:
            self._save_trace(retriever_input, user_prompt, output)
        return output

    def run_execution_graph(
        self,
        record: Dict[str, Any],
        planner_output: Dict[str, Any],
        divider_runs: Sequence[DividerDocRun],
        sample_id_prefix: Optional[str] = None,
        save_trace: bool = True,
    ) -> List[GoldenRetrieverStepRun]:
        runs: List[GoldenRetrieverStepRun] = []
        for step_id, retriever_input in self.build_inputs_from_execution_graph(
            record=record,
            planner_output=planner_output,
            divider_runs=divider_runs,
            sample_id_prefix=sample_id_prefix,
        ):
            output = self.retrieve(retriever_input, save_trace=save_trace)
            runs.append(
                GoldenRetrieverStepRun(
                    step_id=step_id,
                    doc_id=retriever_input.doc_id,
                    trace_path=str(self.trace_path_for_sample_id(retriever_input.sample_id)),
                    retriever_input=retriever_input,
                    retriever_output=output,
                )
            )
        return runs

    def trace_path_for_sample_id(self, sample_id: str) -> Path:
        return self.trace_dir / f"{self._safe_filename(sample_id)}.json"

    def _save_trace(self, retriever_input: GoldenRetrieverInput, user_prompt: str, output: GoldenRetrieverOutput) -> Path:
        trace_path = self.trace_path_for_sample_id(retriever_input.sample_id)
        payload = {
            "input": asdict(retriever_input),
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt,
            "output": output.to_dict(),
        }
        trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return trace_path


__all__ = [
    "GoldenRetriever",
    "GoldenRetrieverInput",
    "GoldenRetrieverOutput",
    "GoldenRetrieverParseError",
    "GoldenRetrieverStepRun",
    "OpenAIChatLLM",
]
