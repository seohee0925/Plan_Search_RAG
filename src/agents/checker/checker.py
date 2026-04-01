from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from src.agents.planner.loong.planner import OpenAIChatLLM, PlannerLLM
except ModuleNotFoundError:  # pragma: no cover
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.agents.planner.loong.planner import OpenAIChatLLM, PlannerLLM  # type: ignore


REQUIRED_HEADERS = [
    "Evidence State Summary",
    "Verified Evidence Units",
    "Projected Answer State",
    "Slot Fill State",
    "Remaining Gaps",
    "Repair Requests",
    "Sufficiency Verdict",
]


@dataclass
class CheckerInput:
    instruction: str
    question: str
    task_goal: str
    task_model: str
    answer_schema: str
    merge_policy: str
    planning_notes: str
    document_catalog: str
    execution_graph: str
    integrated_evidence_state: str
    sample_id: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckerOutput:
    raw_text: str
    evidence_state_summary: str
    verified_evidence_units: List[Dict[str, str]]
    projected_answer_state: List[Dict[str, str]]
    slot_fill_state: List[Dict[str, str]]
    remaining_gaps: List[Dict[str, str]]
    repair_requests: List[Dict[str, str]]
    sufficiency_verdict: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CheckerParseError(ValueError):
    pass


class Checker:
    def __init__(
        self,
        llm: PlannerLLM,
        project_root: Optional[Path] = None,
        prompt_dir: str = "prompts",
        trace_dir: str = "trace_logs/checker",
    ) -> None:
        self.llm = llm
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[3]
        self.prompt_dir = self.project_root / prompt_dir
        self.trace_dir = self.project_root / trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.system_prompt = (self.prompt_dir / "checker_sys.txt").read_text(encoding="utf-8").strip()
        self.user_prompt_template = (self.prompt_dir / "checker_user.txt").read_text(encoding="utf-8").strip()

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
            raise CheckerParseError("No markdown section headers were found in checker output.")
        sections: Dict[str, str] = {}
        for idx, match in enumerate(matches):
            header = match.group(1).strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
            sections[header] = raw_text[start:end].strip()
        missing = [header for header in REQUIRED_HEADERS if header not in sections]
        if missing:
            raise CheckerParseError(f"Checker output is missing required sections: {missing}")
        return sections

    @staticmethod
    def _extract_numeric_value(text: str) -> Optional[float]:
        raw = str(text or "").strip()
        if not raw:
            return None
        parenthesized = bool(
            re.search(r"\(\s*[$¥€£]?\s*\d[\d,]*(?:\.\d+)?\s*(?:元|%|美元|股)?\s*\)", raw)
        )
        match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", raw)
        if not match:
            return None
        try:
            value = float(match.group(0).replace(",", ""))
        except Exception:
            return None
        if parenthesized and value > 0:
            value = -value
        return value

    @classmethod
    def _canonicalize_scalar(cls, text: str) -> str:
        raw = str(text or "").strip()
        numeric = cls._extract_numeric_value(raw)
        if numeric is None:
            return raw
        magnitude = f"{abs(numeric):f}".rstrip("0").rstrip(".")
        sign = "-" if numeric < 0 else ""
        if "$" in raw:
            return f"{sign}${magnitude}"
        if "元" in raw:
            return f"{sign}{magnitude}元"
        if "%" in raw:
            return f"{sign}{magnitude}%"
        return f"{sign}{magnitude}"

    @staticmethod
    def _extract_claim_subject(claim_text: str) -> Optional[str]:
        def strip_company_suffix(text: str) -> str:
            stripped = re.sub(r"(股份有限公司|有限责任公司|有限公司)$", "", text).strip()
            return stripped or text.strip()

        text = str(claim_text or "").strip()
        if not text or text.casefold() == "none":
            return None
        english_match = re.match(r"(.+?)\s+has\s+.+", text, flags=re.IGNORECASE)
        if english_match:
            return english_match.group(1).strip()
        chinese_match = re.match(r"(.+?)(?:具有|拥有|为|是).+", text)
        if chinese_match:
            return strip_company_suffix(chinese_match.group(1).strip())
        return None

    @staticmethod
    def _compact_doc_label(*candidates: str) -> str:
        def strip_company_suffix(text: str) -> str:
            stripped = re.sub(r"(股份有限公司|有限责任公司|有限公司)$", "", text).strip()
            return stripped or text.strip()

        for raw_candidate in candidates:
            text = str(raw_candidate or "").strip()
            if not text:
                continue
            english_report_match = re.search(r"(?:report|document)\s+(?:of|for)\s+(.+?)(?:,|\.|$)", text, flags=re.IGNORECASE)
            if english_report_match:
                return english_report_match.group(1).strip()
            chinese_report_match = re.search(r"([^，。,；;]+?)(?:股份有限公司|有限责任公司|有限公司)?20\d{2}年[^，。,；;]*报告", text)
            if chinese_report_match:
                return strip_company_suffix(chinese_report_match.group(1).strip())
            case_match = re.search(r"(因[^，。；;]{4,80}?一案)", text)
            if case_match:
                return case_match.group(1).strip()
            if not text.casefold().startswith("this document is "):
                return strip_company_suffix(text)
        return ""

    @classmethod
    def _build_integrated_evidence_state(cls, step_runs: Sequence[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for run in step_runs:
            input_obj = run.get("input", {})
            output_obj = run.get("output", {})
            input_meta = input_obj.get("metadata", {}) or {}
            lines.append(
                f"- [{run.get('step_id', '')}] doc={input_obj.get('doc_id', '')} | anchor={input_obj.get('doc_anchor', '')} | target={input_obj.get('search_target_name', '')}"
            )
            lines.append(f"  focus_regions: {input_obj.get('focus_regions', '')}")
            if input_meta:
                lines.append(
                    f"  plan_intent: read_strategy={input_meta.get('read_strategy', '')} | step_anchor={input_meta.get('step_anchor', '')} | stop_when={input_meta.get('stop_when', '')}"
                )
            lines.append("  evidence_units:")
            evidence_units = output_obj.get("evidence_units", []) or []
            if evidence_units:
                for line in cls._format_bullets(evidence_units).splitlines():
                    lines.append(f"  {line}")
            else:
                lines.append("  - none")
            lines.append(f"  search_status: {output_obj.get('search_status', '')}")
        return "\n".join(lines)

    @classmethod
    def build_input(
        cls,
        planner_trace: Dict[str, Any],
        divider_batch_trace: Dict[str, Any],
        retriever_batch_trace: Dict[str, Any],
        sample_id: Optional[str] = None,
    ) -> CheckerInput:
        planner_input = planner_trace.get("input", {})
        planner_output = planner_trace.get("output", {})
        divider_runs = divider_batch_trace.get("runs", [])
        retriever_runs = retriever_batch_trace.get("runs", [])
        catalog_lines = []
        for run in divider_runs:
            output = run.get("output", {})
            catalog_lines.append(
                f"- {output.get('doc_id', '')} | title={output.get('display_title', '')} | anchor={output.get('doc_anchor', '')} | packets={output.get('packet_count', '')} | regions={output.get('region_count', '')}"
            )

        return CheckerInput(
            instruction=str(planner_input.get("instruction", "")).strip(),
            question=str(planner_input.get("question", "")).strip(),
            task_goal=str(planner_output.get("task_goal", "")).strip(),
            task_model=cls._format_bullets(planner_output.get("task_model", [])),
            answer_schema=cls._format_bullets(planner_output.get("answer_schema", [])),
            merge_policy=str(planner_output.get("merge_policy", "")).strip(),
            planning_notes=str(planner_output.get("planning_notes", "")).strip(),
            document_catalog="\n".join(catalog_lines),
            execution_graph=cls._format_bullets(planner_output.get("doc_execution_graph", [])),
            integrated_evidence_state=cls._build_integrated_evidence_state(retriever_runs),
            sample_id=sample_id or f"{planner_input.get('sample_id', 'unknown')}_checker",
            metadata={
                "planner_output": planner_output,
                "divider_runs": divider_runs,
                "retriever_runs": retriever_runs,
            },
        )

    def render_user_prompt(self, checker_input: CheckerInput) -> str:
        return self.user_prompt_template.format(
            instruction=checker_input.instruction,
            question=checker_input.question,
            task_goal=checker_input.task_goal,
            task_model=checker_input.task_model,
            answer_schema=checker_input.answer_schema,
            merge_policy=checker_input.merge_policy,
            planning_notes=checker_input.planning_notes,
            document_catalog=checker_input.document_catalog,
            execution_graph=checker_input.execution_graph,
            integrated_evidence_state=checker_input.integrated_evidence_state,
        )

    @staticmethod
    def _search_target_coverage(retriever_runs: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        coverage: Dict[str, Dict[str, Any]] = {}
        for run in retriever_runs:
            input_obj = run.get("input", {})
            output_obj = run.get("output", {})
            target = str(input_obj.get("search_target_name", "")).strip()
            doc_id = str(input_obj.get("doc_id", "")).strip()
            if not target:
                continue
            statuses = [str(item.get("status", "")).strip() for item in output_obj.get("evidence_units", []) or []]
            coverage[target] = {
                "doc_id": doc_id,
                "search_status": str(output_obj.get("search_status", "")).strip(),
                "statuses": statuses,
            }
        return coverage

    @staticmethod
    def _schema_keys(answer_schema: Sequence[Dict[str, Any]]) -> List[str]:
        return [str(item.get("key", "")).strip() for item in answer_schema if str(item.get("key", "")).strip()]

    @staticmethod
    def _slot_state_is_unresolved(state: str) -> bool:
        normalized = str(state or "").strip().lower()
        return normalized.startswith("unresolved")

    @staticmethod
    def _slot_state_is_terminal(state: str) -> bool:
        normalized = str(state or "").strip().lower()
        return normalized in {
            "selected",
            "rejected_off_target",
            "rejected_no_evidence_after_scoped_search",
        }

    def check(self, checker_input: CheckerInput, save_trace: bool = True) -> CheckerOutput:
        user_prompt = self.render_user_prompt(checker_input)
        raw_text = self.llm.generate(self.system_prompt, user_prompt)
        sections = self._extract_sections(raw_text)
        output = CheckerOutput(
            raw_text=raw_text,
            evidence_state_summary=sections["Evidence State Summary"].strip(),
            verified_evidence_units=self._parse_bullets(sections["Verified Evidence Units"]),
            projected_answer_state=self._parse_bullets(sections["Projected Answer State"]),
            slot_fill_state=self._parse_bullets(sections["Slot Fill State"]),
            remaining_gaps=self._parse_bullets(sections["Remaining Gaps"]),
            repair_requests=self._parse_bullets(sections["Repair Requests"]),
            sufficiency_verdict=sections["Sufficiency Verdict"].strip().upper() == "TRUE",
        )
        output = self._apply_guardrails(checker_input, output)
        if save_trace:
            self._save_trace(checker_input, user_prompt, output)
        return output

    def _apply_guardrails(self, checker_input: CheckerInput, output: CheckerOutput) -> CheckerOutput:
        planner_output = checker_input.metadata.get("planner_output", {}) or {}
        retriever_runs = checker_input.metadata.get("retriever_runs", []) or []
        divider_runs = checker_input.metadata.get("divider_runs", []) or []
        task_model = planner_output.get("task_model", []) or []
        answer_schema = planner_output.get("answer_schema", []) or []
        task_model_primary = task_model[0] if task_model else {}
        answer_topology = str(task_model_primary.get("answer_topology", "")).strip().lower()
        atomic_decision = str(task_model_primary.get("atomic_decision", "")).strip().lower()
        coverage_mode = str(task_model_primary.get("coverage_mode", "")).strip().lower()
        schema_keys = self._schema_keys(answer_schema)

        # Closed-set projection guard.
        allowed_values = {
            str(run.get("output", {}).get("display_title", "")).strip()
            for run in divider_runs
            if str(run.get("output", {}).get("display_title", "")).strip()
        }
        filtered_projected: List[Dict[str, str]] = []
        for item in output.projected_answer_state:
            value = str(item.get("value", "")).strip()
            if answer_topology in {"mapping", "grouping", "set", "list"} and atomic_decision == "relation_judgment":
                if value and value not in allowed_values and not value.startswith("["):
                    continue
            filtered_projected.append(item)
        output.projected_answer_state = filtered_projected

        # Deterministic ranking fallback from grounded evidence units when a single numeric winner is explicit.
        if atomic_decision == "comparison" or answer_topology == "ranking":
            numeric_claims: List[tuple[float, str, str]] = []
            for run in retriever_runs:
                input_obj = run.get("input", {})
                output_obj = run.get("output", {})
                input_meta = input_obj.get("metadata", {}) or {}
                for evidence_unit in output_obj.get("evidence_units", []) or []:
                    claim_text = str(evidence_unit.get("extracted_text", "")).strip()
                    numeric = self._extract_numeric_value(claim_text)
                    if numeric is not None:
                        label = (
                            self._extract_claim_subject(claim_text)
                            or self._compact_doc_label(
                                str(input_meta.get("step_anchor", "")).strip(),
                                str(input_obj.get("doc_anchor", "")).strip(),
                                str(input_obj.get("display_title", "")).strip(),
                            )
                            or str(input_obj.get("display_title", "")).strip()
                        )
                        basis = str(evidence_unit.get("packet_refs", "")).strip() or str(evidence_unit.get("id", "")).strip()
                        numeric_claims.append((numeric, label, basis))
                        break
            if numeric_claims:
                numeric_claims.sort(key=lambda item: item[0], reverse=True)
                top_value, top_label, top_basis = numeric_claims[0]
                tied = [item for item in numeric_claims if item[0] == top_value]
                if len(tied) == 1:
                    output.projected_answer_state = [
                        {
                            "id": "STATE_AUTO_RANK_1",
                            "answer_key": schema_keys[0] if schema_keys else "answer",
                            "value": top_label,
                            "basis": top_basis or "local numeric comparison",
                        }
                    ]
                    output.remaining_gaps = []
                    output.repair_requests = []
                    output.sufficiency_verdict = True

        if answer_topology == "scalar":
            for item in output.projected_answer_state:
                item["value"] = self._canonicalize_scalar(str(item.get("value", "")).strip())

        coverage = self._search_target_coverage(retriever_runs)
        expected_targets = {
            str(item.get("name", "")).strip()
            for item in planner_output.get("search_targets", []) or []
            if str(item.get("name", "")).strip()
        }
        if coverage_mode == "exhaustive" and expected_targets and coverage.keys() != expected_targets:
            output.sufficiency_verdict = False
            if not output.remaining_gaps:
                output.remaining_gaps = [
                    {
                        "id": "GAP_AUTO_1",
                        "answer_key": "all",
                        "issue": "missing_planned_search_targets",
                        "why": "Not every planned search target produced a retriever packet.",
                    }
                ]

        if coverage_mode == "exhaustive" and atomic_decision == "relation_judgment":
            unresolved = [
                target
                for target, snapshot in coverage.items()
                if snapshot["search_status"] == "partial_search"
            ]
            if unresolved:
                output.sufficiency_verdict = False
                if not output.remaining_gaps:
                    output.remaining_gaps = [
                    {
                        "id": "GAP_AUTO_2",
                        "answer_key": "all",
                        "issue": "partial_relation_search",
                        "why": "At least one planned edge search did not complete cleanly.",
                        }
                    ]

        if coverage_mode == "exhaustive" and atomic_decision == "assignment" and schema_keys:
            projected_keys = {
                str(item.get("answer_key", "")).strip()
                for item in output.projected_answer_state
                if str(item.get("answer_key", "")).strip()
            }
            if len(projected_keys) < len(schema_keys):
                output.sufficiency_verdict = False
                if not output.remaining_gaps:
                    output.remaining_gaps = [
                        {
                            "id": "GAP_AUTO_3",
                            "answer_key": "all",
                            "issue": "incomplete_assignment_projection",
                            "why": "Not every required answer key is present in the projected answer state.",
                    }
                ]

        expected_doc_order: List[tuple[str, str]] = []
        for run in divider_runs:
            doc_id = str(run.get("output", {}).get("doc_id", "")).strip()
            title = str(run.get("output", {}).get("display_title", "")).strip()
            if doc_id:
                expected_doc_order.append((doc_id, title))

        projected_values = {
            str(item.get("value", "")).strip()
            for item in output.projected_answer_state
            if str(item.get("value", "")).strip()
        }
        retriever_by_doc: Dict[str, Dict[str, Any]] = {}
        doc_to_step: Dict[str, str] = {}
        for run in retriever_runs:
            input_obj = run.get("input", {}) or {}
            output_obj = run.get("output", {}) or {}
            input_meta = input_obj.get("metadata", {}) or {}
            doc_id = str(input_obj.get("doc_id", "")).strip()
            if not doc_id:
                continue
            retriever_by_doc[doc_id] = {
                "search_status": str(output_obj.get("search_status", "")).strip(),
                "evidence_units": output_obj.get("evidence_units", []) or [],
            }
            step_id = str(input_meta.get("step_id", "")).strip() or str(run.get("step_id", "")).strip()
            if step_id:
                doc_to_step[doc_id] = step_id

        slot_by_doc: Dict[str, Dict[str, str]] = {}
        for item in output.slot_fill_state:
            doc_id = str(item.get("doc", "")).strip()
            if doc_id:
                slot_by_doc[doc_id] = dict(item)

        if coverage_mode == "exhaustive" and answer_topology in {"list", "set"} and expected_doc_order:
            normalized_slots: List[Dict[str, str]] = []
            unresolved_slots: List[Dict[str, str]] = []
            for index, (doc_id, title) in enumerate(expected_doc_order, start=1):
                slot = dict(slot_by_doc.get(doc_id, {}))
                search_snapshot = retriever_by_doc.get(doc_id, {})
                state = str(slot.get("state", "")).strip().lower()
                basis = str(slot.get("basis", "")).strip()
                note = str(slot.get("note", "")).strip()
                if not state:
                    if title and title in projected_values:
                        state = "selected"
                        basis = basis or next(
                            (
                                str(item.get("basis", "")).strip()
                                for item in output.projected_answer_state
                                if str(item.get("value", "")).strip() == title
                            ),
                            "",
                        )
                        note = note or "Projected into the current answer state."
                    else:
                        search_status = str(search_snapshot.get("search_status", "")).strip()
                        evidence_units = search_snapshot.get("evidence_units", []) or []
                        if search_status == "scoped_no_hit" or (
                            search_status == "scoped_complete" and not evidence_units
                        ):
                            state = "rejected_no_evidence_after_scoped_search"
                            note = note or "Scoped search completed without usable evidence."
                        elif search_status == "scoped_with_evidence":
                            state = "unresolved_scope_gap"
                            note = note or "Current scoped evidence still needs checker selection."
                        else:
                            state = "unresolved_scope_gap"
                            note = note or "This document slot is not yet resolved."

                if state == "rejected_no_evidence_after_scoped_search":
                    search_status = str(search_snapshot.get("search_status", "")).strip()
                    evidence_units = search_snapshot.get("evidence_units", []) or []
                    if search_status != "scoped_no_hit" and evidence_units:
                        state = "unresolved_scope_gap"
                        note = note or "Usable or ambiguous evidence exists, so this slot cannot be closed as no-evidence yet."

                if state == "selected" and title and title not in projected_values:
                    state = "unresolved_conflict"
                    note = note or "Marked selected without matching projected answer value."

                normalized = {
                    "id": str(slot.get("id", "")).strip() or f"SLOT_AUTO_{index}",
                    "doc": doc_id,
                    "title": str(slot.get("title", "")).strip() or title,
                    "state": state,
                    "basis": basis or doc_to_step.get(doc_id, ""),
                    "note": note,
                }
                normalized_slots.append(normalized)
                if self._slot_state_is_unresolved(state):
                    unresolved_slots.append(normalized)

            output.slot_fill_state = normalized_slots

            if unresolved_slots:
                output.sufficiency_verdict = False
                if not output.remaining_gaps:
                    output.remaining_gaps = [
                        {
                            "id": f"GAP_AUTO_SLOT_{idx}",
                            "answer_key": schema_keys[0] if schema_keys else "all",
                            "issue": f"{slot['doc']} remains {slot['state']}",
                            "why": slot.get("note", "") or slot.get("basis", "") or "This document slot is still unresolved.",
                        }
                        for idx, slot in enumerate(unresolved_slots, start=1)
                    ]
                output.repair_requests = [
                    {
                        "id": f"REPAIR_AUTO_SLOT_{idx}",
                        "action": "revisit_step",
                        "target": slot.get("basis", "") or slot.get("doc", ""),
                        "why": slot.get("note", "") or f"{slot.get('doc', '')} is still unresolved.",
                    }
                    for idx, slot in enumerate(unresolved_slots, start=1)
                ]
            elif all(self._slot_state_is_terminal(slot.get("state", "")) for slot in output.slot_fill_state):
                output.sufficiency_verdict = True
                output.remaining_gaps = []
                output.repair_requests = []

        if not output.sufficiency_verdict and not output.remaining_gaps:
            output.remaining_gaps = [
                {
                    "id": "GAP_AUTO_4",
                    "answer_key": "all",
                    "issue": "checker_state_inconsistent",
                    "why": "The verdict is FALSE but no explicit remaining gap was recorded.",
                }
            ]

        return output

    def _save_trace(self, checker_input: CheckerInput, user_prompt: str, output: CheckerOutput) -> Path:
        trace_path = self.trace_dir / f"{self._safe_filename(checker_input.sample_id)}.json"
        payload = {
            "input": asdict(checker_input),
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt,
            "output": output.to_dict(),
        }
        trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return trace_path


__all__ = [
    "Checker",
    "CheckerInput",
    "CheckerOutput",
    "CheckerParseError",
    "OpenAIChatLLM",
]
