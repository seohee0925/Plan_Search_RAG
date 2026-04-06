from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


TITLE_START = "<标题起始符>"
TITLE_END = "<标题终止符>"

REQUIRED_HEADERS = [
    "Task Goal",
    "Task Model",
    "Answer Schema",
    "Search Targets",
    "Document Execution Graph",
    "Merge Policy",
    "Planning Notes",
]


class PlannerLLM(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...


@dataclass
class PlannerInput:
    instruction: str
    question: str
    document_catalog: str
    document_catalog_items: List[Dict[str, Any]] = field(default_factory=list)
    checker_feedback_or_none: str = "None"
    sample_id: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannerOutput:
    raw_text: str
    task_goal: str
    task_model: List[Dict[str, str]]
    answer_schema: List[Dict[str, str]]
    search_targets: List[Dict[str, str]]
    doc_execution_graph: List[Dict[str, str]]
    merge_policy: str
    planning_notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PlannerParseError(ValueError):
    pass


class OpenAIChatLLM:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        if load_dotenv is not None:
            load_dotenv()

        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required. Install it with `pip install openai python-dotenv`."
            ) from exc

        resolved_api_key = api_key or self._first_env(
            "OPENAI_API_KEY",
            "API_KEY",
            "OPENAI_TOKEN",
            "TOKEN",
            "API_TOKEN",
        )
        resolved_model = model or self._first_env(
            "PLANNER_MODEL",
            "OPENAI_MODEL",
            "MODEL_NAME",
            "MODEL",
            default="gpt-4o",
        )
        resolved_base_url = base_url or self._first_env("OPENAI_BASE_URL", "BASE_URL")

        if not resolved_api_key:
            raise ValueError(
                "API key not found in .env. Set OPENAI_API_KEY or API_KEY or OPENAI_TOKEN."
            )

        self._client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url, timeout=timeout)
        self._model = resolved_model
        self._temperature = temperature

    @staticmethod
    def _first_env(*names: str, default: Optional[str] = None) -> Optional[str]:
        for name in names:
            value = os.getenv(name)
            if value:
                return value
        return default

    @property
    def model(self) -> str:
        return self._model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Planner LLM returned empty content.")
        return content


class Planner:
    def __init__(
        self,
        llm: PlannerLLM,
        project_root: Optional[Path] = None,
        prompt_dir: str = "prompts",
        trace_dir: str = "trace_logs/planner",
    ) -> None:
        self.llm = llm
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        self.prompt_dir = self.project_root / prompt_dir
        self.trace_dir = self.project_root / trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.system_prompt = (self.prompt_dir / "planner_sys.txt").read_text(encoding="utf-8").strip()
        self.user_prompt_template = (self.prompt_dir / "planner_user.txt").read_text(encoding="utf-8").strip()

    @staticmethod
    def _safe_filename(name: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._")
        return safe or "unknown"

    @staticmethod
    def _normalize_title(title: str) -> str:
        return re.sub(r"\s+", " ", title).strip()

    @staticmethod
    def split_loong_docs(raw_docs: str) -> Tuple[List[str], List[str]]:
        if not raw_docs:
            return [], []
        parts = raw_docs.split(TITLE_START)
        contents: List[str] = []
        titles: List[str] = []
        for part in parts[1:]:
            if TITLE_END not in part:
                continue
            title, content = part.split(TITLE_END, 1)
            title = Planner._normalize_title(title)
            content = content.strip()
            if title:
                titles.append(title)
                contents.append(content)
        return contents, titles

    @staticmethod
    def build_doc_title_bundle(titles: Sequence[str]) -> str:
        return "\n".join(
            f"- DOC{idx}: {Planner._normalize_title(title)}"
            for idx, title in enumerate(titles, start=1)
            if str(title).strip()
        )

    @classmethod
    def from_divider_runs(
        cls,
        record: Dict[str, Any],
        divider_runs: Sequence[Any],
        checker_feedback_or_none: str = "None",
        sample_id: Optional[str] = None,
    ) -> PlannerInput:
        catalog_items: List[Dict[str, Any]] = []
        catalog_lines: List[str] = []
        for run in divider_runs:
            output = run.divider_output
            role_map = {
                str(item.get("region_id", "")).strip(): str(item.get("role", "")).strip()
                for item in output.region_roles
                if str(item.get("region_id", "")).strip()
            }
            region_descriptors = []
            for region in output.region_store[:8]:
                region_id = str(region.get("region_id", "")).strip()
                role = role_map.get(region_id, str(region.get("region_type", "")).strip())
                granularity = str(region.get("granularity", "meso")).strip()
                unit_family = str(region.get("unit_family", "")).strip()
                section = str(region.get("section_path", "")).strip() or "ROOT"
                preview = str(region.get("preview", "")).strip()
                preview = re.sub(r"\s+", " ", preview)
                if len(preview) > 90:
                    preview = preview[:87].rstrip() + "..."
                family_suffix = f"/{unit_family}" if unit_family else ""
                region_descriptors.append(f"{region_id}:{role}/{granularity}{family_suffix}@{section}[{preview}]")
            catalog_items.append(
                {
                    "doc_id": output.doc_id,
                    "display_title": output.display_title,
                    "doc_anchor": output.doc_anchor,
                    "packet_count": output.packet_count,
                    "region_count": output.region_count,
                    "region_store": output.region_store,
                    "region_roles": output.region_roles,
                    "search_views": output.search_views,
                }
            )
            catalog_lines.append(
                f"- {output.doc_id} | title={output.display_title} | anchor={output.doc_anchor} | "
                f"packets={output.packet_count} | regions={output.region_count} | "
                f"region_index={'; '.join(region_descriptors) if region_descriptors else 'none'}"
            )

        return PlannerInput(
            instruction=str(record.get("instruction", "")).strip(),
            question=str(record.get("question", "")).strip(),
            document_catalog="\n".join(catalog_lines),
            document_catalog_items=catalog_items,
            checker_feedback_or_none=checker_feedback_or_none,
            sample_id=sample_id or str(record.get("id", "unknown")),
            metadata={
                "record_id": record.get("id"),
                "level": record.get("level"),
                "set": record.get("set"),
                "type": record.get("type"),
                "language": record.get("language"),
                "doc_count": len(catalog_items),
            },
        )

    @classmethod
    def from_loong_record(
        cls,
        record: Dict[str, Any],
        checker_feedback_or_none: str = "None",
    ) -> PlannerInput:
        raw_docs = str(record.get("docs", "") or "")
        _, titles = cls.split_loong_docs(raw_docs)
        catalog_lines = [
            f"- DOC{idx} | title={title} | anchor={title} | packets=unknown | regions=unknown | region_index=unknown"
            for idx, title in enumerate(titles, start=1)
        ]
        return PlannerInput(
            instruction=str(record.get("instruction", "")).strip(),
            question=str(record.get("question", "")).strip(),
            document_catalog="\n".join(catalog_lines),
            checker_feedback_or_none=checker_feedback_or_none,
            sample_id=str(record.get("id", "unknown")),
            metadata={
                "record_id": record.get("id"),
                "level": record.get("level"),
                "set": record.get("set"),
                "type": record.get("type"),
                "language": record.get("language"),
                "doc_count": len(titles),
            },
        )

    def render_user_prompt(self, planner_input: PlannerInput) -> str:
        return self.user_prompt_template.format(
            instruction=planner_input.instruction,
            question=planner_input.question,
            document_catalog=planner_input.document_catalog,
            checker_feedback_or_none=planner_input.checker_feedback_or_none,
        )

    @staticmethod
    def _extract_markdown_sections(raw_text: str) -> Dict[str, str]:
        matches = list(re.finditer(r"^###\s+(.+?)\s*$", raw_text, flags=re.MULTILINE))
        if not matches:
            return {}
        sections: Dict[str, str] = {}
        for idx, match in enumerate(matches):
            header = match.group(1).strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
            sections[header] = raw_text[start:end].strip()
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

    @staticmethod
    def _normalize_relation_text(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text.casefold())).strip()

    @classmethod
    def _head_variant(cls, title: str) -> str:
        head = re.split(r"\s*(?::|-|–|—)\s*", title, maxsplit=1)[0].strip()
        return head or title.strip()

    @staticmethod
    def _find_named_target(question: str, catalog_items: Sequence[Dict[str, Any]]) -> Optional[str]:
        normalized_question = Planner._normalize_relation_text(question)
        if not normalized_question:
            return None
        for item in catalog_items:
            title = str(item.get("display_title", "")).strip()
            if Planner._normalize_relation_text(title) == normalized_question:
                return str(item.get("doc_id", "")).strip()
        return None

    @staticmethod
    def _answer_keys_by_semantics(answer_schema: Sequence[Dict[str, str]]) -> Dict[str, str]:
        keys: Dict[str, str] = {}
        for item in answer_schema:
            key = str(item.get("key", "")).strip()
            meaning = str(item.get("meaning", "")).casefold()
            if "reference" in meaning or "reference" in key.casefold():
                keys["reference"] = key
            if "citation" in meaning or "cite" in meaning or "citation" in key.casefold():
                keys["citation"] = key
        return keys

    @staticmethod
    def _region_role_map(doc_item: Dict[str, Any]) -> Dict[str, str]:
        role_map: Dict[str, str] = {}
        for item in doc_item.get("region_roles", []) or []:
            region_id = str(item.get("region_id", "")).strip()
            if not region_id:
                continue
            role = str(item.get("role", "")).strip()
            why = str(item.get("why", "")).strip()
            role_map[region_id] = f"{role} {why}".strip().casefold()
        return role_map

    @classmethod
    def _choose_generic_regions(
        cls,
        doc_item: Dict[str, Any],
        step: Dict[str, str],
        search_target: Dict[str, str],
        task_model_primary: Dict[str, str],
    ) -> List[str]:
        region_store = doc_item.get("region_store", []) or []
        if not region_store:
            return []

        role_map = cls._region_role_map(doc_item)
        existing = [
            part.strip()
            for part in str(step.get("focus_regions", "")).split(",")
            if part.strip() and part.strip() != "ALL"
        ]
        atomic_decision = str(task_model_primary.get("atomic_decision", "")).strip().lower()
        answer_topology = str(task_model_primary.get("answer_topology", "")).strip().lower()
        ask_text = " ".join(
            [
                str(search_target.get("name", "")).strip(),
                str(search_target.get("ask", "")).strip(),
                str(search_target.get("success_condition", "")).strip(),
            ]
        ).casefold()
        numeric_cues = bool(
            re.search(r"\b(number|amount|value|rate|ratio|share|asset|assets|revenue|eps|profit|loss|price|count)\b", ask_text)
            or re.search(r"(金额|数值|比例|股|资产|收入|利润|损失|价格|数量)", ask_text)
        )
        classification_cues = bool(
            re.search(r"\b(case|type|category|classify|which|title|titles|whether|belongs)\b", ask_text)
            or re.search(r"(案由|类型|类别|哪些|是否|属于|标题)", ask_text)
        )

        scored: List[tuple[int, int, str, str]] = []
        for index, region in enumerate(region_store):
            region_id = str(region.get("region_id", "")).strip()
            if not region_id:
                continue
            region_type = str(region.get("region_type", "")).strip().casefold()
            granularity = str(region.get("granularity", "meso")).strip().casefold()
            unit_family = str(region.get("unit_family", "")).strip().casefold()
            section_path = str(region.get("section_path", "")).strip().casefold()
            role_text = role_map.get(region_id, "")
            combined = f"{region_type} {unit_family} {granularity} {section_path} {role_text}"
            score = 0
            if region_id in existing:
                score += 100
            if atomic_decision == "comparison" or answer_topology == "ranking" or numeric_cues:
                if any(token in combined for token in ["table", "comparative", "numeric"]):
                    score += 10
                    if granularity == "macro":
                        score += 6
                    elif granularity == "meso":
                        score += 4
                    elif granularity == "semantic":
                        score += 4
                    else:
                        score += 1
                if any(token in combined for token in ["reason", "decision", "note", "fallback"]):
                    score += 4
                    if granularity == "meso":
                        score += 3
                    elif granularity == "macro":
                        score += 2
                    elif granularity == "semantic":
                        score += 3
                if any(token in combined for token in ["financial_row_bundle", "financial_table_bundle", "financial_table_note_bundle", "table_row_bundle"]):
                    score += 8
                if "metadata" in combined:
                    score += 2
            else:
                if any(token in combined for token in ["issue", "reason", "decision", "disposition", "lead", "fallback"]):
                    score += 10
                    if granularity == "meso":
                        score += 5
                    elif granularity == "macro":
                        score += 3
                    elif granularity == "semantic":
                        score += 5
                    else:
                        score += 1
                if classification_cues and "metadata" in combined:
                    score += 4
                if classification_cues and "table" in combined:
                    score += 2
                if classification_cues and any(
                    token in combined
                    for token in ["legal_case_profile", "legal_reasoning_bundle", "legal_decision_bundle", "case_profile_region", "reasoning_bundle_region", "decision_bundle_region"]
                ):
                    score += 8
            if atomic_decision == "relation_judgment":
                if granularity == "semantic":
                    score += 5
                elif granularity == "meso":
                    score += 3
                elif granularity == "macro":
                    score += 1
                if any(token in combined for token in ["paper_reference_section_bundle", "paper_citation_context", "reference_section_region", "citation_context_region"]):
                    score += 8
            if not score:
                if "metadata" in combined:
                    score = 2
                elif "paragraph" in combined:
                    score = 1
            scored.append((score, -index, region_id, combined))

        scored.sort(reverse=True)
        chosen: List[str] = []
        seen_kinds: set[str] = set()
        for _, _, region_id, combined in scored:
            kind = "metadata"
            if any(token in combined for token in ["issue", "reason", "decision", "disposition", "lead", "fallback"]):
                kind = "reasoning"
            elif any(token in combined for token in ["table", "comparative", "numeric"]):
                kind = "structured"
            elif any(token in combined for token in ["attribution", "reference", "bibliography"]):
                kind = "attribution"
            if region_id in chosen:
                continue
            if kind in seen_kinds and len(chosen) >= 2:
                continue
            chosen.append(region_id)
            seen_kinds.add(kind)
            if len(chosen) >= 3:
                break

        return chosen or existing[:1]

    @staticmethod
    def _choose_relation_regions(doc_item: Dict[str, Any]) -> List[str]:
        chosen: List[str] = []
        role_texts = {
            str(item.get("region_id", "")).strip(): (
                f"{item.get('role', '')} {item.get('why', '')}".casefold()
            )
            for item in doc_item.get("region_roles", [])
            if str(item.get("region_id", "")).strip()
        }
        for region in doc_item.get("region_store", []):
            region_id = str(region.get("region_id", "")).strip()
            region_type = str(region.get("region_type", "")).casefold()
            granularity = str(region.get("granularity", "meso")).casefold()
            unit_family = str(region.get("unit_family", "")).casefold()
            section_path = str(region.get("section_path", "")).casefold()
            role_text = role_texts.get(region_id, "")
            if (
                "attribution" in region_type
                or "reference" in section_path
                or "bibliography" in section_path
                or "attribution" in role_text
                or "paper_reference_section_bundle" in unit_family
                or "paper_citation_context" in unit_family
                or "reference_section_region" in region_type
                or "citation_context_region" in region_type
            ):
                priority = 0
                if "paper_reference_section_bundle" in unit_family or "reference_section_region" in region_type:
                    priority = 5
                elif "paper_citation_context" in unit_family or "citation_context_region" in region_type:
                    priority = 4
                elif granularity == "meso":
                    priority = 3
                elif granularity == "semantic":
                    priority = 2
                else:
                    priority = 1
                chosen.append((priority, region_id))
        if chosen:
            chosen.sort(reverse=True)
            return [region_id for _, region_id in chosen[:4]]
        return [
            str(region.get("region_id", "")).strip()
            for region in doc_item.get("region_store", [])[:3]
            if str(region.get("region_id", "")).strip()
        ]

    @classmethod
    def _postprocess_relation_graph(
        cls,
        planner_input: PlannerInput,
        output: PlannerOutput,
    ) -> PlannerOutput:
        task_model_primary = output.task_model[0] if output.task_model else {}
        if str(task_model_primary.get("atomic_decision", "")).strip().lower() != "relation_judgment":
            return output
        if str(task_model_primary.get("search_regime", "")).strip().lower() != "closed_set":
            return output
        if str(task_model_primary.get("coverage_mode", "")).strip().lower() != "exhaustive":
            return output

        target_doc_id = cls._find_named_target(planner_input.question, planner_input.document_catalog_items)
        if not target_doc_id:
            return output

        title_by_doc = {
            str(item.get("doc_id", "")).strip(): str(item.get("display_title", "")).strip()
            for item in planner_input.document_catalog_items
        }
        doc_item_by_id = {
            str(item.get("doc_id", "")).strip(): item for item in planner_input.document_catalog_items
        }
        answer_keys = cls._answer_keys_by_semantics(output.answer_schema)
        reference_key = answer_keys.get("reference", "Reference")
        citation_key = answer_keys.get("citation", "Citation")
        target_title = title_by_doc.get(target_doc_id, "")
        target_head = cls._head_variant(target_title)

        search_targets: List[Dict[str, str]] = []
        steps: List[Dict[str, str]] = []

        for doc_id, title in title_by_doc.items():
            if doc_id == target_doc_id:
                for candidate_doc_id, candidate_title in title_by_doc.items():
                    if candidate_doc_id == target_doc_id:
                        continue
                    candidate_head = cls._head_variant(candidate_title)
                    target_name = f"EDGE_{doc_id}_TO_{candidate_doc_id}"
                    search_targets.append(
                        {
                            "id": f"TARGET_{doc_id}_{candidate_doc_id}",
                            "name": target_name,
                            "ask": (
                                f"Does {title} explicitly reference the provided candidate {candidate_title} "
                                f"or its unique short variant {candidate_head}?"
                            ),
                            "evidence_shape": "reference_entry_or_short_paragraph",
                            "success_condition": f"grounded outgoing relation for answer key {reference_key}",
                        }
                    )
                    focus_regions = ",".join(cls._choose_relation_regions(doc_item_by_id.get(doc_id, {})))
                    steps.append(
                        {
                            "id": f"STEP_{doc_id}_{candidate_doc_id}",
                            "doc": doc_id,
                            "anchor": str(doc_item_by_id.get(doc_id, {}).get("doc_anchor", title)).strip(),
                            "search_target": target_name,
                            "focus_regions": focus_regions or "ALL",
                            "read_strategy": "locate_then_read_contiguous",
                            "stop_when": "find grounded target-specific outgoing relation evidence or explicit scoped no-hit",
                        }
                    )
            else:
                target_name = f"EDGE_{doc_id}_TO_{target_doc_id}"
                search_targets.append(
                    {
                        "id": f"TARGET_{doc_id}_{target_doc_id}",
                        "name": target_name,
                        "ask": (
                            f"Does {title} explicitly cite or reference the target document {target_title} "
                            f"or its unique short variant {target_head}?"
                        ),
                        "evidence_shape": "reference_entry_or_short_paragraph",
                        "success_condition": f"grounded incoming relation for answer key {citation_key}",
                    }
                )
                focus_regions = ",".join(cls._choose_relation_regions(doc_item_by_id.get(doc_id, {})))
                steps.append(
                    {
                        "id": f"STEP_{doc_id}_{target_doc_id}",
                        "doc": doc_id,
                        "anchor": str(doc_item_by_id.get(doc_id, {}).get("doc_anchor", title)).strip(),
                        "search_target": target_name,
                        "focus_regions": focus_regions or "ALL",
                        "read_strategy": "locate_then_read_contiguous",
                        "stop_when": "find grounded target-specific incoming relation evidence or explicit scoped no-hit",
                    }
                )

        if search_targets and steps:
            output.search_targets = search_targets
            output.doc_execution_graph = steps
            output.merge_policy = (
                f"For {reference_key}, collect only outgoing links verified from {target_doc_id}. "
                f"For {citation_key}, collect only non-target documents that themselves ground an incoming link to {target_title}. "
                "Ignore out-of-set titles even when grounded in the document. "
                "Because coverage is exhaustive, unresolved edges remain gaps until all planned edge checks finish."
            )
        return output

    @classmethod
    def _postprocess_execution_graph(
        cls,
        planner_input: PlannerInput,
        output: PlannerOutput,
    ) -> PlannerOutput:
        task_model_primary = output.task_model[0] if output.task_model else {}
        if not output.doc_execution_graph:
            return output

        target_by_name = {
            str(item.get("name", "")).strip(): item
            for item in output.search_targets
            if str(item.get("name", "")).strip()
        }
        doc_item_by_id = {
            str(item.get("doc_id", "")).strip(): item
            for item in planner_input.document_catalog_items
            if str(item.get("doc_id", "")).strip()
        }
        coverage_mode = str(task_model_primary.get("coverage_mode", "")).strip().lower()
        atomic_decision = str(task_model_primary.get("atomic_decision", "")).strip().lower()

        updated_steps: List[Dict[str, str]] = []
        for step in output.doc_execution_graph:
            doc_id = str(step.get("doc", "")).strip()
            search_target_name = str(step.get("search_target", "")).strip()
            doc_item = doc_item_by_id.get(doc_id)
            search_target = target_by_name.get(search_target_name, {})
            if not doc_item:
                updated_steps.append(step)
                continue

            ranked_regions = cls._choose_generic_regions(
                doc_item=doc_item,
                step=step,
                search_target=search_target,
                task_model_primary=task_model_primary,
            )
            if coverage_mode == "exhaustive" and atomic_decision != "relation_judgment" and len(ranked_regions) >= 2:
                step["focus_regions"] = ",".join(ranked_regions[:3])
                if str(step.get("read_strategy", "")).strip() in {"", "locate_then_read_contiguous"}:
                    step["read_strategy"] = "focused_then_expand"
            elif ranked_regions:
                step["focus_regions"] = ",".join(ranked_regions[:2])
            updated_steps.append(step)

        output.doc_execution_graph = updated_steps
        return output

    @staticmethod
    def _looks_like_category_target(*texts: str) -> bool:
        combined = " ".join(str(text or "") for text in texts).casefold()
        if not combined:
            return False
        return bool(
            re.search(
                r"\b(case reason|case type|category|belongs|membership|classif|dispute type|cause of action)\b",
                combined,
            )
            or re.search(r"(案由|类别|类型|是否属于|属于该类|纠纷|资格认定|给付|登记|受理)", combined)
        )

    @staticmethod
    def _extract_target_label(*texts: str) -> str:
        patterns = [
            r"'([^']+)'",
            r'"([^"]+)"',
            r"“([^”]+)”",
            r"‘([^’]+)’",
        ]
        for text in texts:
            raw = str(text or "")
            for pattern in patterns:
                match = re.search(pattern, raw)
                if match:
                    return match.group(1).strip()
        return ""

    @classmethod
    def _postprocess_search_targets(
        cls,
        output: PlannerOutput,
    ) -> PlannerOutput:
        task_model_primary = output.task_model[0] if output.task_model else {}
        atomic_decision = str(task_model_primary.get("atomic_decision", "")).strip().lower()
        answer_topology = str(task_model_primary.get("answer_topology", "")).strip().lower()
        if atomic_decision not in {"extraction", "assignment"} and answer_topology not in {"list", "set", "grouping", "mapping"}:
            return output

        updated_targets: List[Dict[str, str]] = []
        for item in output.search_targets:
            ask = str(item.get("ask", "")).strip()
            success = str(item.get("success_condition", "")).strip()
            ask_lower = ask.casefold()
            success_lower = success.casefold()
            if cls._looks_like_category_target(ask, success):
                target_label = cls._extract_target_label(ask, success) or "the target category"
                quoted_label = target_label if target_label.startswith("'") else f"'{target_label}'"
                item = {
                    **item,
                    "ask": (
                        "Collect the most decisive local formulation about the dispute, benefit, entitlement, "
                        f"qualification, refusal, approval, payment, administrative action, or case type that helps determine whether this document may belong to {quoted_label}."
                    ),
                    "success_condition": (
                        "At least one grounded local formulation is extracted that either directly states "
                        f"{quoted_label} or provides decisive dispute, benefit, entitlement, qualification, refusal, approval, payment, administrative action, or case-type wording for later checker projection."
                    ),
                }
            elif "mentioning" in ask_lower or "explicitly mentioned" in success_lower:
                item = {
                    **item,
                    "ask": (
                        f"{ask.rstrip('.')} or other local evidence that directly states or strongly entails the target category."
                    ),
                    "success_condition": (
                        f"{success.rstrip('.')} or a direct local entailment of the target category."
                    ),
                }
            updated_targets.append(item)
        output.search_targets = updated_targets
        return output

    def plan(self, planner_input: PlannerInput, save_trace: bool = True) -> PlannerOutput:
        user_prompt = self.render_user_prompt(planner_input)
        raw_text = self.llm.generate(self.system_prompt, user_prompt)
        parsed = self.parse_response(raw_text)
        output = PlannerOutput(
            raw_text=raw_text,
            task_goal=parsed["Task Goal"].strip(),
            task_model=self._parse_kv_bullets(parsed["Task Model"]),
            answer_schema=self._parse_kv_bullets(parsed["Answer Schema"]),
            search_targets=self._parse_kv_bullets(parsed["Search Targets"]),
            doc_execution_graph=self._parse_kv_bullets(parsed["Document Execution Graph"]),
            merge_policy=parsed["Merge Policy"].strip(),
            planning_notes=parsed["Planning Notes"].strip(),
        )
        output = self._postprocess_relation_graph(planner_input, output)
        output = self._postprocess_search_targets(output)
        output = self._postprocess_execution_graph(planner_input, output)
        if save_trace:
            self._save_trace(planner_input, user_prompt, output)
        return output

    def _save_trace(self, planner_input: PlannerInput, user_prompt: str, output: PlannerOutput) -> Path:
        trace_path = self.trace_dir / f"{self._safe_filename(planner_input.sample_id)}.json"
        payload = {
            "input": asdict(planner_input),
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt,
            "output": output.to_dict(),
        }
        trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return trace_path

    @staticmethod
    def parse_response(raw_text: str) -> Dict[str, str]:
        sections = Planner._extract_markdown_sections(raw_text)
        missing = [header for header in REQUIRED_HEADERS if header not in sections]
        if missing:
            raise PlannerParseError(f"Planner output is missing required sections: {missing}")
        return sections


def load_loong_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


__all__ = [
    "OpenAIChatLLM",
    "Planner",
    "PlannerInput",
    "PlannerOutput",
    "PlannerParseError",
    "load_loong_jsonl",
]
