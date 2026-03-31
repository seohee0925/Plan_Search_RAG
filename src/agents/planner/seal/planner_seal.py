from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from planner import (
        OpenAIChatLLM,
        Planner as BasePlanner,
        PlannerInput,
        PlannerLLM,
        PlannerOutput,
        PlannerParseError,
        REQUIRED_HEADERS,
    )
except ModuleNotFoundError:  # pragma: no cover
    from .planner import (  # type: ignore
        OpenAIChatLLM,
        Planner as BasePlanner,
        PlannerInput,
        PlannerLLM,
        PlannerOutput,
        PlannerParseError,
        REQUIRED_HEADERS,
    )


class Planner(BasePlanner):
    def __init__(
        self,
        llm: PlannerLLM,
        project_root: Optional[Path] = None,
        prompt_dir: str = "prompts_seal",
        trace_dir: str = "trace_logs/planner_seal",
    ) -> None:
        super().__init__(
            llm=llm,
            project_root=project_root,
            prompt_dir=prompt_dir,
            trace_dir=trace_dir,
        )

    @staticmethod
    def _safe_str(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _truncate_tokens(text: str, max_tokens: int = 120) -> str:
        words = text.split()
        if len(words) <= max_tokens:
            return " ".join(words)
        return " ".join(words[:max_tokens]) + " ..."

    @classmethod
    def _derive_title_from_text(cls, text: str) -> str:
        """
        Derive a lightweight title from the first non-empty line of text.
        """
        if not text:
            return ""

        for raw_line in text.splitlines():
            line = cls._normalize_whitespace(raw_line)
            if not line:
                continue
            if len(line.split()) > 30:
                return " ".join(line.split()[:18]) + " ..."
            return line

        return ""

    @classmethod
    def _extract_headers_from_text(cls, text: str, max_headers: int = 2) -> List[str]:
        """
        Extract a few header-like short lines from the beginning of the text.
        """
        if not text:
            return []

        headers: List[str] = []
        for raw_line in text.splitlines():
            line = cls._normalize_whitespace(raw_line)
            if not line:
                continue
            if 2 <= len(line.split()) <= 14:
                headers.append(line)
            if len(headers) >= max_headers:
                break

        seen = set()
        deduped: List[str] = []
        for header in headers:
            if header not in seen:
                seen.add(header)
                deduped.append(header)
        return deduped

    @classmethod
    def _extract_preview_from_text(cls, text: str, max_tokens: int = 120) -> str:
        if not text:
            return ""

        normalized = cls._normalize_whitespace(text)
        return cls._truncate_tokens(normalized, max_tokens=max_tokens)

    @classmethod
    def _doc_to_preview_record(cls, item: Dict[str, Any], doc_idx: int) -> Dict[str, str]:
        text = cls._safe_str(item.get("text", ""))
        explicit_title = cls._safe_str(item.get("title", ""))
        date = cls._safe_str(item.get("date", ""))
        derived_title = explicit_title or cls._derive_title_from_text(text)
        preview = cls._extract_preview_from_text(text, max_tokens=120)
        headers = cls._extract_headers_from_text(text, max_headers=2)

        return {
            "doc_id": f"DOC{doc_idx}",
            "title": derived_title or f"Untitled document {doc_idx}",
            "date": date,
            "preview": preview,
            "headers": " | ".join(headers) if headers else "",
        }

    @classmethod
    def _extract_preview_records_from_seal_docs(cls, raw_docs: Any) -> List[Dict[str, str]]:
        if not isinstance(raw_docs, Sequence) or isinstance(raw_docs, (str, bytes)):
            return []

        records: List[Dict[str, str]] = []
        for idx, item in enumerate(raw_docs, start=1):
            if not isinstance(item, dict):
                continue
            records.append(cls._doc_to_preview_record(item, idx))
        return records

    @classmethod
    def build_doc_preview_bundle(cls, preview_records: Sequence[Dict[str, str]]) -> str:
        lines: List[str] = []
        for rec in preview_records:
            lines.append(f"- {rec['doc_id']}:")
            lines.append(f" title: {rec['title']}")
            if rec.get("date"):
                lines.append(f" date: {rec['date']}")
            if rec.get("headers"):
                lines.append(f" headers: {rec['headers']}")
            if rec.get("preview"):
                lines.append(f" preview: {rec['preview']}")
        return "\n".join(lines)

    @classmethod
    def from_longseal_record(
        cls,
        record: Dict[str, Any],
        doc_field: str = "30_docs",
        checker_feedback_or_none: str = "None",
        sample_id: Optional[str] = None,
    ) -> PlannerInput:
        docs = record.get(doc_field, [])
        preview_records = cls._extract_preview_records_from_seal_docs(docs)
        doc_preview_bundle = cls.build_doc_preview_bundle(preview_records)

        metadata = {
            "source": "longseal",
            "doc_field": doc_field,
            "doc_count": (
                len(docs)
                if isinstance(docs, Sequence) and not isinstance(docs, (str, bytes))
                else 0
            ),
            "preview_count": len(preview_records),
            "question_types": list(record.get("question_types", []) or []),
            "topic": record.get("topic"),
            "freshness": record.get("freshness"),
            "effective_year": record.get("effective_year"),
            "search_results": record.get("search_results"),
            "canary": record.get("canary"),
            "preview_bundle_preview": doc_preview_bundle.splitlines()[:12],
            "doc_ids": [rec["doc_id"] for rec in preview_records],
        }

        resolved_sample_id = sample_id or str(
            record.get("canary") or record.get("question") or "seal_sample"
        )

        # Keep using PlannerInput.doc_title_bundle as a transport field for
        # compatibility with the shared PlannerInput dataclass.
        return PlannerInput(
            instruction="",
            question=str(record.get("question", "")).strip(),
            doc_title_bundle=doc_preview_bundle,
            checker_feedback_or_none=checker_feedback_or_none,
            sample_id=resolved_sample_id,
            metadata=metadata,
        )

    def render_user_prompt(self, planner_input: PlannerInput) -> str:
        """
        The SEAL prompt uses {doc_preview_bundle}, not {doc_title_bundle}.
        We reuse planner_input.doc_title_bundle as the storage field for
        compatibility with the shared PlannerInput dataclass.
        """
        return self.user_prompt_template.format(
            instruction=planner_input.instruction,
            question=planner_input.question,
            doc_preview_bundle=planner_input.doc_title_bundle,
            checker_feedback_or_none=planner_input.checker_feedback_or_none,
        )

    @staticmethod
    def parse_response(raw_text: str) -> Dict[str, str]:
        return Planner._parse_seal_response(raw_text)

    def plan(self, planner_input: PlannerInput, save_trace: bool = True) -> PlannerOutput:
        user_prompt = self.render_user_prompt(planner_input)
        raw_text = self.llm.generate(self.system_prompt, user_prompt)
        parsed = self._parse_seal_response(raw_text)

        output = PlannerOutput(
            raw_text=raw_text,
            task_goal=parsed["Task Goal"],
            answer_form=parsed["Answer Form"],
            completion_criterion=parsed["Completion Criterion"],
            search_strategy=parsed["Search Strategy"],
            evidence_needs=self._parse_kv_bullets(parsed["Evidence Needs"]),
            document_routing=self._parse_kv_bullets(parsed["Document Routing"]),
            search_requests=self._parse_kv_bullets(parsed["Search Requests"]),
        )

        if save_trace:
            self._save_trace(planner_input, user_prompt, output)

        return output

    @staticmethod
    def _parse_seal_response(raw_text: str) -> Dict[str, str]:
        matches = list(re.finditer(r"^###\s+(.+?)\s*$", raw_text, flags=re.MULTILINE))
        if not matches:
            raise PlannerParseError("No markdown section headers were found in planner output.")

        sections: Dict[str, str] = {}
        for idx, match in enumerate(matches):
            header = match.group(1).strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
            sections[header] = raw_text[start:end].strip()

        if "Candidate Prioritization" in sections and "Document Routing" not in sections:
            sections["Document Routing"] = sections["Candidate Prioritization"]

        missing = [header for header in REQUIRED_HEADERS if header not in sections]
        if missing:
            raise PlannerParseError(
                f"Planner output is missing required sections: {missing}"
            )

        return sections


def load_seal_parquet(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"SEAL parquet file not found: {path}")

    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "The 'pyarrow' package is required to read SEAL parquet files."
        ) from exc

    return pq.read_table(path).to_pylist()


__all__ = [
    "OpenAIChatLLM",
    "Planner",
    "PlannerInput",
    "PlannerLLM",
    "PlannerOutput",
    "PlannerParseError",
    "load_seal_parquet",
]
