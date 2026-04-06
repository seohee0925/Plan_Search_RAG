from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


class KoreanTranslationReportWriter:
    def __init__(
        self,
        llm: Any,
        project_root: Path,
        chunk_char_limit: int = 1400,
        report_dir: str = "trace_logs/translation_reports",
    ) -> None:
        self.llm = llm
        self.project_root = Path(project_root)
        self.chunk_char_limit = max(400, int(chunk_char_limit))
        self.report_dir = self.project_root / report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, str] = {}

    @staticmethod
    def _safe_filename(name: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._")
        return safe or "unknown"

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _has_chinese(text: str) -> bool:
        return bool(_CHINESE_RE.search(text))

    def _split_long_piece(self, text: str) -> List[str]:
        if len(text) <= self.chunk_char_limit:
            return [text]

        pieces = re.split(r"(?<=[。！？；;.!?])", text)
        pieces = [piece for piece in pieces if piece]
        if len(pieces) <= 1:
            return [text[index : index + self.chunk_char_limit] for index in range(0, len(text), self.chunk_char_limit)]

        chunks: List[str] = []
        current = ""
        for piece in pieces:
            if len(piece) > self.chunk_char_limit:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._split_long_piece(piece))
                continue
            if current and len(current) + len(piece) > self.chunk_char_limit:
                chunks.append(current)
                current = piece
                continue
            current += piece
        if current:
            chunks.append(current)
        return chunks or [text]

    def _chunk_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_char_limit:
            return [text]

        chunks: List[str] = []
        current = ""
        pieces = re.split(r"(\n+)", text)
        for piece in pieces:
            if not piece:
                continue
            subpieces = self._split_long_piece(piece)
            for subpiece in subpieces:
                if current and len(current) + len(subpiece) > self.chunk_char_limit:
                    chunks.append(current)
                    current = subpiece
                    continue
                current += subpiece
        if current:
            chunks.append(current)
        return chunks or [text]

    def _translate_chunk(self, text: str) -> str:
        system_prompt = (
            "You are a professional translator. Translate Chinese into natural Korean. "
            "Keep line breaks, markdown, JSON punctuation, bullet markers, table separators, IDs, numbers, dates, units, and code-like tokens unchanged whenever possible. "
            "Do not summarize. Do not explain. Return only the translated text."
        )
        user_prompt = (
            "다음 텍스트를 자연스러운 한국어로 번역하세요.\n"
            "규칙:\n"
            "- 줄바꿈과 문단 구조를 유지하세요.\n"
            "- DOC1, STEP_1, R7 같은 식별자는 유지하세요.\n"
            "- 숫자, 날짜, 단위, 표 구분자(|), JSON/마크다운 기호를 함부로 바꾸지 마세요.\n"
            "- 코드나 경로처럼 보이는 토큰은 가능한 한 유지하세요.\n"
            "- 요약하지 말고, 설명하지 말고, 번역문만 출력하세요.\n\n"
            f"<<<TEXT\n{text}\nTEXT>>>"
        )
        translated = self.llm.generate(system_prompt, user_prompt).strip()
        translated = re.sub(r"^```(?:text)?\s*", "", translated)
        translated = re.sub(r"\s*```$", "", translated)
        translated = translated.strip()
        return translated or text

    def translate_text(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        if not self._has_chinese(text):
            self._cache[text] = text
            return text

        translated_parts: List[str] = []
        for chunk in self._chunk_text(text):
            try:
                translated_parts.append(self._translate_chunk(chunk))
            except Exception:
                translated_parts.append(chunk)
        translated = "".join(translated_parts)
        self._cache[text] = translated
        return translated

    def translate_recursive(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.translate_text(value)
        if isinstance(value, list):
            return [self.translate_recursive(item) for item in value]
        if isinstance(value, dict):
            return {key: self.translate_recursive(item) for key, item in value.items()}
        return value

    @staticmethod
    def _format_json_block(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2)

    @staticmethod
    def _section(title: str, body: str) -> str:
        return f"### {title}\n{body.strip()}"

    def _extract_documents(self, divider_batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []
        for run in divider_batch.get("runs", []) or []:
            input_obj = run.get("input", {})
            output_obj = run.get("output", {})
            documents.append(
                {
                    "doc_id": run.get("doc_id"),
                    "title": input_obj.get("display_title"),
                    "document_text": input_obj.get("document_text"),
                    "packet_count": output_obj.get("packet_count"),
                    "region_count": output_obj.get("region_count"),
                }
            )
        return documents

    def _extract_divider_section(self, divider_batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for run in divider_batch.get("runs", []) or []:
            output_obj = run.get("output", {})
            items.append(
                {
                    "doc_id": run.get("doc_id"),
                    "display_title": output_obj.get("display_title"),
                    "doc_anchor": output_obj.get("doc_anchor"),
                    "divider_notes": output_obj.get("divider_notes"),
                    "doc_map_summary": output_obj.get("doc_map_summary"),
                    "region_roles": output_obj.get("region_roles"),
                    "packet_store": output_obj.get("packet_store"),
                    "region_store": output_obj.get("region_store"),
                    "search_views": output_obj.get("search_views"),
                    "raw_text": output_obj.get("raw_text"),
                }
            )
        return items

    def _extract_planner_section(self, planner_trace: Dict[str, Any]) -> Dict[str, Any]:
        input_obj = planner_trace.get("input", {})
        output_obj = planner_trace.get("output", {})
        return {
            "input": {
                "instruction": input_obj.get("instruction"),
                "question": input_obj.get("question"),
                "checker_feedback_or_none": input_obj.get("checker_feedback_or_none"),
                "document_catalog": input_obj.get("document_catalog"),
            },
            "output": output_obj,
        }

    def _extract_retriever_section(self, retriever_batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for run in retriever_batch.get("runs", []) or []:
            input_obj = run.get("input", {})
            output_obj = run.get("output", {})
            items.append(
                {
                    "step_id": run.get("step_id"),
                    "doc_id": run.get("doc_id"),
                    "input": {
                        "search_target_name": input_obj.get("search_target_name"),
                        "search_target_ask": input_obj.get("search_target_ask"),
                        "search_target_success_condition": input_obj.get("search_target_success_condition"),
                        "display_title": input_obj.get("display_title"),
                        "doc_anchor": input_obj.get("doc_anchor"),
                        "focus_regions": input_obj.get("focus_regions"),
                        "scoped_region_map": input_obj.get("scoped_region_map"),
                        "scoped_packet_view": input_obj.get("scoped_packet_view"),
                        "metadata": input_obj.get("metadata"),
                    },
                    "output": output_obj,
                }
            )
        return items

    def _extract_checker_section(self, checker_trace: Dict[str, Any]) -> Dict[str, Any]:
        input_obj = checker_trace.get("input", {})
        output_obj = checker_trace.get("output", {})
        return {
            "input": {
                "question": input_obj.get("question"),
                "task_goal": input_obj.get("task_goal"),
                "answer_schema": input_obj.get("answer_schema"),
                "execution_graph": input_obj.get("execution_graph"),
                "integrated_evidence_state": input_obj.get("integrated_evidence_state"),
            },
            "output": output_obj,
        }

    def _extract_generator_section(self, generator_trace: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "input": generator_trace.get("input", {}),
            "output": generator_trace.get("output", {}),
            "reference_answer": generator_trace.get("reference_answer"),
        }

    def write_sample_report(self, pipeline_trace_path: Path) -> Path:
        pipeline_trace = self._load_json(Path(pipeline_trace_path))
        divider_batch = self._load_json(Path(pipeline_trace["module_traces"]["divider_batch"]))

        sample_summary = self.translate_recursive(
            {
                "sample_id": pipeline_trace.get("sample_id"),
                "record_meta": pipeline_trace.get("record_meta"),
                "task_input": pipeline_trace.get("task_input"),
                "gold_answer": pipeline_trace.get("gold_answer"),
                "projected_answer": pipeline_trace.get("projected_answer"),
                "generated_answer": pipeline_trace.get("generated_answer"),
                "projection_eval": pipeline_trace.get("projection_eval"),
                "generator_eval": pipeline_trace.get("generator_eval"),
            }
        )
        documents = self.translate_recursive(self._extract_documents(divider_batch))
        divider_section = self.translate_recursive(self._extract_divider_section(divider_batch))

        sections: List[str] = [
            self._section("샘플 요약", self._format_json_block(sample_summary)),
            self._section("문서 번역", self._format_json_block(documents)),
            self._section("Divider 결과 번역", self._format_json_block(divider_section)),
        ]

        for cycle in pipeline_trace.get("cycles", []) or []:
            cycle_index = cycle.get("cycle_index", 0)
            planner_trace = self._load_json(Path(cycle["planner_trace"]))
            retriever_batch = self._load_json(Path(cycle["retriever_batch_trace"]))
            checker_trace = self._load_json(Path(cycle["checker_trace"]))

            cycle_summary = self.translate_recursive(cycle)
            planner_section = self.translate_recursive(self._extract_planner_section(planner_trace))
            retriever_section = self.translate_recursive(self._extract_retriever_section(retriever_batch))
            checker_section = self.translate_recursive(self._extract_checker_section(checker_trace))

            sections.append(self._section(f"Cycle {cycle_index} 요약", self._format_json_block(cycle_summary)))
            sections.append(self._section(f"Cycle {cycle_index} Planner 번역", self._format_json_block(planner_section)))
            sections.append(self._section(f"Cycle {cycle_index} Retriever 번역", self._format_json_block(retriever_section)))
            sections.append(self._section(f"Cycle {cycle_index} Checker 번역", self._format_json_block(checker_section)))

        generator_trace_path = pipeline_trace.get("module_traces", {}).get("generator")
        if generator_trace_path:
            generator_trace = self._load_json(Path(generator_trace_path))
            generator_section = self.translate_recursive(self._extract_generator_section(generator_trace))
            sections.append(self._section("Generator 번역", self._format_json_block(generator_section)))

        report_text = "\n\n".join(sections).strip() + "\n"
        output_path = self.report_dir / f"{self._safe_filename(str(pipeline_trace.get('sample_id', 'unknown')))}_ko.txt"
        output_path.write_text(report_text, encoding="utf-8")
        return output_path

    def write_batch_summary_report(self, batch_payload: Dict[str, Any]) -> Path:
        translated = self.translate_recursive(batch_payload)
        output_path = self.report_dir / f"{self._safe_filename(str(batch_payload.get('sample_prefix', 'batch')))}__batch_ko.txt"
        output_path.write_text(self._format_json_block(translated) + "\n", encoding="utf-8")
        return output_path
