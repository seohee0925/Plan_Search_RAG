from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dawon.agents.planner import OpenAIChatLLM, PlannerLLM
except ModuleNotFoundError:  # pragma: no cover
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from dawon.agents.planner import OpenAIChatLLM, PlannerLLM  # type: ignore


REQUIRED_HEADERS = [
    "Render Summary",
    "Final Answer",
]


@dataclass
class GeneratorInput:
    instruction: str
    question: str
    task_goal: str
    task_model: str
    answer_schema: str
    projected_answer_state: str
    evidence_state_summary: str
    sample_id: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    reference_answer: Any = None


@dataclass
class GeneratorOutput:
    raw_text: str
    render_summary: str
    final_answer: str
    parsed_final_answer: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GeneratorParseError(ValueError):
    pass


class Generator:
    def __init__(
        self,
        llm: PlannerLLM,
        project_root: Optional[Path] = None,
        prompt_dir: str = "prompts",
        trace_dir: str = "trace_logs/generator",
    ) -> None:
        self.llm = llm
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        self.prompt_dir = self.project_root / prompt_dir
        self.trace_dir = self.project_root / trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.system_prompt = (self.prompt_dir / "generator_sys.txt").read_text(encoding="utf-8").strip()
        self.user_prompt_template = (self.prompt_dir / "generator_user.txt").read_text(encoding="utf-8").strip()

    @staticmethod
    def _safe_filename(name: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._")
        return safe or "unknown"

    @staticmethod
    def _format_bullets(items: List[Dict[str, Any]]) -> str:
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
    def _coerce_final_answer(text: str) -> Any:
        cleaned = text.strip()
        if not cleaned:
            return None
        fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL)
        if fenced:
            cleaned = fenced.group(1).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return cleaned

    @staticmethod
    def _extract_sections(raw_text: str) -> Dict[str, str]:
        matches = list(re.finditer(r"^###\s+(.+?)\s*$", raw_text, flags=re.MULTILINE))
        if not matches:
            raise GeneratorParseError("No markdown section headers were found in generator output.")
        sections: Dict[str, str] = {}
        for idx, match in enumerate(matches):
            header = match.group(1).strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
            sections[header] = raw_text[start:end].strip()
        missing = [header for header in REQUIRED_HEADERS if header not in sections]
        if missing:
            raise GeneratorParseError(f"Generator output is missing required sections: {missing}")
        return sections

    @classmethod
    def build_input(
        cls,
        planner_trace: Dict[str, Any],
        checker_output: Dict[str, Any],
        sample_id: Optional[str] = None,
        gold_answer: Any = None,
    ) -> GeneratorInput:
        planner_input = planner_trace.get("input", {})
        planner_output = planner_trace.get("output", {})
        return GeneratorInput(
            instruction=str(planner_input.get("instruction", "")).strip(),
            question=str(planner_input.get("question", "")).strip(),
            task_goal=str(planner_output.get("task_goal", "")).strip(),
            task_model=cls._format_bullets(planner_output.get("task_model", [])),
            answer_schema=cls._format_bullets(planner_output.get("answer_schema", [])),
            projected_answer_state=cls._format_bullets(checker_output.get("projected_answer_state", [])),
            evidence_state_summary=str(checker_output.get("evidence_state_summary", "")).strip(),
            sample_id=sample_id or f"{planner_input.get('sample_id', 'unknown')}_generator",
            metadata={
                "planner_sample_id": planner_input.get("sample_id"),
                "checker_sample_id": checker_output.get("sample_id"),
            },
            reference_answer=gold_answer,
        )

    def render_user_prompt(self, generator_input: GeneratorInput) -> str:
        return self.user_prompt_template.format(
            instruction=generator_input.instruction,
            question=generator_input.question,
            task_goal=generator_input.task_goal,
            task_model=generator_input.task_model,
            answer_schema=generator_input.answer_schema,
            projected_answer_state=generator_input.projected_answer_state,
            evidence_state_summary=generator_input.evidence_state_summary,
        )

    def generate_answer(self, generator_input: GeneratorInput, save_trace: bool = True) -> GeneratorOutput:
        user_prompt = self.render_user_prompt(generator_input)
        raw_text = self.llm.generate(self.system_prompt, user_prompt)
        sections = self._extract_sections(raw_text)
        output = GeneratorOutput(
            raw_text=raw_text,
            render_summary=sections["Render Summary"].strip(),
            final_answer=sections["Final Answer"].strip(),
            parsed_final_answer=self._coerce_final_answer(sections["Final Answer"].strip()),
        )
        if save_trace:
            self._save_trace(generator_input, user_prompt, output)
        return output

    def _save_trace(self, generator_input: GeneratorInput, user_prompt: str, output: GeneratorOutput) -> Path:
        path = self.trace_dir / f"{self._safe_filename(generator_input.sample_id)}.json"
        payload = {
            "input": asdict(generator_input),
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt,
            "output": output.to_dict(),
            "reference_answer": generator_input.reference_answer,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


__all__ = [
    "Generator",
    "GeneratorInput",
    "GeneratorOutput",
    "GeneratorParseError",
    "OpenAIChatLLM",
]
