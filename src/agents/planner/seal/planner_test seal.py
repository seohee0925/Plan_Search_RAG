from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from planner import OpenAIChatLLM, Planner, PlannerInput, load_loong_jsonl


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _trace_path(project_root: Path, sample_id: str) -> Path:
    return (
        project_root
        / "trace_logs"
        / "planner"
        / f"{Planner._safe_filename(sample_id)}.json"
    )


def build_manual_input(args: argparse.Namespace) -> PlannerInput:
    if not args.instruction or not args.question or not args.doc_titles:
        raise ValueError(
            "Manual mode requires --instruction, --question, and --doc_titles."
        )

    titles = [title.strip() for title in args.doc_titles.split("||") if title.strip()]

    return PlannerInput(
        instruction=args.instruction.strip(),
        question=args.question.strip(),
        doc_title_bundle=Planner.build_doc_title_bundle(titles),
        checker_feedback_or_none=args.checker_feedback,
        sample_id=args.sample_id,
        metadata={"source": "manual"},
    )


def build_loong_input(args: argparse.Namespace) -> PlannerInput:
    jsonl_path = Path(args.jsonl_path)
    records = load_loong_jsonl(jsonl_path)

    if not records:
        raise ValueError(f"No records found in {jsonl_path}")

    if args.index < 0 or args.index >= len(records):
        raise IndexError(f"index={args.index} is out of range for {len(records)} records")

    record = records[args.index]
    planner_input = Planner.from_loong_record(
        record,
        checker_feedback_or_none=args.checker_feedback,
    )

    if args.sample_id:
        planner_input.sample_id = args.sample_id

    return planner_input


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run planner module test with GPT-4o or another OpenAI-compatible model."
    )
    parser.add_argument("--jsonl_path", type=str, default=None, help="Path to Loong jsonl file")
    parser.add_argument("--index", type=int, default=0, help="Index of Loong record")
    parser.add_argument("--instruction", type=str, default=None, help="Manual mode instruction")
    parser.add_argument("--question", type=str, default=None, help="Manual mode question")
    parser.add_argument(
        "--doc_titles",
        type=str,
        default=None,
        help="Manual mode doc titles joined by ||, e.g. 'Title A||Title B||Title C'",
    )
    parser.add_argument("--checker_feedback", type=str, default="None")
    parser.add_argument("--sample_id", type=str, default="planner_test")
    parser.add_argument("--model", type=str, default=None, help="Override model from .env")
    parser.add_argument("--base_url", type=str, default=None, help="Optional OpenAI-compatible base URL")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    project_root = _project_root()
    llm = OpenAIChatLLM(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
    )
    planner = Planner(llm=llm, project_root=project_root)

    if args.jsonl_path:
        planner_input = build_loong_input(args)
    else:
        planner_input = build_manual_input(args)

    output = planner.plan(planner_input, save_trace=True)
    trace_path = _trace_path(project_root, planner_input.sample_id)

    print("=" * 80)
    print(f"model: {llm.model}")
    print(f"sample_id: {planner_input.sample_id}")
    print("=" * 80)
    print("input metadata:")
    print(json.dumps(planner_input.metadata, ensure_ascii=False, indent=2))
    print("=" * 80)
    print("doc titles:")
    print(planner_input.doc_title_bundle or "(empty)")
    print("=" * 80)
    print(json.dumps(asdict(output), ensure_ascii=False, indent=2))
    print("=" * 80)
    print(f"trace saved to: {trace_path}")


if __name__ == "__main__":
    main()
