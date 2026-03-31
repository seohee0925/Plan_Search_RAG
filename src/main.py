from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from orchestrator import (
    build_domain_level_manifest,
    default_loong_jsonl,
    default_project_root,
    evaluate_prediction,
    load_loong_records,
    project_answer_from_checker,
    save_manifest,
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_gold_answer(records: list[dict[str, Any]], selected_index: int | None, record_id: str | None) -> Any:
    if isinstance(selected_index, int) and 0 <= selected_index < len(records):
        return records[selected_index].get("answer")
    if record_id:
        for record in records:
            if str(record.get("id", "")) == str(record_id):
                return record.get("answer")
    raise ValueError("Could not resolve gold answer from provided metadata.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Utility entrypoint for manifest building and architecture-aware evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("build-manifest", help="Build a generic domain/level manifest from Loong.")
    manifest_parser.add_argument("--jsonl_path", type=str, default=None)
    manifest_parser.add_argument("--max_per_combo", type=int, default=1)
    manifest_parser.add_argument("--output_path", type=str, default=None)

    pipeline_parser = subparsers.add_parser("evaluate-pipeline", help="Inspect a saved pipeline trace and reprint its evaluation payload.")
    pipeline_parser.add_argument("--pipeline_trace_path", type=str, required=True)

    checker_parser = subparsers.add_parser("evaluate-checker", help="Project an answer from planner+checker traces and compare it against Loong gold.")
    checker_parser.add_argument("--planner_trace_path", type=str, required=True)
    checker_parser.add_argument("--checker_trace_path", type=str, required=True)
    checker_parser.add_argument("--jsonl_path", type=str, default=None)
    checker_parser.add_argument("--selected_index", type=int, default=None)
    checker_parser.add_argument("--record_id", type=str, default=None)

    args = parser.parse_args()
    project_root = default_project_root()

    if args.command == "build-manifest":
        jsonl_path = Path(args.jsonl_path) if args.jsonl_path else default_loong_jsonl(project_root)
        output_path = (
            Path(args.output_path)
            if args.output_path
            else project_root / "artifacts" / "eval" / "loong_domain_level_manifest.json"
        )
        records = load_loong_records(jsonl_path)
        manifest = build_domain_level_manifest(records, max_per_combo=args.max_per_combo)
        saved = save_manifest(manifest, output_path)
        print(json.dumps({"output_path": str(saved), "count": len(manifest)}, ensure_ascii=False, indent=2))
        return

    if args.command == "evaluate-pipeline":
        payload = _load_json(Path(args.pipeline_trace_path))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "evaluate-checker":
        planner_trace = _load_json(Path(args.planner_trace_path))
        checker_trace = _load_json(Path(args.checker_trace_path))
        projected_answer = project_answer_from_checker(
            checker_output=checker_trace.get("output", {}),
            planner_output=planner_trace.get("output", {}),
        )
        jsonl_path = Path(args.jsonl_path) if args.jsonl_path else default_loong_jsonl(project_root)
        records = load_loong_records(jsonl_path)
        selected_index = args.selected_index
        if selected_index is None:
            selected_index = planner_trace.get("input", {}).get("metadata", {}).get("selected_index")
        record_id = args.record_id or planner_trace.get("input", {}).get("metadata", {}).get("record_id")
        gold_answer = _resolve_gold_answer(records, selected_index, record_id)
        evaluation = evaluate_prediction(projected_answer, gold_answer)
        print(
            json.dumps(
                {
                    "projected_answer": projected_answer,
                    "gold_answer": gold_answer,
                    "evaluation": evaluation.to_dict(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return


if __name__ == "__main__":
    main()
