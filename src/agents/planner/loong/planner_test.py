from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agents.divider.divider import Divider
from src.agents.planner.loong.planner import OpenAIChatLLM, Planner, load_loong_jsonl


def _select_record(records: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    filtered_mode = any([args.record_type, args.language, args.set_id is not None, args.require_question])
    if not filtered_mode:
        if args.index < 0 or args.index >= len(records):
            raise IndexError(f"index={args.index} is out of range for {len(records)} records")
        return args.index, records[args.index]

    matches: List[Tuple[int, Dict[str, Any]]] = []
    for index, record in enumerate(records):
        if args.record_type and record.get("type") != args.record_type:
            continue
        if args.language and record.get("language") != args.language:
            continue
        if args.set_id is not None and int(record.get("set", 0) or 0) != args.set_id:
            continue
        if args.require_question and not str(record.get("question", "")).strip():
            continue
        matches.append((index, record))

    if not matches:
        raise ValueError("No Loong record matched the requested planner_test filters.")
    if args.match_rank < 0 or args.match_rank >= len(matches):
        raise IndexError(f"match_rank={args.match_rank} is out of range for {len(matches)} matched records")
    return matches[args.match_rank]


def _divider_batch_path(project_root: Path, sample_id: str) -> Path:
    return project_root / "trace_logs" / "divider" / f"{Planner._safe_filename(sample_id)}__batch.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Divider + Planner on one Loong record using the redesigned architecture.")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to Loong jsonl file")
    parser.add_argument("--index", type=int, default=0, help="Direct Loong record index when no filters are used")
    parser.add_argument("--type", dest="record_type", type=str, default=None)
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--set_id", type=int, default=None)
    parser.add_argument("--match_rank", type=int, default=0)
    parser.add_argument("--require_question", action="store_true")
    parser.add_argument("--sample_id", type=str, default="planner_test")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    records = load_loong_jsonl(Path(args.jsonl_path))
    selected_index, record = _select_record(records, args)

    llm = OpenAIChatLLM(model=args.model, base_url=args.base_url, temperature=args.temperature)
    divider = Divider(llm=llm, project_root=project_root)
    planner = Planner(llm=llm, project_root=project_root)

    divider_runs = divider.run_loong_record(record=record, sample_id_prefix=args.sample_id, save_trace=True)
    divider_batch_payload = {
        "sample_id": args.sample_id,
        "resolved_record_index": selected_index,
        "runs": [run.to_dict() for run in divider_runs],
    }
    divider_batch_path = _divider_batch_path(project_root, args.sample_id)
    divider_batch_path.parent.mkdir(parents=True, exist_ok=True)
    divider_batch_path.write_text(json.dumps(divider_batch_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    planner_input = Planner.from_divider_runs(record=record, divider_runs=divider_runs, sample_id=args.sample_id)
    planner_input.metadata["selected_index"] = selected_index
    planner_input.metadata["record_id"] = record.get("id")
    planner_output = planner.plan(planner_input, save_trace=True)
    planner_trace_path = project_root / "trace_logs" / "planner" / f"{Planner._safe_filename(args.sample_id)}.json"

    print("=" * 96)
    print(f"model: {llm.model}")
    print(f"sample_id: {args.sample_id}")
    print(f"selected_index: {selected_index}")
    print("=" * 96)
    print("divider anchors:")
    for run in divider_runs:
        output = run.divider_output
        print(f"- {output.doc_id} | {output.doc_anchor} | packets={output.packet_count} regions={output.region_count}")
    print("=" * 96)
    print("planner search targets:")
    for item in planner_output.search_targets:
        print(f"- {item.get('name', '')}: {item.get('ask', '')}")
    print("=" * 96)
    print("planner execution graph:")
    for item in planner_output.doc_execution_graph:
        print(
            f"- {item.get('id', '')} | doc={item.get('doc', '')} | target={item.get('search_target', '')} | regions={item.get('focus_regions', '')}"
        )
    print("=" * 96)
    print(f"divider batch trace: {divider_batch_path}")
    print(f"planner trace: {planner_trace_path}")


if __name__ == "__main__":
    main()
