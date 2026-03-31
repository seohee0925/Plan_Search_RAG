from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agents.divider.divider import DividerDocRun, DividerInput, DividerOutput
from src.agents.golden_retriever.golden_retriever import GoldenRetriever
from src.agents.planner.loong.planner import OpenAIChatLLM, load_loong_jsonl


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _default_jsonl_path(project_root: Path) -> Path:
    processed = project_root / "Loong" / "data" / "loong_process.jsonl"
    raw = project_root / "Loong" / "data" / "loong.jsonl"
    if processed.exists():
        return processed
    if raw.exists():
        return raw
    raise FileNotFoundError(f"Loong dataset not found: {processed} or {raw}")


def _default_divider_batch_path(project_root: Path, planner_sample_id: str) -> Path:
    return project_root / "trace_logs" / "divider" / f"{GoldenRetriever._safe_filename(planner_sample_id)}__batch.json"


def _batch_trace_path(project_root: Path, sample_id_prefix: str) -> Path:
    return project_root / "trace_logs" / "golden_retriever" / f"{GoldenRetriever._safe_filename(sample_id_prefix)}__batch.json"


def _resolve_loong_record(records: List[Dict[str, Any]], planner_trace: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    metadata = planner_trace.get("input", {}).get("metadata", {})
    selected_index = metadata.get("selected_index")
    record_id = metadata.get("record_id")
    if isinstance(selected_index, int) and 0 <= selected_index < len(records):
        record = records[selected_index]
        if not record_id or record.get("id") == record_id:
            return selected_index, record
    if record_id:
        for index, record in enumerate(records):
            if record.get("id") == record_id:
                return index, record
    trace_question = str(planner_trace.get("input", {}).get("question", "")).strip()
    trace_instruction = str(planner_trace.get("input", {}).get("instruction", "")).strip()
    for index, record in enumerate(records):
        if (
            str(record.get("question", "")).strip() == trace_question
            and str(record.get("instruction", "")).strip() == trace_instruction
        ):
            return index, record
    raise ValueError("Could not resolve the Loong record referenced by the planner trace.")


def _load_divider_runs(divider_batch_trace: Dict[str, Any]) -> List[DividerDocRun]:
    runs: List[DividerDocRun] = []
    for run in divider_batch_trace.get("runs", []):
        runs.append(
            DividerDocRun(
                doc_id=str(run.get("doc_id", "")).strip(),
                trace_path=str(run.get("trace_path", "")).strip(),
                divider_input=DividerInput(**run.get("input", {})),
                divider_output=DividerOutput(**run.get("output", {})),
            )
        )
    return runs


def save_batch_trace(
    project_root: Path,
    sample_id_prefix: str,
    planner_trace_path: Path,
    divider_batch_path: Path,
    resolved_record_index: int,
    runs: Sequence[Any],
) -> Path:
    path = _batch_trace_path(project_root, sample_id_prefix)
    payload = {
        "planner_trace_path": str(planner_trace_path),
        "divider_batch_path": str(divider_batch_path),
        "resolved_record_index": resolved_record_index,
        "runs": [run.to_dict() for run in runs],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the redesigned scoped retriever from a saved planner trace.")
    parser.add_argument("--planner_trace_path", type=str, required=True)
    parser.add_argument("--divider_batch_path", type=str, default=None)
    parser.add_argument("--jsonl_path", type=str, default=None)
    parser.add_argument("--sample_id", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    planner_trace_path = Path(args.planner_trace_path)
    planner_trace = _load_json(planner_trace_path)
    planner_sample_id = str(planner_trace.get("input", {}).get("sample_id", "planner_trace")).strip()
    divider_batch_path = Path(args.divider_batch_path) if args.divider_batch_path else _default_divider_batch_path(project_root, planner_sample_id)
    divider_batch_trace = _load_json(divider_batch_path)
    divider_runs = _load_divider_runs(divider_batch_trace)

    jsonl_path = Path(args.jsonl_path) if args.jsonl_path else _default_jsonl_path(project_root)
    records = load_loong_jsonl(jsonl_path)
    resolved_record_index, record = _resolve_loong_record(records, planner_trace)

    sample_id_prefix = args.sample_id or planner_sample_id
    llm = OpenAIChatLLM(model=args.model, base_url=args.base_url, temperature=args.temperature)
    retriever = GoldenRetriever(llm=llm, project_root=project_root)
    runs = retriever.run_execution_graph(
        record=record,
        planner_output=planner_trace.get("output", {}),
        divider_runs=divider_runs,
        sample_id_prefix=sample_id_prefix,
        save_trace=True,
    )
    batch_trace_path = save_batch_trace(
        project_root=project_root,
        sample_id_prefix=sample_id_prefix,
        planner_trace_path=planner_trace_path,
        divider_batch_path=divider_batch_path,
        resolved_record_index=resolved_record_index,
        runs=runs,
    )

    print("=" * 96)
    print(f"model: {llm.model}")
    print(f"sample_id_prefix: {sample_id_prefix}")
    print(f"resolved_record_index: {resolved_record_index}")
    print("=" * 96)
    for run in runs:
        output = run.retriever_output
        print(f"{run.step_id} | doc={run.doc_id} | status={output.search_status}")
        if output.evidence_units:
            for unit in output.evidence_units[:3]:
                print(
                    f"  - {unit.get('status', '')} | packets={unit.get('packet_refs', '')} | {unit.get('extracted_text', '')[:180]}"
                )
        else:
            print("  - no evidence units")
        print("-" * 96)
    print(f"batch trace saved to: {batch_trace_path}")


if __name__ == "__main__":
    main()
