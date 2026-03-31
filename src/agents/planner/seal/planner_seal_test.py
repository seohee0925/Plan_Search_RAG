from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

try:
    from planner import Planner as BasePlanner
    from planner_seal import OpenAIChatLLM, Planner, load_seal_parquet
except ModuleNotFoundError:  # pragma: no cover
    from .planner import Planner as BasePlanner  # type: ignore
    from .planner_seal import OpenAIChatLLM, Planner, load_seal_parquet  # type: ignore


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _trace_path(project_root: Path, sample_id: str) -> Path:
    return (
        project_root
        / "trace_logs"
        / "planner_seal"
        / f"{BasePlanner._safe_filename(sample_id)}.json"
    )


def build_longseal_input(args: argparse.Namespace):
    parquet_path = Path(args.parquet_path)
    records = load_seal_parquet(parquet_path)

    if not records:
        raise ValueError(f"No records found in {parquet_path}")

    if args.index < 0 or args.index >= len(records):
        raise IndexError(f"index={args.index} is out of range for {len(records)} records")

    record = records[args.index]
    return Planner.from_longseal_record(
        record,
        doc_field=args.doc_field,
        checker_feedback_or_none=args.checker_feedback,
        sample_id=args.sample_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run planner on SEAL LongSEAL parquet rows."
    )
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to LongSEAL parquet file")
    parser.add_argument("--index", type=int, default=0, help="Row index in LongSEAL parquet")
    parser.add_argument(
        "--doc_field",
        type=str,
        default="30_docs",
        choices=["12_docs", "20_docs", "30_docs"],
        help="Candidate document field to bundle into planner input",
    )
    parser.add_argument("--checker_feedback", type=str, default="None")
    parser.add_argument("--sample_id", type=str, default="planner_seal_test")
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
    planner_input = build_longseal_input(args)
    output = planner.plan(planner_input, save_trace=True)
    trace_path = _trace_path(project_root, planner_input.sample_id)

    print("=" * 80)
    print(f"model: {llm.model}")
    print(f"sample_id: {planner_input.sample_id}")
    print("=" * 80)
    print("input metadata:")
    print(json.dumps(planner_input.metadata, ensure_ascii=False, indent=2))
    print("=" * 80)
    print("doc previews:")
    print(planner_input.doc_title_bundle or "(empty)")
    print("=" * 80)
    print(json.dumps(asdict(output), ensure_ascii=False, indent=2))
    print("=" * 80)
    print(f"trace saved to: {trace_path}")


if __name__ == "__main__":
    main()
