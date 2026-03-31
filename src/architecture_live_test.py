from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

from orchestrator import default_loong_jsonl, default_project_root, load_loong_records, project_answer_from_checker

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agents.checker.checker import Checker
from src.agents.divider.divider import Divider
from src.agents.generator.generator import Generator
from src.agents.golden_retriever.golden_retriever import GoldenRetriever
from src.agents.planner.loong.planner import OpenAIChatLLM, Planner


def _pick_record(records: List[Dict[str, Any]], record_type: str, level: int, set_id: int) -> tuple[int, Dict[str, Any]]:
    for idx, record in enumerate(records):
        if record.get("type") == record_type and int(record.get("level", 0) or 0) == level and int(record.get("set", 0) or 0) == set_id:
            return idx, record
    raise ValueError(f"No record found for type={record_type}, level={level}, set_id={set_id}")


def _print_block(title: str, body: str) -> None:
    print("=" * 96)
    print(title)
    print("=" * 96)
    print(body.strip() if body.strip() else "(empty)")


def _live_trace_path(project_root: Path, sample_id: str) -> Path:
    path = project_root / "trace_logs" / "live"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{GoldenRetriever._safe_filename(sample_id)}.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Live pretty runner for the redesigned architecture.")
    parser.add_argument("--jsonl_path", type=str, default=None)
    parser.add_argument("--record_type", type=str, required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--set_id", type=int, default=1)
    parser.add_argument("--sample_prefix", type=str, default="live_arch")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    project_root = default_project_root()
    jsonl_path = Path(args.jsonl_path) if args.jsonl_path else default_loong_jsonl(project_root)
    records = load_loong_records(jsonl_path)
    selected_index, record = _pick_record(records, args.record_type, args.level, args.set_id)
    sample_id = f"{args.sample_prefix}_{args.record_type}_level{args.level}_{selected_index}"

    llm = OpenAIChatLLM(model=args.model, base_url=args.base_url, temperature=args.temperature)
    divider = Divider(llm=llm, project_root=project_root)
    planner = Planner(llm=llm, project_root=project_root)
    retriever = GoldenRetriever(llm=llm, project_root=project_root)
    checker = Checker(llm=llm, project_root=project_root)
    generator = Generator(llm=llm, project_root=project_root)

    divider_runs = divider.run_loong_record(record=record, sample_id_prefix=sample_id, save_trace=True)
    divider_lines = []
    for run in divider_runs:
        out = run.divider_output
        divider_lines.append(f"{out.doc_id} | {out.doc_anchor}")
        divider_lines.append(f"  title={out.display_title}")
        divider_lines.append(f"  packets={out.packet_count} regions={out.region_count}")
        for role in out.region_roles[:5]:
            divider_lines.append(
                f"  {role.get('region_id','')} -> {role.get('role','')} | {role.get('why','')}"
            )
    _print_block("Divider", "\n".join(divider_lines))

    planner_input = Planner.from_divider_runs(record=record, divider_runs=divider_runs, sample_id=sample_id)
    planner_output = planner.plan(planner_input, save_trace=True)
    planner_lines = [
        f"task_goal={planner_output.task_goal}",
        "",
        "search_targets:",
    ]
    for item in planner_output.search_targets:
        planner_lines.append(f"  {item.get('name','')} -> {item.get('ask','')}")
    planner_lines.append("")
    planner_lines.append("execution_graph:")
    for item in planner_output.doc_execution_graph:
        planner_lines.append(
            f"  {item.get('id','')} | doc={item.get('doc','')} | target={item.get('search_target','')} | regions={item.get('focus_regions','')}"
        )
    _print_block("Planner", "\n".join(planner_lines))

    retriever_runs = retriever.run_execution_graph(
        record=record,
        planner_output=planner_output.to_dict(),
        divider_runs=divider_runs,
        sample_id_prefix=sample_id,
        save_trace=True,
    )
    retriever_lines = []
    for run in retriever_runs:
        out = run.retriever_output
        retriever_lines.append(f"{run.step_id} | doc={run.doc_id} | status={out.search_status}")
        for unit in out.evidence_units[:3]:
            retriever_lines.append(
                f"  {unit.get('status','')} | {unit.get('packet_refs','')} | {unit.get('extracted_text','')[:180]}"
            )
    _print_block("Retriever", "\n".join(retriever_lines))

    planner_trace = json.loads((project_root / "trace_logs" / "planner" / f"{Planner._safe_filename(sample_id)}.json").read_text(encoding="utf-8"))
    divider_batch = {"runs": [run.to_dict() for run in divider_runs]}
    retriever_batch = {"runs": [run.to_dict() for run in retriever_runs]}
    checker_input = checker.build_input(
        planner_trace=planner_trace,
        divider_batch_trace=divider_batch,
        retriever_batch_trace=retriever_batch,
        sample_id=f"{sample_id}_checker",
    )
    checker_output = checker.check(checker_input, save_trace=True)
    projected_answer = project_answer_from_checker(checker_output.to_dict(), planner_output.to_dict())
    checker_lines = [
        f"verdict={checker_output.sufficiency_verdict}",
        f"evidence_state={checker_output.evidence_state_summary}",
        "",
        "projected_answer_state:",
    ]
    for item in checker_output.projected_answer_state:
        checker_lines.append(f"  {item.get('answer_key','')} -> {item.get('value','')} | {item.get('basis','')}")
    if checker_output.remaining_gaps:
        checker_lines.append("")
        checker_lines.append("remaining_gaps:")
        for item in checker_output.remaining_gaps:
            checker_lines.append(f"  {item.get('issue','')} | {item.get('why','')}")
    _print_block("Checker", "\n".join(checker_lines))

    report_blocks = [
        ("Divider", "\n".join(divider_lines)),
        ("Planner", "\n".join(planner_lines)),
        ("Retriever", "\n".join(retriever_lines)),
        ("Checker", "\n".join(checker_lines)),
    ]

    if checker_output.sufficiency_verdict:
        generator_input = generator.build_input(planner_trace=planner_trace, checker_output=checker_output.to_dict(), sample_id=f"{sample_id}_generator")
        generator_output = generator.generate_answer(generator_input, save_trace=True)
        generator_body = f"{generator_output.render_summary}\n\n{generator_output.final_answer}"
        _print_block("Generator", generator_body)
        report_blocks.append(("Generator", generator_body))
        final_payload = {"projected_answer": projected_answer, "generated_answer": generator_output.parsed_final_answer}
        print(json.dumps(final_payload, ensure_ascii=False, indent=2))
    else:
        final_payload = {"projected_answer": projected_answer, "generated_answer": None}
        print(json.dumps(final_payload, ensure_ascii=False, indent=2))

    report_lines = [f"# Live Architecture Trace\n", f"- sample_id: {sample_id}", f"- selected_index: {selected_index}"]
    for title, body in report_blocks:
        report_lines.append(f"\n## {title}\n")
        report_lines.append("```text")
        report_lines.append(body.strip() if body.strip() else "(empty)")
        report_lines.append("```")
    report_lines.append("\n## Final Output\n")
    report_lines.append("```json")
    report_lines.append(json.dumps(final_payload, ensure_ascii=False, indent=2))
    report_lines.append("```")
    _live_trace_path(project_root, sample_id).write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
