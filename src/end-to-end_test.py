from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List

from orchestrator import (
    ManifestItem,
    build_domain_level_manifest,
    default_loong_jsonl,
    default_project_root,
    evaluate_prediction,
    load_loong_records,
    project_answer_from_checker,
)

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agents.checker.checker import Checker
from src.agents.divider.divider import Divider
from src.agents.generator.generator import Generator
from src.agents.golden_retriever.golden_retriever import GoldenRetriever
from src.agents.planner.loong.planner import OpenAIChatLLM, Planner


def _trace_dir(project_root: Path, name: str) -> Path:
    path = project_root / "trace_logs" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_trace(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _select_manifest_items(records: List[Dict[str, Any]], args: argparse.Namespace) -> List[ManifestItem]:
    if args.manifest_path:
        payload = json.loads(Path(args.manifest_path).read_text(encoding="utf-8"))
        manifest = [
            ManifestItem(
                selected_index=int(item["selected_index"]),
                record_id=str(item.get("record_id", "unknown")),
                set_id=int(item.get("set_id", 0) or 0),
                record_type=str(item.get("record_type", "unknown")),
                level=int(item.get("level", 0) or 0),
                language=str(item.get("language", "unknown")),
                question=str(item.get("question", "")).strip(),
                sample_id=str(
                    item.get(
                        "sample_id",
                        f"{item.get('record_type', 'unknown')}_level{item.get('level', 0)}_{item['selected_index']}",
                    )
                ).strip(),
            )
            for item in payload
        ]
        if args.max_items is not None:
            manifest = manifest[: args.max_items]
        return manifest
    if args.selected_index is not None:
        record = records[args.selected_index]
        return [
            ManifestItem(
                selected_index=args.selected_index,
                record_id=str(record.get("id", "unknown")),
                set_id=int(record.get("set", 0) or 0),
                record_type=str(record.get("type", "unknown")),
                level=int(record.get("level", 0) or 0),
                language=str(record.get("language", "unknown")),
                question=str(record.get("question", "")).strip(),
                sample_id=f"{record.get('type', 'unknown')}_level{record.get('level', 0)}_{args.selected_index}",
            )
        ]
    manifest = build_domain_level_manifest(records, max_per_combo=args.max_per_combo)
    if args.set_id is not None:
        manifest = [item for item in manifest if item.set_id == args.set_id]
    if args.record_type:
        manifest = [item for item in manifest if item.record_type == args.record_type]
    if args.level is not None:
        manifest = [item for item in manifest if item.level == args.level]
    if args.max_items is not None:
        manifest = manifest[: args.max_items]
    return manifest


def _divider_batch_trace_path(project_root: Path, sample_id: str) -> Path:
    return _trace_dir(project_root, "divider") / f"{GoldenRetriever._safe_filename(sample_id)}__batch.json"


def _retriever_batch_trace_path(project_root: Path, sample_id: str) -> Path:
    return _trace_dir(project_root, "golden_retriever") / f"{GoldenRetriever._safe_filename(sample_id)}__batch.json"


def _pipeline_trace_path(project_root: Path, sample_id: str) -> Path:
    return _trace_dir(project_root, "pipeline") / f"{GoldenRetriever._safe_filename(sample_id)}.json"


def _pipeline_batch_trace_path(project_root: Path, sample_prefix: str) -> Path:
    return _trace_dir(project_root, "pipeline") / f"{GoldenRetriever._safe_filename(sample_prefix)}__batch.json"


def _checker_feedback_text(checker_output: Dict[str, Any]) -> str:
    slot_states = checker_output.get("slot_fill_state", []) or []
    gaps = checker_output.get("remaining_gaps", []) or []
    repairs = checker_output.get("repair_requests", []) or []
    unresolved_slots = [
        slot
        for slot in slot_states
        if str(slot.get("state", "")).strip().lower().startswith("unresolved")
    ]
    if not unresolved_slots and not gaps and not repairs:
        return "None"
    lines: List[str] = []
    summary = str(checker_output.get("evidence_state_summary", "")).strip()
    if summary:
        lines.append(f"Evidence summary: {summary}")
    if unresolved_slots:
        lines.append("Unresolved slots:")
        for slot in unresolved_slots:
            lines.append(
                f"- doc={slot.get('doc', '')}; title={slot.get('title', '')}; state={slot.get('state', '')}; "
                f"basis={slot.get('basis', '')}; note={slot.get('note', '')}"
            )
    if gaps:
        lines.append("Uncertain points:")
        for gap in gaps:
            lines.append(
                f"- answer_key={gap.get('answer_key', '')}; issue={gap.get('issue', '')}; why={gap.get('why', '')}"
            )
    if repairs:
        lines.append("Requested repairs:")
        for repair in repairs:
            lines.append(
                f"- action={repair.get('action', '')}; target={repair.get('target', '')}; why={repair.get('why', '')}"
            )
    return "\n".join(lines)


def _extract_repair_scope(
    checker_output: Dict[str, Any],
    planner_output: Dict[str, Any],
) -> Dict[str, List[str]]:
    unresolved_slots = [
        slot
        for slot in (checker_output.get("slot_fill_state", []) or [])
        if str(slot.get("state", "")).strip().lower().startswith("unresolved")
    ]
    if unresolved_slots:
        doc_ids = sorted(
            {
                str(slot.get("doc", "")).strip()
                for slot in unresolved_slots
                if str(slot.get("doc", "")).strip()
            }
        )
        step_ids = sorted(
            {
                step_id
                for slot in unresolved_slots
                for step_id in re.findall(r"\bSTEP(?:_[A-Za-z0-9]+)+\b", str(slot.get("basis", "")).strip())
            }
        )
        return {"doc_ids": doc_ids, "step_ids": step_ids, "search_targets": []}

    texts: List[str] = []
    for gap in checker_output.get("remaining_gaps", []) or []:
        texts.extend(
            [
                str(gap.get("answer_key", "")).strip(),
                str(gap.get("issue", "")).strip(),
                str(gap.get("why", "")).strip(),
            ]
        )
    for repair in checker_output.get("repair_requests", []) or []:
        texts.extend(
            [
                str(repair.get("action", "")).strip(),
                str(repair.get("target", "")).strip(),
                str(repair.get("why", "")).strip(),
            ]
        )
    combined = "\n".join(text for text in texts if text)
    if not combined:
        return {"doc_ids": [], "step_ids": [], "search_targets": []}

    explicit_doc_ids = set(re.findall(r"\bDOC\d+\b", combined))
    explicit_step_ids = set(re.findall(r"\bSTEP(?:_[A-Za-z0-9]+)+\b", combined))

    step_to_doc: Dict[str, str] = {}
    step_to_target: Dict[str, str] = {}
    for step in planner_output.get("doc_execution_graph", []) or []:
        step_id = str(step.get("id", "")).strip()
        doc_id = str(step.get("doc", "")).strip()
        target_name = str(step.get("search_target", "")).strip()
        anchor = str(step.get("anchor", "")).strip()
        if step_id:
            step_to_doc[step_id] = doc_id
            step_to_target[step_id] = target_name
        if anchor and anchor in combined and doc_id:
            explicit_doc_ids.add(doc_id)
        if target_name and target_name in combined:
            explicit_doc_ids.add(doc_id)

    target_name_to_docs: Dict[str, set[str]] = {}
    for step in planner_output.get("doc_execution_graph", []) or []:
        target_name = str(step.get("search_target", "")).strip()
        doc_id = str(step.get("doc", "")).strip()
        if target_name and doc_id:
            target_name_to_docs.setdefault(target_name, set()).add(doc_id)

    explicit_target_names = set()
    for item in planner_output.get("search_targets", []) or []:
        target_name = str(item.get("name", "")).strip()
        if not target_name or target_name not in combined:
            continue
        # Ignore overly generic target names that fan out to most or all documents.
        target_docs = target_name_to_docs.get(target_name, set())
        if len(target_docs) <= 2:
            explicit_target_names.add(target_name)

    for step_id in list(explicit_step_ids):
        doc_id = step_to_doc.get(step_id)
        if doc_id:
            explicit_doc_ids.add(doc_id)
        target_name = step_to_target.get(step_id)
        if target_name:
            explicit_target_names.add(target_name)

    return {
        "doc_ids": sorted(explicit_doc_ids),
        "step_ids": sorted(explicit_step_ids),
        "search_targets": sorted(explicit_target_names),
    }


def _select_divider_runs_for_repair(divider_runs: List[Any], repair_scope: Dict[str, List[str]]) -> List[Any]:
    target_doc_ids = set(repair_scope.get("doc_ids", []) or [])
    if not target_doc_ids:
        return divider_runs
    filtered = [run for run in divider_runs if getattr(run, "doc_id", None) in target_doc_ids]
    return filtered or divider_runs


def _filter_planner_output_for_repair(
    planner_output: Dict[str, Any],
    repair_scope: Dict[str, List[str]],
) -> Dict[str, Any]:
    target_doc_ids = set(repair_scope.get("doc_ids", []) or [])
    target_step_ids = set(repair_scope.get("step_ids", []) or [])
    target_search_targets = set(repair_scope.get("search_targets", []) or [])
    if not target_doc_ids and not target_step_ids and not target_search_targets:
        return planner_output

    filtered = dict(planner_output)
    graph = planner_output.get("doc_execution_graph", []) or []
    kept_steps = []
    for step in graph:
        step_id = str(step.get("id", "")).strip()
        doc_id = str(step.get("doc", "")).strip()
        target_name = str(step.get("search_target", "")).strip()
        if (
            (target_doc_ids and doc_id in target_doc_ids)
            or (target_step_ids and step_id in target_step_ids)
            or (target_search_targets and target_name in target_search_targets)
        ):
            kept_steps.append(step)
    if kept_steps:
        filtered["doc_execution_graph"] = kept_steps
        kept_target_names = {
            str(step.get("search_target", "")).strip()
            for step in kept_steps
            if str(step.get("search_target", "")).strip()
        }
        if kept_target_names:
            filtered["search_targets"] = [
                item
                for item in (planner_output.get("search_targets", []) or [])
                if str(item.get("name", "")).strip() in kept_target_names
            ]
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the redesigned Divider→Planner→Retriever→Checker→Generator pipeline.")
    parser.add_argument("--jsonl_path", type=str, default=None)
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--set_id", type=int, default=None)
    parser.add_argument("--selected_index", type=int, default=None)
    parser.add_argument("--record_type", type=str, default=None)
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--max_per_combo", type=int, default=1)
    parser.add_argument("--max_items", type=int, default=3)
    parser.add_argument("--sample_prefix", type=str, default="e2e_arch")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_repair_rounds", type=int, default=1)
    args = parser.parse_args()

    project_root = default_project_root()
    jsonl_path = Path(args.jsonl_path) if args.jsonl_path else default_loong_jsonl(project_root)
    records = load_loong_records(jsonl_path)
    manifest_items = _select_manifest_items(records, args)

    llm = OpenAIChatLLM(model=args.model, base_url=args.base_url, temperature=args.temperature)
    divider = Divider(llm=llm, project_root=project_root)
    planner = Planner(llm=llm, project_root=project_root)
    retriever = GoldenRetriever(llm=llm, project_root=project_root)
    checker = Checker(llm=llm, project_root=project_root)
    generator = Generator(llm=llm, project_root=project_root)

    summaries: List[Dict[str, Any]] = []
    for item in manifest_items:
        record = records[item.selected_index]
        sample_id = f"{args.sample_prefix}_{item.sample_id}"

        divider_runs = divider.run_loong_record(record=record, sample_id_prefix=sample_id, save_trace=True)
        divider_batch_payload = {
            "sample_id": sample_id,
            "resolved_record_index": item.selected_index,
            "runs": [run.to_dict() for run in divider_runs],
        }
        divider_batch_path = _save_json(_divider_batch_trace_path(project_root, sample_id), divider_batch_payload)

        cycle_logs: List[Dict[str, Any]] = []
        checker_output = None
        planner_output = None
        planner_trace_path = None
        retriever_batch_path = None
        checker_input = None
        checker_feedback = "None"
        repair_scope = {"doc_ids": [], "step_ids": [], "search_targets": []}

        for cycle_index in range(args.max_repair_rounds + 1):
            cycle_sample_id = sample_id if cycle_index == 0 else f"{sample_id}_repair{cycle_index}"
            cycle_divider_runs = divider_runs if cycle_index == 0 else _select_divider_runs_for_repair(divider_runs, repair_scope)
            planner_input = Planner.from_divider_runs(
                record=record,
                divider_runs=cycle_divider_runs,
                checker_feedback_or_none=checker_feedback,
                sample_id=cycle_sample_id,
            )
            planner_output = planner.plan(planner_input, save_trace=True)
            if cycle_index > 0:
                filtered_planner_payload = _filter_planner_output_for_repair(planner_output.to_dict(), repair_scope)
                planner_output.search_targets = filtered_planner_payload.get("search_targets", planner_output.search_targets)
                planner_output.doc_execution_graph = filtered_planner_payload.get("doc_execution_graph", planner_output.doc_execution_graph)
                _save_json(
                    _trace_dir(project_root, "planner") / f"{Planner._safe_filename(cycle_sample_id)}.json",
                    {
                        "input": _load_trace(_trace_dir(project_root, "planner") / f"{Planner._safe_filename(cycle_sample_id)}.json")["input"],
                        "system_prompt": _load_trace(_trace_dir(project_root, "planner") / f"{Planner._safe_filename(cycle_sample_id)}.json")["system_prompt"],
                        "user_prompt": _load_trace(_trace_dir(project_root, "planner") / f"{Planner._safe_filename(cycle_sample_id)}.json")["user_prompt"],
                        "output": planner_output.to_dict(),
                    },
                )
            planner_trace_path = _trace_dir(project_root, "planner") / f"{Planner._safe_filename(cycle_sample_id)}.json"

            retriever_runs = retriever.run_execution_graph(
                record=record,
                planner_output=planner_output.to_dict(),
                divider_runs=cycle_divider_runs,
                sample_id_prefix=cycle_sample_id,
                save_trace=True,
            )
            retriever_batch_payload = {
                "sample_id": cycle_sample_id,
                "resolved_record_index": item.selected_index,
                "runs": [run.to_dict() for run in retriever_runs],
            }
            retriever_batch_path = _save_json(_retriever_batch_trace_path(project_root, cycle_sample_id), retriever_batch_payload)

            checker_input = checker.build_input(
                planner_trace=_load_trace(planner_trace_path),
                divider_batch_trace=_load_trace(divider_batch_path),
                retriever_batch_trace=_load_trace(retriever_batch_path),
                sample_id=f"{cycle_sample_id}_checker",
            )
            checker_output = checker.check(checker_input, save_trace=True)
            next_repair_scope = (
                _extract_repair_scope(checker_output.to_dict(), planner_output.to_dict())
                if not checker_output.sufficiency_verdict
                else {"doc_ids": [], "step_ids": [], "search_targets": []}
            )
            cycle_logs.append(
                {
                    "cycle_index": cycle_index,
                    "planner_trace": str(planner_trace_path),
                    "retriever_batch_trace": str(retriever_batch_path),
                    "checker_trace": str(_trace_dir(project_root, "checker") / f"{GoldenRetriever._safe_filename(checker_input.sample_id)}.json"),
                    "checker_verdict": checker_output.sufficiency_verdict,
                    "remaining_gaps": checker_output.remaining_gaps,
                    "repair_requests": checker_output.repair_requests,
                    "slot_fill_state": checker_output.slot_fill_state,
                    "repair_scope": next_repair_scope,
                }
            )
            if checker_output.sufficiency_verdict:
                break
            checker_feedback = _checker_feedback_text(checker_output.to_dict())
            repair_scope = next_repair_scope

        projected_answer = project_answer_from_checker(
            checker_output=checker_output.to_dict(),
            planner_output=planner_output.to_dict(),
        )

        generated_answer = None
        generator_trace_path = None
        gold_answer = record.get("answer")
        if checker_output.sufficiency_verdict:
            generator_input = generator.build_input(
                planner_trace=_load_trace(planner_trace_path),
                checker_output=checker_output.to_dict(),
                sample_id=f"{sample_id}_generator",
                gold_answer=gold_answer,
            )
            generator_output = generator.generate_answer(generator_input, save_trace=True)
            generated_answer = generator_output.parsed_final_answer
            generator_trace_path = str(_trace_dir(project_root, "generator") / f"{GoldenRetriever._safe_filename(generator_input.sample_id)}.json")

        projection_eval = evaluate_prediction(projected_answer, gold_answer)
        generator_eval = evaluate_prediction(generated_answer, gold_answer)

        pipeline_trace = {
            "sample_id": sample_id,
            "resolved_record_index": item.selected_index,
            "record_meta": {
                "record_id": record.get("id"),
                "set_id": record.get("set"),
                "record_type": record.get("type"),
                "level": record.get("level"),
                "language": record.get("language"),
            },
            "module_traces": {
                "divider_batch": str(divider_batch_path),
                "planner": str(planner_trace_path),
                "retriever_batch": str(retriever_batch_path),
                "checker": str(_trace_dir(project_root, "checker") / f"{GoldenRetriever._safe_filename(checker_input.sample_id)}.json"),
                "generator": generator_trace_path,
            },
            "cycles": cycle_logs,
            "gold_answer": gold_answer,
            "projected_answer": projected_answer,
            "generated_answer": generated_answer,
            "projection_eval": projection_eval.to_dict(),
            "generator_eval": generator_eval.to_dict(),
        }
        pipeline_trace_path = _save_json(_pipeline_trace_path(project_root, sample_id), pipeline_trace)

        summaries.append(
            {
                "sample_id": sample_id,
                "record_type": item.record_type,
                "level": item.level,
                "selected_index": item.selected_index,
                "checker_verdict": checker_output.sufficiency_verdict,
                "gold_answer": gold_answer,
                "projected_answer": projected_answer,
                "generated_answer": generated_answer,
                "projection_exact_match": projection_eval.exact_match,
                "projection_pair_f1": projection_eval.pair_f1,
                "generator_exact_match": generator_eval.exact_match,
                "generator_pair_f1": generator_eval.pair_f1,
                "pipeline_trace_path": str(pipeline_trace_path),
            }
        )

    batch_payload = {
        "sample_prefix": args.sample_prefix,
        "jsonl_path": str(jsonl_path),
        "manifest_path": args.manifest_path,
        "filters": {
            "set_id": args.set_id,
            "selected_index": args.selected_index,
            "record_type": args.record_type,
            "level": args.level,
            "max_per_combo": args.max_per_combo,
            "max_items": args.max_items,
            "max_repair_rounds": args.max_repair_rounds,
        },
        "runs": summaries,
    }
    _save_json(_pipeline_batch_trace_path(project_root, args.sample_prefix), batch_payload)
    print(json.dumps(batch_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
