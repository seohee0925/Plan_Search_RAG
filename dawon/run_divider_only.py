from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

DAWON_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DAWON_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dawon.direct_qwen_llm import DirectQwenLLM
from dawon.agents.divider import Divider
from dawon.orchestrator_utils import build_domain_level_manifest, default_loong_jsonl, load_loong_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run only Dawon's Divider for fast prompt iteration.")
    parser.add_argument("--jsonl_path", type=str, default=None)
    parser.add_argument("--selected_index", type=int, default=None)
    parser.add_argument("--record_type", type=str, default=None)
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--max_items", type=int, default=1)
    parser.add_argument("--sample_prefix", type=str, default="dawon_divider")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-32B-Instruct")
    parser.add_argument("--model_path", type=str, default="/workspace/StructRAG/model/Qwen2.5-32B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--skip_healthcheck", action="store_true")
    parser.add_argument("--divider_mode", type=str, default="default", choices=["default", "granv1", "granv2"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jsonl_path = Path(args.jsonl_path) if args.jsonl_path else default_loong_jsonl(REPO_ROOT)
    records = load_loong_records(jsonl_path)

    if args.selected_index is not None:
        indices = [args.selected_index]
    else:
        manifest = build_domain_level_manifest(records, max_per_combo=1)
        if args.record_type:
            manifest = [item for item in manifest if item.record_type == args.record_type]
        if args.level is not None:
            manifest = [item for item in manifest if item.level == args.level]
        indices = [item.selected_index for item in manifest[: args.max_items]]

    llm = DirectQwenLLM(
        model_path=args.model_path,
        model_name=args.model_name,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    if not args.skip_healthcheck:
        llm.healthcheck()

    divider = Divider(llm=llm, project_root=DAWON_ROOT, divider_mode=args.divider_mode)

    outputs: List[Dict[str, Any]] = []
    for index in indices:
        record = records[index]
        sample_id = f"{args.sample_prefix}_{record.get('type', 'unknown')}_{index}"
        runs = divider.run_loong_record(record=record, sample_id_prefix=sample_id, save_trace=True)
        outputs.append(
            {
                "selected_index": index,
                "record_id": record.get("id"),
                "question": record.get("question"),
                "runs": [run.to_dict() for run in runs],
            }
        )

    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
