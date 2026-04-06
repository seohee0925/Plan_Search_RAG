# Dawon Independent Pipeline

This folder is an independent workspace for Dawon's experiments.
Its prompts, traces, logs, run outputs, and agent implementations are isolated under `dawon/`.

## What Is Independent Here

- Prompt files live in `dawon/prompts/`
- Trace logs are written to `dawon/trace_logs/`
- Direct Qwen execution lives in `dawon/direct_qwen_llm.py`
- Independent agent copies live in `dawon/agents/`
- Independent evaluation utilities live in `dawon/orchestrator_utils.py`
- Team sample manifest is copied into `dawon/team_samples_manifest.json`
- The original team `src/` pipeline is not modified by Dawon experiments

## Recommended Workflow

1. Use the `structrag` virtualenv so `torch` and `transformers` are available.
2. Iterate on `dawon/prompts/divider_sys.txt` and `dawon/prompts/divider_user.txt`.
3. Run `run_divider_only.py` for fast divider experiments.
4. Run `run_pipeline.py` when you want the full Divider -> Planner -> Golden Retriever -> Checker -> Generator -> evaluation flow.
5. Check `dawon/trace_logs/translation_reports/` for Korean `.txt` reports generated after inference.

## Direct Qwen Setup

```bash
source /workspace/venvs/structrag/bin/activate
```

Defaults:
- GPUs: `0,1,2,3`
- model path: `/workspace/StructRAG/model/Qwen2.5-32B-Instruct`
- model name: `Qwen2.5-32B-Instruct`
- dtype: `float16`
- device map: `auto`

## Divider-Only Experiment

```bash
cd /workspace/Plan_Search_RAG
CUDA_VISIBLE_DEVICES=0,1,2,3 /workspace/venvs/structrag/bin/python dawon/run_divider_only.py --selected_index 0
```

Useful when you are tuning only the divider prompt.

Try multi-granularity mode:

```bash
cd /workspace/Plan_Search_RAG
CUDA_VISIBLE_DEVICES=0,1,2,3 /workspace/venvs/structrag/bin/python dawon/run_divider_only.py \
  --selected_index 1050 \
  --sample_prefix dawon_financial_granv1 \
  --divider_mode granv1
```

Try semantic chunk mode:

```bash
cd /workspace/Plan_Search_RAG
CUDA_VISIBLE_DEVICES=0,1,2,3 /workspace/venvs/structrag/bin/python dawon/run_divider_only.py \
  --selected_index 1050 \
  --sample_prefix dawon_financial_granv2 \
  --divider_mode granv2
```

## Full Pipeline Run

```bash
cd /workspace/Plan_Search_RAG
CUDA_VISIBLE_DEVICES=0,1,2,3 /workspace/venvs/structrag/bin/python dawon/run_pipeline.py \
  --selected_index 0 \
  --sample_prefix dawon_trial1 \
  --model_path /workspace/StructRAG/model/Qwen2.5-32B-Instruct
```

You can also use the wrapper:

```bash
cd /workspace/Plan_Search_RAG/dawon
./run_qwen_pipeline.sh --selected_index 0 --sample_prefix dawon_trial1
```

To enable the multi-granularity divider experiment:

```bash
cd /workspace/Plan_Search_RAG
CUDA_VISIBLE_DEVICES=0,1,2,3 /workspace/venvs/structrag/bin/python dawon/run_pipeline.py \
  --selected_index 1050 \
  --sample_prefix dawon_financial_granv1 \
  --divider_mode granv1
```

To enable the semantic-divider experiment:

```bash
cd /workspace/Plan_Search_RAG
CUDA_VISIBLE_DEVICES=0,1,2,3 /workspace/venvs/structrag/bin/python dawon/run_pipeline.py \
  --selected_index 1050 \
  --sample_prefix dawon_financial_granv2 \
  --divider_mode granv2
```

## Team Sample Runs

These are the three samples your teammates already selected:
- `paper`: selected index `0`
- `financial`: selected index `1050`
- `legal`: selected index `800`

Run all three together:

```bash
cd /workspace/Plan_Search_RAG/dawon
./run_team_samples.sh
```

Run all three together with `granv1`:

```bash
cd /workspace/Plan_Search_RAG/dawon
./run_team_samples_granv1.sh
```

Run all three together with `granv2`:

```bash
cd /workspace/Plan_Search_RAG/dawon
./run_team_samples_granv2.sh
```

Run them individually:

```bash
cd /workspace/Plan_Search_RAG
CUDA_VISIBLE_DEVICES=0,1,2,3 /workspace/venvs/structrag/bin/python dawon/run_pipeline.py --selected_index 0 --sample_prefix dawon_paper
CUDA_VISIBLE_DEVICES=0,1,2,3 /workspace/venvs/structrag/bin/python dawon/run_pipeline.py --selected_index 1050 --sample_prefix dawon_financial
CUDA_VISIBLE_DEVICES=0,1,2,3 /workspace/venvs/structrag/bin/python dawon/run_pipeline.py --selected_index 800 --sample_prefix dawon_legal
```

If you want to keep the exact same sample grouping as the existing experiment output, use:

```bash
cd /workspace/Plan_Search_RAG
CUDA_VISIBLE_DEVICES=0,1,2,3 /workspace/venvs/structrag/bin/python dawon/run_pipeline.py \
  --manifest_path /workspace/Plan_Search_RAG/dawon/team_samples_manifest.json \
  --sample_prefix dawon_team_samples \
  --divider_mode granv1
```

Or keep the same grouping with `granv2`:

```bash
cd /workspace/Plan_Search_RAG
CUDA_VISIBLE_DEVICES=0,1,2,3 /workspace/venvs/structrag/bin/python dawon/run_pipeline.py \
  --manifest_path /workspace/Plan_Search_RAG/dawon/team_samples_manifest.json \
  --sample_prefix dawon_team_samples_granv2 \
  --divider_mode granv2
```

## Where Results Go

- Divider batch: `dawon/trace_logs/divider/*__batch.json`
- Planner trace: `dawon/trace_logs/planner/*.json`
- Retriever batch: `dawon/trace_logs/golden_retriever/*__batch.json`
- Checker trace: `dawon/trace_logs/checker/*.json`
- Generator trace: `dawon/trace_logs/generator/*.json`
- Final pipeline summary: `dawon/trace_logs/pipeline/*__batch.json`
- Korean translation reports: `dawon/trace_logs/translation_reports/*_ko.txt`

## Korean Translation Reports

`dawon/run_pipeline.py` now generates Korean `.txt` reports after inference by default.

What gets translated:
- question and instruction
- full document text
- divider outputs
- planner / retriever / checker intermediate traces
- generator output and final answers

The translation step reuses the same Direct Qwen 32B model that runs inference.

If you want to skip translation because runtime is too long:

```bash
cd /workspace/Plan_Search_RAG
CUDA_VISIBLE_DEVICES=0,1,2,3 /workspace/venvs/structrag/bin/python dawon/run_pipeline.py \
  --selected_index 0 \
  --sample_prefix dawon_trial1 \
  --skip_translation_report
```

## Evaluation

`dawon/run_pipeline.py` keeps the original team behavior:
- it runs inference on the selected Loong sample
- it runs repair cycles if Checker says evidence is insufficient
- it runs Generator only when Checker verdict is `TRUE`
- it evaluates `projected_answer` and `generated_answer` against the dataset gold answer

## Divider Tuning Points

If your methodology is about Divider, start with these two files:
- `dawon/prompts/divider_sys.txt`
- `dawon/prompts/divider_user.txt`

If you later want algorithmic divider changes instead of prompt-only changes, create a Dawon-specific divider class under this folder and keep importing the rest of the team pipeline unchanged.

## `granv1` Divider Mode

`--divider_mode granv1` keeps the existing packetization, but it generates three region layers from the same source packets:
- `micro`: smaller local spans
- `meso`: the original baseline-style regions
- `macro`: larger section-level spans

The prompt-facing divider inventory still uses the `meso` layer so the Divider prompt does not explode in length.
The downstream Planner and Retriever can still select from all generated region layers because the saved `region_store` includes all of them.

## `granv2` Divider Mode

`--divider_mode granv2` keeps the baseline `meso` regions, then adds semantic overlay regions that try to preserve document meaning instead of only changing size.

Current semantic overlays include:
- `financial_row_bundle` / `financial_table_bundle` / `financial_table_note_bundle`
- `legal_case_profile` / `legal_reasoning_bundle` / `legal_decision_bundle`
- `paper_reference_entry` / `paper_citation_context`

The Divider prompt inventory still stays compact, while the saved `region_store` includes these semantic regions for downstream Planner and Retriever selection.
