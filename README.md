# Plan Search RAG

An evidence-space-first long-document reasoning pipeline built around a staged agent architecture:

`Divider -> Planner -> Golden Retriever -> Checker -> Generator`

## What It Does

This project is designed for long, noisy multi-document tasks such as:

- paper citation/reference reasoning
- legal classification and result mapping
- financial value extraction, comparison, grouping, and trend synthesis

Instead of pushing whole documents directly through one monolithic prompt, the system:

1. structures each document into an agent-friendly evidence space,
2. plans document-specific search targets,
3. retrieves grounded evidence units from scoped regions,
4. lets the checker decide which evidence is usable and whether the answer state is complete,
5. renders the final answer from the projected evidence state.

## Current Architecture

### Divider

- converts raw documents into packetized, region-aware evidence spaces
- preserves original text while adding document topology

### Planner

- reads the structured document map
- creates per-document execution plans and search targets

### Golden Retriever

- does scoped evidence extraction only
- returns grounded evidence units from the current document scope
- does not make the final task judgment

### Checker

- selects usable evidence
- projects answer state
- decides whether the current slots are sufficiently filled
- emits repair requests only for unresolved parts

### Generator

- renders the final answer from checker-approved projected state
- generator traces also store the reference/gold answer for easier inspection

## Project Layout

- `src/` : core pipeline and agent implementations
- `prompts/` : module prompts
- `artifacts/eval/loong_domain_level_manifest.json` : evaluation manifest
- `Loong/` : dataset files
- `1st_model_test/` : local run outputs gathered for model checkpoints or review

## Running Evaluation

Example:

```bash
python src/end-to-end_test.py \
  --manifest_path artifacts/eval/loong_domain_level_manifest.json \
  --max_items 9 \
  --sample_prefix first_model_full_v1 \
  --max_repair_rounds 1
```

## Notes

- `.env` and `trace_logs/` are ignored for safe sharing.
- Large local datasets and generated outputs are also ignored:
  - `Loong/data/`
  - `Loong/output/`
  - `sealqa/longseal.parquet`
  - `1st_model_test/`
- If you want to push this repository for the first time, you need to initialize git, commit, add the remote, and then push.
