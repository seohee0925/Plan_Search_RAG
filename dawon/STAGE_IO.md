# Stage Input and Output

This document summarizes what each stage receives and what it returns in the current pipeline.
The class definitions live in the team code under `src/agents/...`, but Dawon's runs write traces under `dawon/trace_logs/`.

## 1. Divider

Code:
- `src/agents/divider/divider.py`

Primary input object:
- `DividerInput`

Input fields:
- `instruction`: task instruction from the dataset record
- `question`: target question
- `doc_id`: local document id such as `DOC1`
- `display_title`: document title
- `document_text`: raw document text for that single document
- `packet_inventory`: packet preview list built before the LLM sees the document
- `region_inventory`: region preview list built before the LLM sees the document
- `sample_id`: run id
- `metadata`: record metadata

Primary output object:
- `DividerOutput`

Output fields:
- `raw_text`: raw LLM response
- `doc_id`
- `display_title`
- `doc_anchor`: short downstream anchor for the document
- `region_roles`: small set of important regions with roles like `lead_region`, `fallback_region`
- `divider_notes`: brief note about document structure and reading strategy
- `doc_map_summary`: compact summary for later agents
- `packet_store`: full packetized document
- `region_store`: full regionized document
- `search_views`: counts and TOC-like helpers
- `packet_count`
- `region_count`

What this stage really does:
- splits one raw document into packets and regions first
- asks the LLM to label the most useful regions
- outputs a searchable evidence space for Planner and Retriever

## 2. Planner

Code:
- `src/agents/planner/loong/planner.py`

Primary input object:
- `PlannerInput`

Input fields:
- `instruction`
- `question`
- `document_catalog`: one-line summary per document from Divider outputs
- `document_catalog_items`: structured catalog with region stores and roles
- `checker_feedback_or_none`: `None` on first pass, otherwise repair instructions
- `sample_id`
- `metadata`

Primary output object:
- `PlannerOutput`

Output fields:
- `raw_text`
- `task_goal`: what the real task is
- `task_model`: topology, atomic decision, grounding unit, coverage mode
- `answer_schema`: final answer keys and expected types
- `search_targets`: reusable document-local search goals
- `doc_execution_graph`: step-by-step plan over documents and regions
- `merge_policy`: how local evidence should later be merged
- `planning_notes`: risks and rationale

What this stage really does:
- turns Divider's document map into an execution graph
- decides which document and which regions should be searched for each sub-question

## 3. Golden Retriever

Code:
- `src/agents/golden_retriever/golden_retriever.py`

Primary input object:
- `GoldenRetrieverInput`

Input fields:
- `instruction`
- `question`
- `task_goal`
- `task_model`
- `answer_schema`
- `merge_policy`
- `search_target_name`
- `search_target_ask`
- `search_target_success_condition`
- `doc_id`
- `display_title`
- `doc_anchor`
- `focus_regions`
- `scoped_region_map`: human-readable map of chosen regions
- `scoped_packet_view`: actual packet texts around those regions
- `sample_id`
- `metadata`: includes step id, read strategy, stop condition

Primary output object:
- `GoldenRetrieverOutput`

Output fields:
- `raw_text`
- `search_trace`: locate / inspect / extract actions
- `evidence_units`: candidate evidence spans
- `search_status`: `scoped_complete`, `scoped_with_evidence`, or `scoped_no_hit`

What this stage really does:
- reads only the scoped parts of one document
- extracts grounded spans, but does not make the final task judgment

## 4. Checker

Code:
- `src/agents/checker/checker.py`

Primary input object:
- `CheckerInput`

Input fields:
- `instruction`
- `question`
- `task_goal`
- `task_model`
- `answer_schema`
- `merge_policy`
- `planning_notes`
- `document_catalog`
- `execution_graph`
- `integrated_evidence_state`: all retriever outputs merged into one ledger
- `sample_id`
- `metadata`: planner output, divider runs, retriever runs

Primary output object:
- `CheckerOutput`

Output fields:
- `raw_text`
- `evidence_state_summary`
- `verified_evidence_units`: evidence judged usable / weak / off-target / conflicted
- `projected_answer_state`: answer-level projection
- `slot_fill_state`: per-doc resolution state
- `remaining_gaps`
- `repair_requests`
- `sufficiency_verdict`: boolean

What this stage really does:
- decides whether the current evidence is enough
- projects the tentative answer state
- if not enough, generates repair requests for another planning cycle

## 5. Generator

Code:
- `src/agents/generator/generator.py`

Primary input object:
- `GeneratorInput`

Input fields:
- `instruction`
- `question`
- `task_goal`
- `task_model`
- `answer_schema`
- `projected_answer_state`
- `evidence_state_summary`
- `sample_id`
- `metadata`
- `reference_answer`: optional gold answer for inspection only

Primary output object:
- `GeneratorOutput`

Output fields:
- `raw_text`
- `render_summary`
- `final_answer`
- `parsed_final_answer`

What this stage really does:
- converts the Checker-approved answer state into the exact final answer format requested by the task

## Builder Methods Between Stages

Common transitions:
- Divider inputs: `Divider.build_inputs_from_loong(record)`
- Planner input: `Planner.from_divider_runs(record, divider_runs, checker_feedback_or_none)`
- Retriever inputs: `GoldenRetriever.build_inputs_from_execution_graph(record, planner_output, divider_runs)`
- Checker input: `Checker.build_input(planner_trace, divider_batch_trace, retriever_batch_trace)`
- Generator input: `Generator.build_input(planner_trace, checker_output)`

## Practical Divider Advice

If your own methodology is Divider-focused, these are the most important levers:
- packetization strategy in `src/agents/divider/divider.py`
- region-building logic in `src/agents/divider/divider.py`
- role-selection prompts in `dawon/prompts/divider_sys.txt` and `dawon/prompts/divider_user.txt`

Prompt-only experimentation is fastest.
If you want a stronger personal methodology, the next step is creating a Dawon-specific divider class that changes packetization or region grouping before the LLM labels regions.
