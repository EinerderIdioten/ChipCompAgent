# AdvantageScout Refactor Plan

This file tracks the active refactor plan for moving AdvantageScout from a rule-based selection pipeline to an LLM-driven agent workflow.

If the workflow, interfaces, or implementation order changes, update this file in the same change.

## Current Goal

Refactor the runtime so that:

- `top_k` stays in the JSON config
- API settings are provided through environment variables or a local env loader
- row selection is done by an LLM-driven multi-agent workflow
- the output includes `why_selected` and quoted supporting cells

## Frozen Shared Contract

The current shared contract lives in `src/advantage_scout/schema.py`.

### Core Input / Output Types

- `RunConfig`
  - runtime config loaded from JSON
  - contains input source config, baseline source config, `top_k`, and `llm`

- `LLMConfig`
  - model-provider config
  - current fields:
    - `provider`
    - `api_key_env`
    - `base_url`
    - `model`

- `IndexedRow`
  - a normalized row with a stable `row_id`
  - used by the LLM workflow instead of raw list indexes

- `QueryPlan`
  - structured output from the query-understanding stage
  - used to guide tile selection and reranking

- `SelectionResult`
  - final selected baseline row for one query row
  - includes:
    - `query_row_id`
    - `rank`
    - `baseline_row_id`
    - `baseline_row`
    - `why_selected`
    - `supporting_quotes`
    - `comparison_note`

- `AnalysisResult`
  - final runtime container
  - now supports:
    - `query_rows`
    - `baseline_rows`
    - `query_plans`
    - `selections`

## Active File-by-File Plan

### 1. `schema.py`

Status: done for the current contract freeze.

Purpose:

- define the shared interfaces first so later files stay compatible

### 2. `llm.py`

Status: implemented and syntax-checked.

Purpose:

- implement the actual LLM client layer
- keep all model calls behind one interface
- avoid leaking provider details across the rest of the codebase

Planned functions:

- `OpenAICompatibleLLMClient.__init__(config)`
  - stores provider settings from `LLMConfig`

- `build_query_plan(query_row)`
  - asks the model to summarize what matters for retrieving comparable rows
  - returns a `QueryPlan`

- `select_tile_candidates(query_row, query_plan, baseline_tile, top_k)`
  - asks the model to choose candidate rows from one baseline tile
  - this stage is for recall, not final global evaluation
  - returns provisional `SelectionResult` items

- `rerank_candidates(query_row, query_plan, candidates, top_k)`
  - asks the model to rerank pooled tile candidates
  - returns final `SelectionResult` items

- helper functions
  - parse JSON safely
  - convert quoted evidence into `EvidenceQuote`
  - render human-readable explanations if needed

### 3. `service.py`

Status: implemented and syntax-checked.

Purpose:

- orchestrate the end-to-end agent workflow
- keep business flow centralized

Planned runtime flow:

1. load query input
2. load baseline input
3. normalize both
4. validate both structurally
5. assign stable row ids
6. build one `QueryPlan` per query row
7. split large baselines into tiles
8. collect tile-level candidates
9. rerank pooled candidates globally
10. build final `AnalysisResult`
11. optionally log selections

Planned functions:

- `run_config(config)`
  - top-level pipeline entry

- `load_config_file(path)`
  - load JSON config
  - coerce nested `llm` dict into `LLMConfig`

- `_load_input(...)`
  - load raw text, text file, or xlsx

- `_index_rows(rows, prefix)`
  - generate stable `row_id`s

- `_tile_rows(rows)`
  - split baseline rows by context budget

- `_select_candidates(...)`
  - run tile-level selection and deduplicate candidate rows
  - keep enough plausible candidates so strong rows are not dropped early

### 4. `cli.py`

Status: implemented and syntax-checked.

Purpose:

- keep one config-driven command interface

Planned behavior:

- load one JSON config file
- load one optional local env file before the run
- run the service
- print structured JSON output
- optionally write structured JSON output to a real file
- optionally print human-readable explanations derived from `SelectionResult`

### 5. `memory.py`

Status: implemented and syntax-checked.

Purpose:

- store output records for later review

Planned change:

- log `SelectionResult`-style outputs instead of old judgement-only records

### 6. `validator.py`

Purpose:

- stay structural only
- not decide ranking or final selection

Current policy:

- parsing failures and missing identifiers can still block
- the training-style table shape should warn instead of hard fail

### 7. `matcher.py` and `judge.py`

Status: removed from the codebase.

Purpose during migration:

- remain present only until the new path is complete

Planned final state:

- not used by the active selection pipeline
- removed after the LLM workflow replaced them

## Environment Policy

Secrets must not be stored in code or committed config.

Current runtime env vars:

- `DEEPSEEK_API_KEY`
- optionally `DEEPSEEK_BASE_URL`
- optionally `DEEPSEEK_MODEL`

`top_k` remains in JSON config and is not an environment variable.

Status note:

- `.env.example` has been added
- `cli.py` now loads `.env` by default via `--env-file`
- the client now reads:
  - `DEEPSEEK_API_KEY`
  - optional `DEEPSEEK_BASE_URL`
  - optional `DEEPSEEK_MODEL`

## Output Contract

Each selected output row should include:

- `query_row_id`
- `rank`
- `baseline_row_id`
- selected baseline row values
- `why_selected`
- quoted supporting cells
- optional comparison note

The reason text must explain why the row was chosen, not only restate the row.

## Tile Compatibility Principle

Tiling introduces a key evaluation risk:

- one tile may contain only weak rows
- another tile may contain many strong rows
- tile-local rankings are therefore not directly comparable globally

Because of that, the workflow should follow this rule:

- tile stage = candidate retrieval with high recall
- final rerank stage = true global evaluation with high precision

Implementation guidance:

- do not treat tile-local rank as the final global rank
- do not rely on tile-local numeric scores as globally calibrated measures
- reuse one shared query understanding summary across all tiles
- keep a shortlist of plausible rows from each tile
- perform the true final ranking only after pooling candidates across tiles

Fast-path note:

- if the full baseline fits safely in one context window, skip tile retrieval and skip a separate query-plan call
- in that case, run one direct global selection call per query row
- use the tiled multi-stage workflow only when the baseline is too large for one call

Additional fast-path note:

- if both the full baseline and the full query set fit safely in one context window, batch all query rows into one direct global selection call
- use that batched path first when it fits, then fall back to one-call-per-query, then to the tiled workflow

Reliability note:

- large batched JSON responses can become malformed at the provider boundary
- therefore the runtime should batch query rows into smaller groups even when the baseline fits in one context window
- if one batched response still fails, fall back to one direct call per query row for that batch

## Current Next Step

Next step: run an end-to-end LLM-backed config flow with a local `.env` or exported DeepSeek env vars

Status note:

- a direct-selection fast path is now implemented for small baselines that fit in one context window
- this reduces the number of live LLM calls substantially for small end-to-end runs
- next improvement: support writing the actual output to a file for manual inspection
