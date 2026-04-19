powered by arkclaw

# AdvantageScout

`AdvantageScout` is the first step of a larger benchmark-comparison agent for AI chip product work.

Its `v1` purpose is to:

- support two input formats now: raw clipboard text and xlsx files
- normalize both formats into the same intermediate structured table
- validate that table before any LLM selection
- retrieve only the top `x` baseline rows needed for output
- compare rows using LLM reasoning with quoted evidence
- refuse to hallucinate when information is insufficient
- keep the workflow auditable, structured, and easy to update

## Project structure

```text
advantage-scout/
├── .env.example
├── IMPLEMENTATION_PLAN.md
├── README.md
├── memory/
│   ├── benchmark_memory.json
│   └── decisions.jsonl
└── src/advantage_scout/
    ├── input_adapters.py
    ├── normalizer.py
    ├── validator.py
    ├── matcher.py
    ├── judge.py
    ├── llm.py
    ├── memory.py
    ├── service.py
    └── cli.py
```

## Input policy

Both of these are supported now:

- raw clipboard text
- xlsx files

They are equal first-class inputs. The system should not split them into current versus future support. Both should flow through the same downstream steps.

## Output policy

The runtime should accept one explicit output parameter:

- `top_k`: the number of baseline rows to return for each query row

The baseline may be much larger than the final result. The system should only return the top `top_k` rows selected by the agent workflow.

`top_k` should stay in the JSON config file, not in environment variables.

## Environment policy

Model-provider settings should be supplied through environment variables or a local env loader.

Current env vars:

- `DEEPSEEK_API_KEY`
- optional `DEEPSEEK_BASE_URL`
- optional `DEEPSEEK_MODEL`

Use `.env.example` as the template. Do not store real secrets in tracked files.

The CLI loads `.env` by default. You can also point it to another file with `--env-file`.

## Required pipeline

1. ingest input
2. normalize to intermediate table
3. validate normalized table structure
4. stop only if structural validation fails
5. build query understanding summary for each query row
6. split large baselines into tiles by context budget
7. ask the LLM to nominate candidate rows from each tile
8. rerank pooled candidates with the LLM
9. return only the top `top_k` rows
10. write the reason each selected row was chosen
11. optionally log the decision

## Canonical schema

The normalized table should support at least these fields:

- `model_name`
- `model_scale_b`
- `chip_name`
- `chip_number`
- `global_batch_size`
- `micro_batch_size`
- `batch_size`
- `seq_length`
- `tp`
- `pp`
- `cp`
- `vp`
- `throughput`
- `tokens_per_sec_per_gpu`
- `model_tflops_per_gpu`
- `train_days_estimate`
- `workload_type`
- `source`
- `notes`

Header alignment should be alias-based and flexible. Query information and baseline information do not need to fully align.

## Agent Roles

The comparison workflow should be organized as cooperating agents:

1. **Parse Agent**
   - ingest raw text or xlsx
   - normalize headers and values into structured rows
   - assign stable row ids

2. **Validation Agent**
   - check structural issues only
   - detect malformed rows, missing identifiers, and obvious parse failures
   - avoid inventing extra business fields just to satisfy validation

3. **Query Understanding Agent**
   - read each query row
   - summarize which fields look most relevant for retrieval
   - produce structured context for later ranking

4. **Tile Selection Agent**
   - read one baseline tile at a time
   - nominate the best candidate rows from that tile
   - explain why each nominated row is relevant
   - quote the exact cells used as evidence

5. **Rerank Agent**
   - compare all nominated candidates together
   - select the final top `top_k` rows
   - preserve evidence and reasoning for each selected row

6. **Output Writer Agent**
   - write the final output table
   - include why each row was selected
   - include the supporting quoted cells

## LLM Selection Policy

Row selection should not rely on hand-written scoring rules or hard-coded thresholds.

Recommended strategy:

- do not feed the whole large baseline in one call if it exceeds context budget
- do not judge each baseline row independently without global reranking
- use tiled candidate retrieval first, then one final reranking pass

This gives better scalability than one-shot prompting and better final ranking quality than naive row-by-row calls.

## Tile Compatibility Principle

Tiling introduces an evaluation risk:

- one tile may contain mostly weak rows
- another tile may contain stronger rows
- tile-local rankings are therefore not globally comparable by themselves

Because of that, the workflow should follow this rule:

- tile stage = candidate retrieval with high recall
- final rerank stage = true global evaluation with high precision

Implementation guidance:

- do not treat tile-local rank as the final global rank
- do not rely on tile-local scores as globally calibrated measures
- reuse one shared query understanding summary across all tiles
- keep a shortlist of plausible rows from each tile
- perform the true final ranking only after pooling candidates across tiles

Fast-path note:

- if the full baseline fits safely in one context window, skip tile retrieval
- for that case, run one direct global selection call per query row
- use the tiled multi-stage workflow only when the baseline is too large for one call

If both the full query set and the full baseline fit safely in one context window, batch all query rows into one direct global selection call first.

For reliability, the runtime may still split the query set into smaller batched calls and fall back to one-call-per-query if a provider response returns malformed JSON.

## Config Contract

The runtime is config-driven.

Current config fields include:

- query input source fields
- baseline input source fields
- `top_k`
- `llm`
- `log_decisions`

Examples:

- `config.example.json`
- `examples/mixed_input_xlsx_baseline.json`

Example command:

- `PYTHONPATH=src python -m advantage_scout.cli run --config examples/mixed_input_xlsx_baseline.json --env-file .env --with-explanations`
- `PYTHONPATH=src python -m advantage_scout.cli run --config examples/mixed_input_xlsx_baseline.json --env-file .env --with-explanations --output-file output.json`

## Output Table Requirements

Each returned row should include:

- query row id
- selected baseline row id
- rank within the top `top_k`
- selected row values
- `why_selected`
- quoted supporting cells
- optional comparison note

The reason text must state why the agent chose the row, not just restate the row contents.

## Maintenance Note

If the workflow, agent roles, or output contract changes, update this README in the same change.

## Migration Note

The active runtime path is now being migrated toward:

- `schema.py`
- `llm.py`
- `service.py`
- `memory.py`
- `cli.py`

Legacy rule-based files such as `matcher.py` and `judge.py` may still exist during migration, but they should not remain the long-term selection path.

They have now been removed from the active codebase.
