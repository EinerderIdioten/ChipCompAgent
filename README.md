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

1. 🚀 ingest input
2. 🧹 normalize to an intermediate table
3. ✅ validate structure, then stop only on structural errors
4. 🔎 retrieve high-recall local candidates from the full baseline
5. 🤖 rerank candidates with the LLM
6. 🏁 return the top `top_k` rows with reasons and quoted evidence
7. 📝 optionally log the decision

Fallback: if local retrieval looks weak, split the baseline into tiles and use tiled LLM nomination as the slower safety path.

## Canonical schema

The normalized table should support at least these fields:

- 🧾 **Identity**: `model_name`, `model_scale_b`, `chip_name`, `chip_number`
- 📦 **Batching**: `global_batch_size`, `micro_batch_size`, `batch_size`, `seq_length`
- 🧩 **Parallelism**: `tp`, `pp`, `cp`, `vp`
- ⚡ **Performance**: `throughput`, `tokens_per_sec_per_gpu`, `model_tflops_per_gpu`, `train_days_estimate`
- 🗂️ **Context**: `workload_type`, `source`, `notes`

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

Row selection should not rely on brittle thresholds or brute-force LLM scans.

- 🚀 Use fast local retrieval first, then one LLM rerank pass.
- 💥 Do not ask the LLM to scan every tile unless retrieval fails.
- 🧠 Keep retrieval weights configurable now and learnable later.
- 🧾 Always return reasons plus quoted supporting cells.

This keeps latency lower than LLM-over-every-tile scanning while preserving LLM reasoning for the final decision.

## Learnable Candidate Retrieval

The retrieval stage is fast and auditable. It only builds a shortlist; the LLM makes the final ranked choice.

- 🧬 **Model family**: match names like `LLAMA3`, `Qwen3`, `GPT3`, `Nemotron`.
- ⚖️ **Model scale**: parse `B` sizes where possible and compare by log-distance.
- 📏 **Sequence length**: compare `seq_length` softly, not by exact match only.
- 🧩 **Parallelism**: compare `tp`, `pp`, `cp`, and `vp`.
- 🧮 **Batch/chip shape**: compare `global_batch_size`, `micro_batch_size`, and `chip_number`.
- 🎯 **Workload**: boost LLM pretraining; penalize unrelated multimodal/video rows for LLM queries.

Fair initial weights:

```json
{
  "model_family": 0.30,
  "model_scale": 0.25,
  "sequence_length": 0.15,
  "parallelism": 0.15,
  "batch_chip": 0.10,
  "workload": 0.05
}
```

Store these weights in config or memory, not as permanent magic constants. Also record retrieval features, candidate ranks, final LLM choices, and user feedback so a later RL or bandit loop can tune them.

Candidate contract:

- 🔎 score every baseline row locally for each query row
- 📦 keep a high-recall shortlist, for example top `30` to `50`
- 🧯 include extra candidates when fields are missing or scores are close
- 🤖 pass only the shortlist to the LLM for final rerank and reasoning
- 🧠 preserve feature evidence for future learning

## Tile Compatibility Principle

Tiling is useful, but tile-local ranks are not globally comparable.

- 🚀 Default path: local retrieval gives high-recall candidates fast.
- 🧱 Fallback path: tiled LLM nomination only when retrieval coverage is weak.
- 🏁 Final path: rerank pooled candidates globally before returning `top_k`.
- ⚠️ Never treat tile-local rank as final rank.
- 🧯 If batched LLM JSON breaks, split into smaller batches or one query per call.

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
