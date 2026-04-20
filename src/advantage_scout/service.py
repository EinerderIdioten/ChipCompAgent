from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .candidate_retriever import retrieve_candidates
from .input_adapters import RawInput, ingest_clipboard_text, ingest_text_file, ingest_xlsx
from .llm import OpenAICompatibleLLMClient
from .memory import append_decision
from .normalizer import normalize
from .schema import AnalysisResult, IndexedRow, QueryPlan, RunConfig, SelectionResult
from .validator import validate


class AdvantageScoutService:
    def run_config(self, config: RunConfig) -> AnalysisResult:
        query_input = self._load_input(
            input_type=config.input_type,
            input_path=config.input_path,
            input_text=config.input_text,
            sheet_name=config.sheet_name,
            default_source="query",
        )
        baseline_input = self._load_input(
            input_type=config.baseline_type,
            input_path=config.baseline_path,
            input_text=config.baseline_text,
            sheet_name=config.baseline_sheet_name,
            default_source="baseline",
        ) if config.baseline_type else None

        query_table = normalize(query_input)
        query_validation = validate(query_table)
        if not query_validation.valid:
            return AnalysisResult(normalized_table=query_table, validation=query_validation)

        query_rows = self._index_rows(query_table.rows, prefix="q")
        if baseline_input is None:
            return AnalysisResult(
                normalized_table=query_table,
                validation=query_validation,
                query_rows=query_rows,
            )

        baseline_table = normalize(baseline_input)
        baseline_validation = validate(baseline_table)
        if not baseline_validation.valid:
            return AnalysisResult(
                normalized_table=query_table,
                validation=query_validation,
                baseline_table=baseline_table,
                baseline_validation=baseline_validation,
                query_rows=query_rows,
                baseline_rows=self._index_rows(baseline_table.rows, prefix="b"),
            )

        baseline_rows = self._index_rows(baseline_table.rows, prefix="b")
        llm_client = OpenAICompatibleLLMClient(config.llm)
        query_plans: list[QueryPlan] = []
        selections: list[SelectionResult] = []
        baseline_tiles = self._tile_rows(baseline_rows)

        if config.use_local_retrieval:
            query_plans, selections = self._run_local_retrieval_selection(
                llm_client=llm_client,
                query_rows=query_rows,
                baseline_rows=baseline_rows,
                top_k=config.top_k,
                candidate_top_n=config.candidate_top_n,
                retrieval_batch_size=config.retrieval_batch_size,
                retrieval_weights=config.retrieval_weights,
            )
            return self._build_result(
                query_table=query_table,
                query_validation=query_validation,
                baseline_table=baseline_table,
                baseline_validation=baseline_validation,
                query_rows=query_rows,
                baseline_rows=baseline_rows,
                query_plans=query_plans,
                selections=selections,
                log_decisions=config.log_decisions,
            )

        if len(baseline_tiles) == 1 and self._fits_single_batch(query_rows, baseline_rows):
            query_plans, selections = self._run_batched_direct_selection(
                llm_client=llm_client,
                query_rows=query_rows,
                baseline_rows=baseline_rows,
                top_k=config.top_k,
            )
            return self._build_result(
                query_table=query_table,
                query_validation=query_validation,
                baseline_table=baseline_table,
                baseline_validation=baseline_validation,
                query_rows=query_rows,
                baseline_rows=baseline_rows,
                query_plans=query_plans,
                selections=selections,
                log_decisions=config.log_decisions,
            )

        for query_row in query_rows:
            if len(baseline_tiles) == 1:
                query_plan, direct_selections = llm_client.select_rows_direct(
                    query_row=query_row,
                    baseline_rows=baseline_rows,
                    top_k=config.top_k,
                )
                query_plans.append(query_plan)
                selections.extend(direct_selections)
                continue

            query_plan = llm_client.build_query_plan(query_row)
            query_plans.append(query_plan)
            tile_candidates = self._select_candidates(
                llm_client=llm_client,
                query_row=query_row,
                query_plan=query_plan,
                baseline_rows=baseline_rows,
                top_k=config.top_k,
                baseline_tiles=baseline_tiles,
            )
            if not tile_candidates:
                continue
            selections.extend(
                llm_client.rerank_candidates(
                    query_row=query_row,
                    query_plan=query_plan,
                    candidates=tile_candidates,
                    top_k=config.top_k,
                )
            )

        if config.log_decisions:
            for selection in selections:
                append_decision(
                    asdict(selection),
                    context={
                        "query_row_id": selection.query_row_id,
                        "baseline_row_id": selection.baseline_row_id,
                        "query_source": query_table.source,
                        "baseline_source": baseline_table.source,
                    },
                )

        return AnalysisResult(
            normalized_table=query_table,
            validation=query_validation,
            baseline_table=baseline_table,
            baseline_validation=baseline_validation,
            query_rows=query_rows,
            baseline_rows=baseline_rows,
            query_plans=query_plans,
            selections=selections,
        )

    def _run_local_retrieval_selection(
        self,
        llm_client: OpenAICompatibleLLMClient,
        query_rows: list[IndexedRow],
        baseline_rows: list[IndexedRow],
        top_k: int,
        candidate_top_n: int,
        retrieval_batch_size: int,
        retrieval_weights: dict[str, float] | None,
    ) -> tuple[list[QueryPlan], list[SelectionResult]]:
        query_plans: list[QueryPlan] = []
        selections: list[SelectionResult] = []
        shortlist_size = max(candidate_top_n, top_k)

        query_candidate_sets = [
            (
                query_row,
                retrieve_candidates(
                    query_row=query_row,
                    baseline_rows=baseline_rows,
                    top_n=shortlist_size,
                    weights=retrieval_weights,
                ),
            )
            for query_row in query_rows
        ]

        batch_size = max(retrieval_batch_size, 1)
        for batch_start in range(0, len(query_candidate_sets), batch_size):
            batch = query_candidate_sets[batch_start : batch_start + batch_size]
            if len(batch) == 1:
                query_row, candidate_rows = batch[0]
                query_plan, direct_selections = llm_client.select_rows_direct(
                    query_row=query_row,
                    baseline_rows=candidate_rows,
                    top_k=top_k,
                )
                query_plans.append(query_plan)
                selections.extend(direct_selections)
                continue

            try:
                batch_query_plans, batch_selections = llm_client.select_candidate_sets_batch(
                    query_candidate_sets=batch,
                    top_k=top_k,
                )
                query_plans.extend(batch_query_plans)
                selections.extend(batch_selections)
            except Exception:
                for query_row, candidate_rows in batch:
                    query_plan, direct_selections = llm_client.select_rows_direct(
                        query_row=query_row,
                        baseline_rows=candidate_rows,
                        top_k=top_k,
                    )
                    query_plans.append(query_plan)
                    selections.extend(direct_selections)

        return query_plans, selections

    def load_config_file(self, path: str | Path) -> RunConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return RunConfig(**payload)

    def result_to_dict(self, result: AnalysisResult) -> dict[str, Any]:
        return asdict(result)

    def _load_input(
        self,
        input_type: str | None,
        input_path: str | None,
        input_text: str | None,
        sheet_name: str | None,
        default_source: str,
    ) -> RawInput:
        if input_type is None:
            raise ValueError(f"{default_source} input_type is required")

        if input_type == "raw_text":
            if input_text is None:
                raise ValueError(f"{default_source} input_text is required for raw_text")
            return ingest_clipboard_text(input_text, source=default_source)

        if input_type == "text_file":
            if input_path is None:
                raise ValueError(f"{default_source} input_path is required for text_file")
            return ingest_text_file(input_path)

        if input_type == "xlsx":
            if input_path is None:
                raise ValueError(f"{default_source} input_path is required for xlsx")
            return ingest_xlsx(input_path, sheet_name=sheet_name)

        raise ValueError(f"unsupported {default_source} input_type: {input_type}")

    def _index_rows(self, rows: list[dict[str, Any]], prefix: str) -> list[IndexedRow]:
        return [IndexedRow(row_id=f"{prefix}{index}", values=row) for index, row in enumerate(rows, start=1)]

    def _select_candidates(
        self,
        llm_client: OpenAICompatibleLLMClient,
        query_row: IndexedRow,
        query_plan: QueryPlan,
        baseline_rows: list[IndexedRow],
        top_k: int,
        baseline_tiles: list[list[IndexedRow]] | None = None,
    ) -> list[SelectionResult]:
        candidates: list[SelectionResult] = []
        seen_baseline_row_ids: set[str] = set()
        for baseline_tile in baseline_tiles or self._tile_rows(baseline_rows):
            for candidate in llm_client.select_tile_candidates(
                query_row=query_row,
                query_plan=query_plan,
                baseline_tile=baseline_tile,
                top_k=top_k,
            ):
                if candidate.baseline_row_id in seen_baseline_row_ids:
                    continue
                seen_baseline_row_ids.add(candidate.baseline_row_id)
                candidates.append(candidate)
        return candidates

    def _tile_rows(self, rows: list[IndexedRow]) -> list[list[IndexedRow]]:
        max_tile_chars = 12000
        tiles: list[list[IndexedRow]] = []
        current_tile: list[IndexedRow] = []
        current_size = 0

        for row in rows:
            row_size = len(json.dumps(asdict(row), ensure_ascii=False))
            if current_tile and current_size + row_size > max_tile_chars:
                tiles.append(current_tile)
                current_tile = []
                current_size = 0
            current_tile.append(row)
            current_size += row_size

        if current_tile:
            tiles.append(current_tile)
        return tiles

    def _fits_single_batch(self, query_rows: list[IndexedRow], baseline_rows: list[IndexedRow]) -> bool:
        query_chars = sum(len(json.dumps(asdict(row), ensure_ascii=False)) for row in query_rows)
        baseline_chars = sum(len(json.dumps(asdict(row), ensure_ascii=False)) for row in baseline_rows)
        return query_chars + baseline_chars <= 24000

    def _run_batched_direct_selection(
        self,
        llm_client: OpenAICompatibleLLMClient,
        query_rows: list[IndexedRow],
        baseline_rows: list[IndexedRow],
        top_k: int,
    ) -> tuple[list[QueryPlan], list[SelectionResult]]:
        query_plans: list[QueryPlan] = []
        selections: list[SelectionResult] = []

        for query_chunk in self._chunk_query_rows(query_rows):
            try:
                chunk_plans, chunk_selections = llm_client.select_rows_direct_batch(
                    query_rows=query_chunk,
                    baseline_rows=baseline_rows,
                    top_k=top_k,
                )
            except Exception:
                chunk_plans = []
                chunk_selections = []
                for query_row in query_chunk:
                    query_plan, direct_selections = llm_client.select_rows_direct(
                        query_row=query_row,
                        baseline_rows=baseline_rows,
                        top_k=top_k,
                    )
                    chunk_plans.append(query_plan)
                    chunk_selections.extend(direct_selections)

            query_plans.extend(chunk_plans)
            selections.extend(chunk_selections)

        return query_plans, selections

    def _chunk_query_rows(self, query_rows: list[IndexedRow]) -> list[list[IndexedRow]]:
        max_chunk_chars = 4000
        chunks: list[list[IndexedRow]] = []
        current_chunk: list[IndexedRow] = []
        current_size = 0

        for row in query_rows:
            row_size = len(json.dumps(asdict(row), ensure_ascii=False))
            if current_chunk and current_size + row_size > max_chunk_chars:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0
            current_chunk.append(row)
            current_size += row_size

        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _build_result(
        self,
        query_table,
        query_validation,
        baseline_table,
        baseline_validation,
        query_rows,
        baseline_rows,
        query_plans,
        selections,
        log_decisions: bool,
    ) -> AnalysisResult:
        if log_decisions:
            for selection in selections:
                append_decision(
                    asdict(selection),
                    context={
                        "query_row_id": selection.query_row_id,
                        "baseline_row_id": selection.baseline_row_id,
                        "query_source": query_table.source,
                        "baseline_source": baseline_table.source,
                    },
                )

        return AnalysisResult(
            normalized_table=query_table,
            validation=query_validation,
            baseline_table=baseline_table,
            baseline_validation=baseline_validation,
            query_rows=query_rows,
            baseline_rows=baseline_rows,
            query_plans=query_plans,
            selections=selections,
        )
