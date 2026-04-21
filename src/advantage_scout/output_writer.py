from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schema import AnalysisResult, EvidenceQuote, IndexedRow, SelectionResult


TABLE_HEADERS = [
    "query_row_id",
    "query_model_name",
    "query_chip_number",
    "query_global_batch_size",
    "query_micro_batch_size",
    "query_seq_length",
    "query_tp",
    "query_pp",
    "query_cp",
    "query_vp",
    "query_tokens_per_sec_per_gpu",
    "query_model_tflops_per_gpu",
    "query_train_days_estimate",
    "candidate_rank",
    "baseline_row_id",
    "baseline_model_name",
    "baseline_chip_name",
    "baseline_chip_number",
    "baseline_global_batch_size",
    "baseline_micro_batch_size",
    "baseline_seq_length",
    "baseline_tp",
    "baseline_pp",
    "baseline_cp",
    "baseline_vp",
    "baseline_throughput",
    "baseline_workload_type",
    "why_selected",
    "comparison_note",
    "supporting_quotes",
]


def build_selection_table(result: AnalysisResult, max_candidates_per_query: int = 3) -> list[dict[str, Any]]:
    query_lookup = {row.row_id: row for row in result.query_rows}
    grouped_selections: dict[str, list[SelectionResult]] = defaultdict(list)
    for selection in result.selections:
        grouped_selections[selection.query_row_id].append(selection)

    table_rows: list[dict[str, Any]] = []
    for query_row in result.query_rows:
        selections = sorted(grouped_selections.get(query_row.row_id, []), key=lambda item: item.rank)
        for selection in selections[:max_candidates_per_query]:
            table_rows.append(_selection_to_table_row(query_lookup[selection.query_row_id], selection))
    return table_rows


def write_selection_table(
    result: AnalysisResult,
    output_path: str | Path,
    max_candidates_per_query: int = 3,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_selection_table(result, max_candidates_per_query=max_candidates_per_query)

    if path.suffix.lower() == ".xlsx":
        _write_xlsx(path, rows)
        return

    write_selection_csv_rows(path, rows)


def write_selection_csv(
    result: AnalysisResult,
    output_path: str | Path,
    max_candidates_per_query: int = 3,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_selection_table(result, max_candidates_per_query=max_candidates_per_query)
    write_selection_csv_rows(path, rows)


def _selection_to_table_row(query_row: IndexedRow, selection: SelectionResult) -> dict[str, Any]:
    query_values = query_row.values
    baseline_values = selection.baseline_row
    return {
        "query_row_id": selection.query_row_id,
        "query_model_name": query_values.get("model_name"),
        "query_chip_number": query_values.get("chip_number"),
        "query_global_batch_size": query_values.get("global_batch_size"),
        "query_micro_batch_size": query_values.get("micro_batch_size"),
        "query_seq_length": query_values.get("seq_length"),
        "query_tp": query_values.get("tp"),
        "query_pp": query_values.get("pp"),
        "query_cp": query_values.get("cp"),
        "query_vp": query_values.get("vp"),
        "query_tokens_per_sec_per_gpu": query_values.get("tokens_per_sec_per_gpu"),
        "query_model_tflops_per_gpu": query_values.get("model_tflops_per_gpu"),
        "query_train_days_estimate": query_values.get("train_days_estimate"),
        "candidate_rank": selection.rank,
        "baseline_row_id": selection.baseline_row_id,
        "baseline_model_name": baseline_values.get("model_name"),
        "baseline_chip_name": baseline_values.get("chip_name"),
        "baseline_chip_number": baseline_values.get("chip_number"),
        "baseline_global_batch_size": baseline_values.get("global_batch_size"),
        "baseline_micro_batch_size": baseline_values.get("micro_batch_size"),
        "baseline_seq_length": baseline_values.get("seq_length"),
        "baseline_tp": baseline_values.get("tp"),
        "baseline_pp": baseline_values.get("pp"),
        "baseline_cp": baseline_values.get("cp"),
        "baseline_vp": baseline_values.get("vp"),
        "baseline_throughput": baseline_values.get("throughput"),
        "baseline_workload_type": baseline_values.get("workload_type"),
        "why_selected": selection.why_selected,
        "comparison_note": selection.comparison_note,
        "supporting_quotes": _render_quotes(selection.supporting_quotes),
    }


def _write_xlsx(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        from openpyxl import Workbook
    except ImportError as exc:
        raise RuntimeError("xlsx output requires openpyxl; install with `pip install -e .[xlsx]`") from exc

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "selected_candidates"
    worksheet.append(TABLE_HEADERS)
    for row in rows:
        worksheet.append([row.get(header) for header in TABLE_HEADERS])
    workbook.save(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def write_selection_csv_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_path, rows)


def _render_quotes(quotes: list[EvidenceQuote]) -> str:
    rendered_quotes = []
    for quote in quotes:
        payload = asdict(quote)
        rendered_quotes.append(
            "; ".join(f"{key}={value}" for key, value in payload.items() if value not in (None, ""))
        )
    return " | ".join(rendered_quotes)
