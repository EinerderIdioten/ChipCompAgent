from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any
from urllib import error, request

from .schema import EvidenceQuote, IndexedRow, LLMConfig, QueryPlan, SelectionResult


class OpenAICompatibleLLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def select_rows_direct_batch(
        self,
        query_rows: list[IndexedRow],
        baseline_rows: list[IndexedRow],
        top_k: int,
    ) -> tuple[list[QueryPlan], list[SelectionResult]]:
        response = self._chat_json(
            system_prompt=(
                "You are the Direct Selection Agent for AdvantageScout. "
                "Read multiple query rows and a full baseline that fits in context. "
                "For each query row, select the final top rows directly. Return JSON only."
            ),
            user_payload={
                "task": (
                    "Understand each query row, compare it against the full baseline, "
                    "and return the final top rows with reasons and quoted supporting cells."
                ),
                "top_k": top_k,
                "query_rows": [asdict(row) for row in query_rows],
                "baseline_rows": [asdict(row) for row in baseline_rows],
                "required_output_schema": {
                    "results": [
                        {
                            "query_row_id": "string",
                            "query_summary": "string",
                            "selection_focus": "string",
                            "important_schema_keys": ["string"],
                            "ambiguity_notes": ["string"],
                            "selected": [
                                {
                                    "baseline_row_id": "string",
                                    "rank": "integer",
                                    "why_selected": "string",
                                    "comparison_note": "string",
                                    "supporting_quotes": [
                                        {
                                            "schema_key": "string",
                                            "query_value": "any",
                                            "baseline_value": "any",
                                            "note": "string",
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                },
            },
        )
        row_lookup = {row.row_id: row.values for row in baseline_rows}
        query_plans: list[QueryPlan] = []
        selections: list[SelectionResult] = []
        for result in response.get("results", []):
            query_row_id = str(result.get("query_row_id", ""))
            if not query_row_id:
                continue
            query_plans.append(
                QueryPlan(
                    query_row_id=query_row_id,
                    query_summary=str(result.get("query_summary", "")),
                    selection_focus=str(result.get("selection_focus", "")),
                    important_schema_keys=[str(item) for item in result.get("important_schema_keys", [])],
                    ambiguity_notes=[str(item) for item in result.get("ambiguity_notes", [])],
                )
            )
            for item in result.get("selected", []):
                baseline_row_id = str(item.get("baseline_row_id", ""))
                if baseline_row_id not in row_lookup:
                    continue
                selections.append(
                    SelectionResult(
                        query_row_id=query_row_id,
                        rank=int(item.get("rank", 0) or 0),
                        baseline_row_id=baseline_row_id,
                        baseline_row=row_lookup[baseline_row_id],
                        why_selected=str(item.get("why_selected", "")),
                        supporting_quotes=_parse_quotes(item.get("supporting_quotes", [])),
                        comparison_note=_optional_text(item.get("comparison_note")),
                    )
                )
        selections.sort(key=lambda item: (item.query_row_id, item.rank))
        return query_plans, selections

    def select_candidate_sets_batch(
        self,
        query_candidate_sets: list[tuple[IndexedRow, list[IndexedRow]]],
        top_k: int,
    ) -> tuple[list[QueryPlan], list[SelectionResult]]:
        response = self._chat_json(
            system_prompt=(
                "You are the Batched Rerank Agent for AdvantageScout. "
                "Each item contains one query row and its own locally retrieved candidate rows. "
                "For each item, select the final top rows only from that item's candidates. "
                "Return JSON only."
            ),
            user_payload={
                "task": (
                    "For each query item, compare the query row against its candidate rows, "
                    "then return final top rows with reasons and quoted supporting cells."
                ),
                "top_k": top_k,
                "items": [
                    {
                        "query_row_id": query_row.row_id,
                        "query_row": query_row.values,
                        "candidate_rows": [asdict(row) for row in candidate_rows],
                    }
                    for query_row, candidate_rows in query_candidate_sets
                ],
                "required_output_schema": {
                    "results": [
                        {
                            "query_row_id": "string",
                            "query_summary": "string",
                            "selection_focus": "string",
                            "important_schema_keys": ["string"],
                            "ambiguity_notes": ["string"],
                            "selected": [
                                {
                                    "baseline_row_id": "string",
                                    "rank": "integer",
                                    "why_selected": "string",
                                    "comparison_note": "string",
                                    "supporting_quotes": [
                                        {
                                            "schema_key": "string",
                                            "query_value": "any",
                                            "baseline_value": "any",
                                            "note": "string",
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                },
            },
        )

        row_lookup_by_query = {
            query_row.row_id: {row.row_id: row.values for row in candidate_rows}
            for query_row, candidate_rows in query_candidate_sets
        }
        query_plans: list[QueryPlan] = []
        selections: list[SelectionResult] = []
        for result in response.get("results", []):
            query_row_id = str(result.get("query_row_id", ""))
            row_lookup = row_lookup_by_query.get(query_row_id)
            if not row_lookup:
                continue
            query_plans.append(
                QueryPlan(
                    query_row_id=query_row_id,
                    query_summary=str(result.get("query_summary", "")),
                    selection_focus=str(result.get("selection_focus", "")),
                    important_schema_keys=[str(item) for item in result.get("important_schema_keys", [])],
                    ambiguity_notes=[str(item) for item in result.get("ambiguity_notes", [])],
                )
            )
            for item in result.get("selected", []):
                baseline_row_id = str(item.get("baseline_row_id", ""))
                if baseline_row_id not in row_lookup:
                    continue
                selections.append(
                    SelectionResult(
                        query_row_id=query_row_id,
                        rank=int(item.get("rank", 0) or 0),
                        baseline_row_id=baseline_row_id,
                        baseline_row=row_lookup[baseline_row_id],
                        why_selected=str(item.get("why_selected", "")),
                        supporting_quotes=_parse_quotes(item.get("supporting_quotes", [])),
                        comparison_note=_optional_text(item.get("comparison_note")),
                    )
                )
        selections.sort(key=lambda item: (item.query_row_id, item.rank))
        return query_plans, selections

    def select_rows_direct(
        self,
        query_row: IndexedRow,
        baseline_rows: list[IndexedRow],
        top_k: int,
    ) -> tuple[QueryPlan, list[SelectionResult]]:
        response = self._chat_json(
            system_prompt=(
                "You are the Direct Selection Agent for AdvantageScout. "
                "Read one query row and a full baseline that fits in context. "
                "Select the final top rows directly. Return JSON only."
            ),
            user_payload={
                "task": (
                    "Understand the query row, compare it against the full baseline, "
                    "and return the final top rows with reasons and quoted supporting cells."
                ),
                "query_row_id": query_row.row_id,
                "query_row": query_row.values,
                "top_k": top_k,
                "baseline_rows": [asdict(row) for row in baseline_rows],
                "required_output_schema": {
                    "query_summary": "string",
                    "selection_focus": "string",
                    "important_schema_keys": ["string"],
                    "ambiguity_notes": ["string"],
                    "selected": [
                        {
                            "baseline_row_id": "string",
                            "rank": "integer",
                            "why_selected": "string",
                            "comparison_note": "string",
                            "supporting_quotes": [
                                {
                                    "schema_key": "string",
                                    "query_value": "any",
                                    "baseline_value": "any",
                                    "note": "string",
                                }
                            ],
                        }
                    ],
                },
            },
        )

        query_plan = QueryPlan(
            query_row_id=query_row.row_id,
            query_summary=response.get("query_summary", ""),
            selection_focus=response.get("selection_focus", ""),
            important_schema_keys=[str(item) for item in response.get("important_schema_keys", [])],
            ambiguity_notes=[str(item) for item in response.get("ambiguity_notes", [])],
        )

        row_lookup = {row.row_id: row.values for row in baseline_rows}
        selections: list[SelectionResult] = []
        for item in response.get("selected", []):
            baseline_row_id = str(item.get("baseline_row_id", ""))
            if baseline_row_id not in row_lookup:
                continue
            selections.append(
                SelectionResult(
                    query_row_id=query_row.row_id,
                    rank=int(item.get("rank", len(selections) + 1)),
                    baseline_row_id=baseline_row_id,
                    baseline_row=row_lookup[baseline_row_id],
                    why_selected=str(item.get("why_selected", "")),
                    supporting_quotes=_parse_quotes(item.get("supporting_quotes", [])),
                    comparison_note=_optional_text(item.get("comparison_note")),
                )
            )

        selections.sort(key=lambda item: item.rank)
        return query_plan, selections[:top_k]

    def build_query_plan(self, query_row: IndexedRow) -> QueryPlan:
        response = self._chat_json(
            system_prompt=(
                "You are the Query Understanding Agent for AdvantageScout. "
                "Read one benchmark query row and produce a compact structured summary "
                "for retrieval. Return JSON only."
            ),
            user_payload={
                "task": "Summarize the query row for retrieval against a large benchmark baseline.",
                "query_row": query_row.values,
                "required_output_schema": {
                    "query_summary": "string",
                    "selection_focus": "string",
                    "important_schema_keys": ["string"],
                    "ambiguity_notes": ["string"],
                },
            },
        )
        return QueryPlan(
            query_row_id=query_row.row_id,
            query_summary=response.get("query_summary", ""),
            selection_focus=response.get("selection_focus", ""),
            important_schema_keys=[str(item) for item in response.get("important_schema_keys", [])],
            ambiguity_notes=[str(item) for item in response.get("ambiguity_notes", [])],
        )

    def select_tile_candidates(
        self,
        query_row: IndexedRow,
        query_plan: QueryPlan,
        baseline_tile: list[IndexedRow],
        top_k: int,
    ) -> list[SelectionResult]:
        response = self._chat_json(
            system_prompt=(
                "You are the Tile Selection Agent for AdvantageScout. "
                "Read one query row and one baseline tile. Select the most relevant rows "
                "from this tile only. Return JSON only."
            ),
            user_payload={
                "task": "Choose the best candidate baseline rows from this tile.",
                "query_row_id": query_row.row_id,
                "query_row": query_row.values,
                "query_plan": asdict(query_plan),
                "top_k": top_k,
                "baseline_tile": [asdict(row) for row in baseline_tile],
                "required_output_schema": {
                    "candidates": [
                        {
                            "baseline_row_id": "string",
                            "why_selected": "string",
                            "comparison_note": "string",
                            "supporting_quotes": [
                                {
                                    "schema_key": "string",
                                    "query_value": "any",
                                    "baseline_value": "any",
                                    "note": "string",
                                }
                            ],
                        }
                    ]
                },
            },
        )
        candidates: list[SelectionResult] = []
        row_lookup = {row.row_id: row.values for row in baseline_tile}
        for rank, item in enumerate(response.get("candidates", []), start=1):
            baseline_row_id = str(item.get("baseline_row_id", ""))
            if baseline_row_id not in row_lookup:
                continue
            candidates.append(
                SelectionResult(
                    query_row_id=query_row.row_id,
                    rank=rank,
                    baseline_row_id=baseline_row_id,
                    baseline_row=row_lookup[baseline_row_id],
                    why_selected=str(item.get("why_selected", "")),
                    supporting_quotes=_parse_quotes(item.get("supporting_quotes", [])),
                    comparison_note=_optional_text(item.get("comparison_note")),
                )
            )
        return candidates

    def rerank_candidates(
        self,
        query_row: IndexedRow,
        query_plan: QueryPlan,
        candidates: list[SelectionResult],
        top_k: int,
    ) -> list[SelectionResult]:
        response = self._chat_json(
            system_prompt=(
                "You are the Rerank Agent for AdvantageScout. "
                "Read the pooled candidate rows and return the final top rows. "
                "Return JSON only."
            ),
            user_payload={
                "task": "Select the final top baseline rows for this query row.",
                "query_row_id": query_row.row_id,
                "query_row": query_row.values,
                "query_plan": asdict(query_plan),
                "top_k": top_k,
                "candidates": [asdict(candidate) for candidate in candidates],
                "required_output_schema": {
                    "selected": [
                        {
                            "baseline_row_id": "string",
                            "rank": "integer",
                            "why_selected": "string",
                            "comparison_note": "string",
                            "supporting_quotes": [
                                {
                                    "schema_key": "string",
                                    "query_value": "any",
                                    "baseline_value": "any",
                                    "note": "string",
                                }
                            ],
                        }
                    ]
                },
            },
        )
        candidate_lookup = {candidate.baseline_row_id: candidate for candidate in candidates}
        selected: list[SelectionResult] = []
        for item in response.get("selected", []):
            baseline_row_id = str(item.get("baseline_row_id", ""))
            candidate = candidate_lookup.get(baseline_row_id)
            if candidate is None:
                continue
            selected.append(
                SelectionResult(
                    query_row_id=query_row.row_id,
                    rank=int(item.get("rank", len(selected) + 1)),
                    baseline_row_id=baseline_row_id,
                    baseline_row=candidate.baseline_row,
                    why_selected=str(item.get("why_selected", candidate.why_selected)),
                    supporting_quotes=_parse_quotes(item.get("supporting_quotes", [])) or candidate.supporting_quotes,
                    comparison_note=_optional_text(item.get("comparison_note")) or candidate.comparison_note,
                )
            )

        selected.sort(key=lambda item: item.rank)
        return selected[:top_k]

    def _chat_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key env var '{self.config.api_key_env}'. "
                "Set it in your shell or a local .env loader before running."
            )

        body = {
            "model": os.getenv("DEEPSEEK_MODEL", self.config.model),
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        }

        base_url = os.getenv("DEEPSEEK_BASE_URL", self.config.base_url)
        endpoint = base_url.rstrip("/") + "/chat/completions"
        request_data = json.dumps(body).encode("utf-8")
        http_request = request.Request(
            endpoint,
            data=request_data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed: {exc.code} {details}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response payload: {payload}") from exc

        parsed = _extract_json(content)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Expected a JSON object from the LLM, got: {content}")
        return parsed


def render_selection_explanation(selection: SelectionResult) -> str:
    lines = [
        f"query_row_id={selection.query_row_id}",
        f"baseline_row_id={selection.baseline_row_id}",
        f"rank={selection.rank}",
        f"why_selected={selection.why_selected}",
    ]
    if selection.comparison_note:
        lines.append(f"comparison_note={selection.comparison_note}")
    for quote in selection.supporting_quotes:
        lines.append(
            f"quote[{quote.schema_key}]="
            f"query:{quote.query_value!r}, baseline:{quote.baseline_value!r}, note:{quote.note}"
        )
    return "\n".join(lines)


def _extract_json(content: str) -> Any:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(f"Could not find JSON object in LLM output: {content}")
        return json.loads(content[start : end + 1])


def _parse_quotes(items: list[dict[str, Any]]) -> list[EvidenceQuote]:
    quotes: list[EvidenceQuote] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        quotes.append(
            EvidenceQuote(
                schema_key=str(item.get("schema_key", "")),
                query_value=item.get("query_value"),
                baseline_value=item.get("baseline_value"),
                note=str(item.get("note", "")),
            )
        )
    return quotes


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
