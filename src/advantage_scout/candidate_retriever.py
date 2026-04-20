from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .schema import IndexedRow


DEFAULT_RETRIEVAL_WEIGHTS = {
    "model_family": 0.30,
    "model_scale": 0.25,
    "sequence_length": 0.15,
    "parallelism": 0.15,
    "batch_chip": 0.10,
    "workload": 0.05,
}


@dataclass
class RetrievedCandidate:
    row: IndexedRow
    score: float
    features: dict[str, float] = field(default_factory=dict)


def retrieve_candidates(
    query_row: IndexedRow,
    baseline_rows: list[IndexedRow],
    top_n: int,
    weights: dict[str, float] | None = None,
) -> list[IndexedRow]:
    weighted_scores = score_candidates(query_row, baseline_rows, weights=weights)
    return [candidate.row for candidate in weighted_scores[:top_n]]


def score_candidates(
    query_row: IndexedRow,
    baseline_rows: list[IndexedRow],
    weights: dict[str, float] | None = None,
) -> list[RetrievedCandidate]:
    active_weights = _normalized_weights(weights or DEFAULT_RETRIEVAL_WEIGHTS)
    scored = [
        _score_candidate(query_row=query_row, baseline_row=baseline_row, weights=active_weights)
        for baseline_row in baseline_rows
    ]
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored


def _score_candidate(
    query_row: IndexedRow,
    baseline_row: IndexedRow,
    weights: dict[str, float],
) -> RetrievedCandidate:
    query_values = query_row.values
    baseline_values = baseline_row.values
    features = {
        "model_family": _model_family_score(query_values, baseline_values),
        "model_scale": _model_scale_score(query_values, baseline_values),
        "sequence_length": _numeric_similarity(query_values.get("seq_length"), baseline_values.get("seq_length")),
        "parallelism": _group_similarity(query_values, baseline_values, ("tp", "pp", "cp", "vp")),
        "batch_chip": _group_similarity(
            query_values,
            baseline_values,
            ("global_batch_size", "micro_batch_size", "chip_number"),
        ),
        "workload": _workload_score(query_values, baseline_values),
    }
    score = sum(weights[name] * features[name] for name in weights)
    return RetrievedCandidate(row=baseline_row, score=score, features=features)


def _model_family_score(query_values: dict[str, Any], baseline_values: dict[str, Any]) -> float:
    query_family, _ = _parse_model_info(query_values.get("model_name"))
    baseline_family, _ = _parse_model_info(baseline_values.get("model_name"))
    baseline_context = _row_text(baseline_values)

    if _is_multimodal_context(baseline_context):
        return 0.05
    if query_family and baseline_family and query_family == baseline_family:
        return 1.0
    if query_family in _LLM_FAMILIES and baseline_family in _LLM_FAMILIES:
        return 0.55
    if baseline_family in _LLM_FAMILIES or _is_llm_context(baseline_context):
        return 0.45
    return 0.20


def _model_scale_score(query_values: dict[str, Any], baseline_values: dict[str, Any]) -> float:
    query_scale = query_values.get("model_scale_b")
    baseline_scale = baseline_values.get("model_scale_b")
    if query_scale is None:
        _, query_scale = _parse_model_info(query_values.get("model_name"))
    if baseline_scale is None:
        _, baseline_scale = _parse_model_info(baseline_values.get("model_name"))
    return _numeric_similarity(query_scale, baseline_scale, missing_score=0.45)


def _group_similarity(
    query_values: dict[str, Any],
    baseline_values: dict[str, Any],
    fields: tuple[str, ...],
) -> float:
    scores = [
        _numeric_similarity(query_values.get(field), baseline_values.get(field), missing_score=None)
        for field in fields
    ]
    available_scores = [score for score in scores if score is not None]
    if not available_scores:
        return 0.50
    return sum(available_scores) / len(available_scores)


def _numeric_similarity(query_value: Any, baseline_value: Any, missing_score: float | None = 0.50) -> float | None:
    query_number = _to_positive_float(query_value)
    baseline_number = _to_positive_float(baseline_value)
    if query_number is None or baseline_number is None:
        return missing_score
    if query_number == baseline_number:
        return 1.0
    return 1.0 / (1.0 + abs(math.log(query_number) - math.log(baseline_number)))


def _workload_score(query_values: dict[str, Any], baseline_values: dict[str, Any]) -> float:
    query_context = _row_text(query_values)
    baseline_context = _row_text(baseline_values)

    if _is_multimodal_context(baseline_context):
        return 0.05
    if "强化学习" in baseline_context or "reinforcement" in baseline_context:
        return 0.05
    if "预训练" in baseline_context or "pretrain" in baseline_context:
        return 1.0
    if "微调" in baseline_context or "finetune" in baseline_context or "lora" in baseline_context:
        return 0.70 if _looks_like_training_query(query_context) else 0.50
    if _is_llm_context(baseline_context):
        return 0.75
    return 0.45


def _parse_model_info(value: Any) -> tuple[str | None, float | None]:
    text = str(value or "").strip().lower()
    normalized = re.sub(r"[^a-z0-9.]+", " ", text)
    family = None
    for candidate in _LLM_FAMILIES:
        if candidate in normalized:
            family = candidate
            break

    scale = None
    scale_match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", normalized)
    if scale_match:
        scale = float(scale_match.group(1))

    return family, scale


def _to_positive_float(value: Any) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    if isinstance(value, int | float):
        number = float(value)
        return number if number > 0 else None
    match = re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", str(value).replace(",", ""), re.IGNORECASE)
    if not match:
        return None
    number = float(match.group(0))
    return number if number > 0 else None


def _row_text(values: dict[str, Any]) -> str:
    return " ".join(str(value).lower() for value in values.values() if value not in (None, ""))


def _is_llm_context(text: str) -> bool:
    return "大语言模型" in text or "llm" in text or any(family in text for family in _LLM_FAMILIES)


def _is_multimodal_context(text: str) -> bool:
    return "多模态" in text or any(token in text for token in ("t2v", "i2v", "ti2v", "vae", "wan2"))


def _looks_like_training_query(text: str) -> bool:
    return any(token in text for token in ("train", "tflop", "tokens", "gbs", "seq"))


def _normalized_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {name: max(float(weights.get(name, 0.0)), 0.0) for name in DEFAULT_RETRIEVAL_WEIGHTS}
    total = sum(cleaned.values())
    if total <= 0:
        return DEFAULT_RETRIEVAL_WEIGHTS
    return {name: value / total for name, value in cleaned.items()}


_LLM_FAMILIES = (
    "llama",
    "gpt",
    "nemotron",
    "qwen",
    "deepseek",
    "baichuan",
    "chatglm",
    "glm",
    "internlm",
    "mixtral",
    "mistral",
    "yi",
)
