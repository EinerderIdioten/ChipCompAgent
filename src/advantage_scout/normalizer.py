from __future__ import annotations

import re
from typing import Any

from .schema import CANONICAL_HEADERS, HEADER_ALIASES, NUMERIC_FIELDS, NormalizedTable, RawInput


def normalize(raw_input: RawInput) -> NormalizedTable:
    alias_lookup = _build_alias_lookup()
    rows: list[dict[str, Any]] = []
    warnings = list(raw_input.warnings)
    errors: list[str] = []

    for row_index, raw_row in enumerate(raw_input.rows, start=1):
        normalized_row = {header: None for header in CANONICAL_HEADERS}
        unmapped_fields: list[str] = []

        for raw_header, raw_value in raw_row.items():
            value = _clean_value(raw_value)
            if value is None:
                continue

            canonical_header = resolve_header(raw_header, alias_lookup)
            if canonical_header is None:
                unmapped_fields.append(f"{raw_header}={value}")
                continue

            if canonical_header in NUMERIC_FIELDS:
                parsed_number = parse_number(value)
                if parsed_number is None:
                    errors.append(f"row {row_index}: {canonical_header} could not parse numeric value {value!r}")
                    continue
                value = parsed_number

            if normalized_row[canonical_header] not in (None, "") and normalized_row[canonical_header] != value:
                warnings.append(f"row {row_index}: duplicate value for {canonical_header}; kept first value")
                continue
            normalized_row[canonical_header] = value

        if unmapped_fields:
            unmapped_note = "Unmapped fields: " + "; ".join(unmapped_fields)
            normalized_row["notes"] = _append_note(normalized_row.get("notes"), unmapped_note)

        if raw_input.source and normalized_row.get("source") in (None, ""):
            normalized_row["source"] = raw_input.source

        if any(value not in (None, "") for value in normalized_row.values()):
            rows.append(normalized_row)

    return NormalizedTable(
        rows=rows,
        source_type=raw_input.source_type,
        source=raw_input.source,
        warnings=warnings,
        errors=errors,
        raw_headers=raw_input.headers,
    )


def normalize_header(header: Any) -> str:
    text = str(header).strip().lower()
    text = text.replace("/", " per ")
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value

    text = str(value).strip().lower()
    if not text or text in {"-", "na", "n/a", "none", "unknown"}:
        return None

    compact_text = text.replace(",", "")
    match = re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", compact_text)
    if not match:
        return None

    number = float(match.group(0))
    suffix = compact_text[match.end() :].strip()

    if suffix.startswith("k"):
        number *= 1_000
    elif suffix.startswith("m"):
        number *= 1_000_000
    elif suffix.startswith("b") and "param" not in suffix:
        number *= 1_000_000_000

    return int(number) if number.is_integer() else number


def _build_alias_lookup() -> dict[str, str]:
    lookup = {normalize_header(header): header for header in CANONICAL_HEADERS}
    for canonical_header, aliases in HEADER_ALIASES.items():
        for alias in aliases:
            lookup[normalize_header(alias)] = canonical_header
    return lookup


def resolve_header(header: Any, alias_lookup: dict[str, str]) -> str | None:
    normalized_header = normalize_header(header)
    exact_match = alias_lookup.get(normalized_header)
    if exact_match is not None:
        return exact_match

    for alias, canonical_header in sorted(alias_lookup.items(), key=lambda item: len(item[0]), reverse=True):
        if normalized_header.startswith(alias) or alias in normalized_header:
            return canonical_header
    return None


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return value


def _append_note(existing_note: Any, new_note: str) -> str:
    if existing_note in (None, ""):
        return new_note
    return f"{existing_note}; {new_note}"
