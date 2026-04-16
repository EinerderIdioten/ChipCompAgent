from __future__ import annotations

from .schema import IDENTIFIER_FIELDS, NormalizedTable, ValidationIssue, ValidationResult


def validate(table: NormalizedTable) -> ValidationResult:
    issues: list[ValidationIssue] = []

    if not table.rows:
        issues.append(ValidationIssue(level="error", message="no rows extracted"))

    for error in table.errors:
        issues.append(ValidationIssue(level="error", message=error))

    if table.rows and not any(_has_identifier(row) for row in table.rows):
        issues.append(ValidationIssue(level="error", message="all extracted rows are missing key identifiers"))

    for row_index, row in enumerate(table.rows, start=1):
        if not _has_identifier(row):
            issues.append(
                ValidationIssue(
                    level="warning",
                    message="row is missing both model_name and chip_name",
                    row_index=row_index,
                )
            )

        if _mixes_training_and_inference_without_label(row):
            issues.append(
                ValidationIssue(
                    level="warning",
                    message="training-related and throughput-related metrics appear together without workload_type; review manually if needed",
                    row_index=row_index,
                    field="workload_type",
                )
            )

    for warning in table.warnings:
        issues.append(ValidationIssue(level="warning", message=warning))

    return ValidationResult(valid=not any(issue.level == "error" for issue in issues), issues=issues)


def _has_identifier(row: dict[str, object]) -> bool:
    return any(row.get(field) not in (None, "") for field in IDENTIFIER_FIELDS)


def _mixes_training_and_inference_without_label(row: dict[str, object]) -> bool:
    has_training_metric = row.get("train_days_estimate") not in (None, "")
    has_throughput_metric = any(
        row.get(field) not in (None, "")
        for field in ("throughput", "tokens_per_sec_per_gpu", "model_tflops_per_gpu")
    )
    return has_training_metric and has_throughput_metric and row.get("workload_type") in (None, "")
