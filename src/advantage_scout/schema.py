from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


CANONICAL_HEADERS = [
    "model_name",
    "model_scale_b",
    "chip_name",
    "chip_number",
    "global_batch_size",
    "micro_batch_size",
    "batch_size",
    "seq_length",
    "tp",
    "pp",
    "cp",
    "vp",
    "throughput",
    "tokens_per_sec_per_gpu",
    "model_tflops_per_gpu",
    "train_days_estimate",
    "workload_type",
    "source",
    "notes",
]

HEADER_ALIASES = {
    "model_name": ["model", "model name", "network", "llm"],
    "model_scale_b": ["model scale", "model size", "params(b)", "params b", "billions of params"],
    "chip_name": ["chip", "chip name", "gpu", "gpu name", "accelerator", "device"],
    "chip_number": ["#-gpus", "# gpus", "num gpus", "gpu count", "cards", "chip number", "chips"],
    "global_batch_size": ["gbs", "global batch size"],
    "micro_batch_size": ["mbs", "micro batch size"],
    "batch_size": ["batch", "batch size", "bs"],
    "seq_length": ["sequence length", "seq length", "seq", "context length", "input length"],
    "tp": ["tp", "tensor parallel", "tensor parallelism"],
    "pp": ["pp", "pipeline parallel", "pipeline parallelism"],
    "cp": ["cp", "context parallel", "context parallelism"],
    "vp": ["vp", "virtual pipeline", "virtual pipeline stages"],
    "throughput": ["throughput", "tokens/s", "tok/s", "tokens per sec", "total throughput"],
    "tokens_per_sec_per_gpu": ["tokens / sec / gpu", "tokens/sec/gpu", "tok/s/gpu", "throughput per gpu"],
    "model_tflops_per_gpu": ["model tflop / sec / gpu", "model tflops per gpu", "tflops/gpu"],
    "train_days_estimate": ["est. time to train in days", "train days", "estimated training days"],
    "workload_type": ["workload", "task type", "mode"],
    "source": ["source", "url", "origin"],
    "notes": ["notes", "remark", "comments"],
}

NUMERIC_FIELDS = {
    "model_scale_b",
    "chip_number",
    "global_batch_size",
    "micro_batch_size",
    "batch_size",
    "seq_length",
    "tp",
    "pp",
    "cp",
    "vp",
    "throughput",
    "tokens_per_sec_per_gpu",
    "model_tflops_per_gpu",
    "train_days_estimate",
}

IDENTIFIER_FIELDS = {"model_name", "chip_name"}


@dataclass
class RawInput:
    rows: list[dict[str, Any]]
    source_type: str
    source: str | None = None
    headers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class NormalizedTable:
    rows: list[dict[str, Any]]
    source_type: str
    source: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    raw_headers: list[str] = field(default_factory=list)


@dataclass
class ValidationIssue:
    level: str
    message: str
    row_index: int | None = None
    field: str | None = None


@dataclass
class ValidationResult:
    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "warning"]


@dataclass
class EvidenceQuote:
    schema_key: str
    query_value: Any | None = None
    baseline_value: Any | None = None
    note: str = ""


@dataclass
class AnalysisResult:
    normalized_table: NormalizedTable
    validation: ValidationResult
    baseline_table: NormalizedTable | None = None
    baseline_validation: ValidationResult | None = None
    query_rows: list[IndexedRow] = field(default_factory=list)
    baseline_rows: list[IndexedRow] = field(default_factory=list)
    query_plans: list[QueryPlan] = field(default_factory=list)
    selections: list[SelectionResult] = field(default_factory=list)


@dataclass
class IndexedRow:
    row_id: str
    values: dict[str, Any]


@dataclass
class QueryPlan:
    query_row_id: str
    query_summary: str
    selection_focus: str
    important_schema_keys: list[str] = field(default_factory=list)
    ambiguity_notes: list[str] = field(default_factory=list)


@dataclass
class SelectionResult:
    query_row_id: str
    rank: int
    baseline_row_id: str
    baseline_row: dict[str, Any]
    why_selected: str
    supporting_quotes: list[EvidenceQuote] = field(default_factory=list)
    comparison_note: str | None = None


@dataclass
class LLMConfig:
    provider: str = "deepseek"
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"


@dataclass
class RunConfig:
    input_type: str
    input_path: str | None = None
    input_text: str | None = None
    baseline_type: str | None = None
    baseline_path: str | None = None
    baseline_text: str | None = None
    sheet_name: str | None = None
    baseline_sheet_name: str | None = None
    top_k: int = 3
    llm: LLMConfig = field(default_factory=LLMConfig)
    log_decisions: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.llm, dict):
            self.llm = LLMConfig(**self.llm)
