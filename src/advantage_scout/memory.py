from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEMORY_DIR = PROJECT_ROOT / "memory"
DECISIONS_PATH = MEMORY_DIR / "decisions.jsonl"


def append_decision(
    record: Any,
    context: dict[str, Any] | None = None,
    path: str | Path = DECISIONS_PATH,
) -> None:
    decision_path = Path(path)
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "record": _serialize_record(record),
        "context": context or {},
    }
    with decision_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _serialize_record(record: Any) -> Any:
    if is_dataclass(record):
        return asdict(record)
    return record
