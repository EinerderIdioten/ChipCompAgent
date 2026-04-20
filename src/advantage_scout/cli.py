from __future__ import annotations

import argparse
import json
from pathlib import Path

from .env import load_dotenv
from .llm import render_selection_explanation
from .output_writer import build_selection_table, write_selection_table
from .service import AdvantageScoutService


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "run":
        parser.error("only the 'run' command is supported")

    load_dotenv(args.env_file)
    service = AdvantageScoutService()
    config = service.load_config_file(args.config)
    result = service.run_config(config)

    output = service.result_to_dict(result)
    if args.with_explanations:
        output["explanations"] = [render_selection_explanation(item) for item in result.selections]
    output["selection_table"] = build_selection_table(
        result,
        max_candidates_per_query=config.output_max_candidates,
    )

    output_table_path = args.output_table_file or config.output_table_path
    if output_table_path:
        write_selection_table(
            result,
            output_table_path,
            max_candidates_per_query=config.output_max_candidates,
        )

    rendered_output = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output_file:
        args.output_file.write_text(rendered_output + "\n", encoding="utf-8")
    print(rendered_output)
    return 0 if result.validation.valid else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AdvantageScout from a single config file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the benchmark comparison pipeline")
    run_parser.add_argument("--config", type=Path, required=True, help="Path to a JSON config file")
    run_parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional local env file for API settings such as DEEPSEEK_API_KEY",
    )
    run_parser.add_argument(
        "--with-explanations",
        action="store_true",
        help="Include human-readable explanations derived from selected rows and quoted evidence",
    )
    run_parser.add_argument(
        "--output-file",
        type=Path,
        help="Optional path to write the JSON result to disk",
    )
    run_parser.add_argument(
        "--output-table-file",
        type=Path,
        help="Optional path to write a compact selected-candidate table as .xlsx or .csv",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
