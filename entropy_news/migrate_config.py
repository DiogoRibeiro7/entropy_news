"""Command-line helper to migrate legacy hyperparameter files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from entropy_news.utils.migration import legacy_args_to_config


def build_parser() -> argparse.ArgumentParser:
    """Return an argument parser configured for migration tasks."""

    parser = argparse.ArgumentParser(description="Migrate legacy model metadata")
    parser.add_argument("legacy_json", help="Path to the legacy JSON metadata file")
    parser.add_argument(
        "--output",
        default="output/model_config.json",
        help="Destination path for the generated ModelConfig",
    )
    parser.add_argument(
        "--overrides",
        default=None,
        help="Optional JSON string with configuration overrides",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the migration CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    overrides = json.loads(args.overrides) if args.overrides else None
    config = legacy_args_to_config(json.loads(Path(args.legacy_json).read_text()), overrides=overrides)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config.save(output_path)
    print(f"Configuration migrated to {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
