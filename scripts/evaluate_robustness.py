#!/usr/bin/env python3
"""Evaluate saved real-data predictions at row, well, and component grain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_portfolio.evaluation import (  # noqa: E402
    evaluate_registry,
    load_registry,
    write_robustness_result,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "configs" / "robustness_registry.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "robustness_evaluation",
    )
    parser.add_argument(
        "--skip-manifest-verification",
        action="store_true",
        help="Skip source hash checks when iterating locally.",
    )
    args = parser.parse_args()
    registry_path = args.registry.resolve()
    result = evaluate_registry(
        load_registry(registry_path),
        ROOT,
        verify_manifests=not args.skip_manifest_verification,
    )
    write_robustness_result(result, args.output.resolve(), registry_path=registry_path)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "quality_checks": result.summary["quality_checks"],
                "best_macro_component_model_by_contract": result.summary[
                    "best_macro_component_model_by_contract"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
