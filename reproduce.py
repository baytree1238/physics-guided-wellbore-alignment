#!/usr/bin/env python3
"""Cross-platform one-command entry point for the complete smoke reproduction."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BOOTSTRAP = ROOT / "scripts" / "bootstrap.py"
STEPS = (
    ROOT / "scripts" / "run_reproduce.py",
    ROOT / "scripts" / "run_deployment_smoke.py",
    ROOT / "scripts" / "build_portfolio_notebook.py",
    ROOT / "scripts" / "execute_portfolio_notebook.py",
    ROOT / "scripts" / "qa_math_render.py",
    ROOT / "scripts" / "verify_reproduction.py",
)


def main() -> int:
    print("\n[reproduce] pytest", flush=True)
    tests = subprocess.run(
        [sys.executable, str(BOOTSTRAP), "-m", "pytest"],
        cwd=ROOT,
        check=False,
    )
    if tests.returncode:
        return tests.returncode
    for step in STEPS:
        print(f"\n[reproduce] {step.relative_to(ROOT)}", flush=True)
        completed = subprocess.run(
            [sys.executable, str(BOOTSTRAP), str(step)],
            cwd=ROOT,
            check=False,
        )
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
