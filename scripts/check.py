"""Run locally exactly what CI runs.

This exists because CI failed on `ruff format --check` after a commit where
`ruff check` had been run and passed. Two different commands, one of which was
skipped locally — the classic way to discover a formatting failure from an
email six minutes later.

Keep the step list here in sync with .github/workflows/ci.yml. If they drift,
this script stops being worth running.

    uv run python scripts/check.py          # check only, like CI
    uv run python scripts/check.py --fix    # also apply the fixable ones
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def step(name: str, cmd: list[str]) -> tuple[bool, float]:
    started = time.perf_counter()
    print(f"{DIM}$ {' '.join(cmd)}{RESET}")
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.perf_counter() - started
    ok = result.returncode == 0
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"{mark}  {name}  {DIM}({elapsed:.1f}s){RESET}\n")
    return ok, elapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="Apply fixable lint and formatting")
    ap.add_argument("--skip-slow", action="store_true", help="Skip the red-team suite")
    args = ap.parse_args()

    py = sys.executable

    if args.fix:
        subprocess.run([py, "-m", "ruff", "check", "--fix", "."], cwd=ROOT)
        subprocess.run([py, "-m", "ruff", "format", "."], cwd=ROOT)
        print()

    steps: list[tuple[str, list[str]]] = [
        ("lint", [py, "-m", "ruff", "check", "."]),
        # The one that was skipped. `ruff check` does not imply this.
        ("format check", [py, "-m", "ruff", "format", "--check", "."]),
        ("unit tests", [py, "-m", "pytest", "tests/", "-q"]),
    ]
    if not args.skip_slow:
        steps.append(("red-team", [py, "evals/redteam.py"]))

    results = [(name, *step(name, cmd)) for name, cmd in steps]

    print("─" * 46)
    failed = 0
    for name, ok, elapsed in results:
        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  {mark}  {name:<16} {DIM}{elapsed:5.1f}s{RESET}")
        failed += not ok

    if failed:
        print(f"\n{RED}{failed} check(s) failed{RESET}  — CI will fail too.")
        if not args.fix:
            print("Try: uv run python scripts/check.py --fix")
        return 1
    print(f"\n{GREEN}all checks passed{RESET} — safe to push.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
