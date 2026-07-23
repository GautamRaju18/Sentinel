"""Install a GGUF trained in Colab, then measure it against the baseline.

The notebook hands back a single .gguf file. Turning that into a comparison
table is four fiddly manual steps — write a Modelfile, `ollama create`, edit
.env, run two evals — and getting any of them subtly wrong (a missing stop
token, forgetting to flip TRIAGE_BACKEND) produces numbers that look real and
are not. So it is one command.

    uv run python finetune/install_gguf.py ~/Downloads/sentinel-triage-f16.gguf

Add --no-eval to only register the model.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from sentinel.config import PROJECT_ROOT

MODEL_NAME = "sentinel-triage"

# temperature 0 and a tight top_p: this model has exactly one job and should
# not be creative about it. The stop token matters — without it Qwen keeps
# generating past the JSON and the parser sees trailing garbage.
MODELFILE = """\
FROM {gguf}

PARAMETER temperature 0
PARAMETER top_p 0.1
PARAMETER num_ctx 2048
PARAMETER stop "<|im_end|>"

SYSTEM \"\"\"{system}\"\"\"
"""


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kw)


def register(gguf: Path, name: str) -> None:
    from sentinel.agents.prompts import TRIAGE_SYSTEM

    if shutil.which("ollama") is None:
        raise SystemExit("ollama is not on PATH — install it from https://ollama.com")

    modelfile = gguf.parent / "Modelfile"
    modelfile.write_text(
        MODELFILE.format(gguf=gguf.name, system=TRIAGE_SYSTEM.strip()), encoding="utf-8"
    )
    print(f"wrote {modelfile}")
    run(["ollama", "create", name, "-f", str(modelfile)])


def point_env_at(name: str) -> None:
    """Flip .env to the fine-tuned backend, preserving everything else."""
    env = PROJECT_ROOT / ".env"
    if not env.exists():
        raise SystemExit(".env not found — copy .env.example first")

    lines = env.read_text(encoding="utf-8").splitlines()
    out, seen = [], {"MODEL_TRIAGE": False, "TRIAGE_BACKEND": False}
    for line in lines:
        if line.startswith("MODEL_TRIAGE="):
            out.append(f"MODEL_TRIAGE=ollama:{name}")
            seen["MODEL_TRIAGE"] = True
        elif line.startswith("TRIAGE_BACKEND="):
            out.append("TRIAGE_BACKEND=finetuned")
            seen["TRIAGE_BACKEND"] = True
        else:
            out.append(line)
    for key, found in seen.items():
        if not found:
            out.append(f"{key}=" + (f"ollama:{name}" if key == "MODEL_TRIAGE" else "finetuned"))

    env.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"updated .env -> MODEL_TRIAGE=ollama:{name}, TRIAGE_BACKEND=finetuned")


def smoke_test(name: str) -> bool:
    """Ask it one alert before spending minutes on the full eval.

    Uses the HTTP API, not `ollama run`. The CLI is a terminal renderer: it
    interleaves spinner redraws and ANSI escapes into stdout, so the "JSON"
    you capture contains control codes and duplicated fragments. On Windows it
    additionally fails to decode as cp1252. The model was fine; the harness
    was reading a rendered animation.
    """
    import json

    import httpx

    print("\nsmoke test …")
    alert = (
        "[prometheus] PodRestartLoop — billing-gateway\n"
        "labels: severity=critical team=payments env=production\n\n"
        "ALERT PodRestartLoop firing for billing-gateway\n"
        "restarts in the last hour: 7\n"
        "Containers terminating with exit code 137 (OOMKilled)."
    )
    try:
        response = httpx.post(
            "http://localhost:11434/api/chat",
            json={
                "model": name,
                "messages": [{"role": "user", "content": f"Classify this alert:\n\n{alert}"}],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=180,
        )
        response.raise_for_status()
        raw = response.json()["message"]["content"].strip()
    except Exception as e:
        print(f"  [FAIL] could not reach ollama: {type(e).__name__}: {e}")
        return False

    print(f"  raw: {raw[:300]}")
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        print("  [FAIL] no JSON object in the response")
        return False
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as e:
        print(f"  [FAIL] JSON did not parse: {e}")
        return False
    print(f"  parsed: {parsed}")
    missing = {"severity", "category"} - set(parsed)
    if missing:
        print(f"  [FAIL] missing keys: {missing}")
        return False
    print("  [OK] valid structured output")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", help="Path to the .gguf downloaded from Colab")
    ap.add_argument("--name", default=MODEL_NAME)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    gguf = Path(args.gguf).expanduser().resolve()
    if not gguf.exists():
        raise SystemExit(f"not found: {gguf}")
    print(f"gguf: {gguf}  ({gguf.stat().st_size / 1e9:.2f} GB)\n")

    register(gguf, args.name)
    point_env_at(args.name)

    if not smoke_test(args.name):
        raise SystemExit(
            "\nThe model did not return usable JSON. Do not trust an eval run on this — "
            "check that training actually converged before measuring it."
        )

    if args.no_eval:
        return

    print("\nrunning the held-out eval (this takes a few minutes) …")
    py = sys.executable
    run(
        [
            py,
            "evals/triage_eval.py",
            "--split",
            "test",
            "--limit",
            "40",
            "--out",
            "evals/results_finetuned.json",
        ],
        cwd=PROJECT_ROOT,
    )
    print()
    run([py, "evals/compare.py"], cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
