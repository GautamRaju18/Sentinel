"""Merge the LoRA adapter into the base model and register it with Ollama.

The adapter is useless to the rest of the system on its own — the graph talks
to models through Ollama, so the trained weights have to arrive there. Steps:

  1. load the base model in fp16 (not 4-bit: merging into quantised weights
     loses most of what was learned)
  2. merge the adapter
  3. save the merged model
  4. convert to GGUF and register a Modelfile with Ollama

Step 4 needs llama.cpp's converter. If it is not present the script stops after
step 3 and prints the exact commands to finish by hand, rather than failing in
a way that looks like the training was wasted.

    uv run --extra train python finetune/export_ollama.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_ADAPTER = HERE / "outputs" / "triage-qwen1.5b-lora"
DEFAULT_MERGED = HERE / "merged" / "triage-qwen1.5b"
MODEL_NAME = "sentinel-triage"

MODELFILE = """\
FROM {gguf}

# Deterministic classification — this model has exactly one job and should not
# be creative about it.
PARAMETER temperature 0
PARAMETER top_p 0.1
PARAMETER num_ctx 2048
PARAMETER stop "<|im_end|>"

SYSTEM \"\"\"{system}\"\"\"
"""


def merge(adapter: Path, out: Path, base: str) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading base {base} in fp16 …")
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
    )
    print(f"applying adapter {adapter} …")
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base, trust_remote_code=True).save_pretrained(out)
    print(f"merged model -> {out}")
    return out


def find_converter() -> Path | None:
    for candidate in (
        Path("llama.cpp/convert_hf_to_gguf.py"),
        Path("../llama.cpp/convert_hf_to_gguf.py"),
        HERE / "llama.cpp" / "convert_hf_to_gguf.py",
    ):
        if candidate.exists():
            return candidate.resolve()
    return None


def to_gguf(merged: Path) -> Path | None:
    converter = find_converter()
    gguf = merged.parent / f"{merged.name}-f16.gguf"
    if converter is None:
        print(
            "\nllama.cpp converter not found. Finish manually:\n"
            "  git clone https://github.com/ggerganov/llama.cpp\n"
            f"  python llama.cpp/convert_hf_to_gguf.py {merged} --outfile {gguf} "
            f"--outtype f16\n"
        )
        return None
    print("converting to GGUF …")
    subprocess.run(
        ["python", str(converter), str(merged), "--outfile", str(gguf), "--outtype", "f16"],
        check=True,
    )
    return gguf


def register(gguf: Path, name: str) -> None:
    from sentinel.agents.prompts import TRIAGE_SYSTEM

    if shutil.which("ollama") is None:
        print("ollama not on PATH; skipping registration")
        return
    modelfile = gguf.parent / "Modelfile"
    modelfile.write_text(
        MODELFILE.format(gguf=gguf.name, system=TRIAGE_SYSTEM.strip()), encoding="utf-8"
    )
    print(f"registering {name} with ollama …")
    subprocess.run(["ollama", "create", name, "-f", str(modelfile)], check=True)
    print(
        f"\nDone. Point the triage tier at it:\n"
        f"  MODEL_TRIAGE=ollama:{name}\n"
        f"  TRIAGE_BACKEND=finetuned\n\n"
        f"Then compare against the baseline:\n"
        f"  uv run python evals/triage_eval.py --out evals/results_finetuned.json\n"
        f"  uv run python evals/compare.py"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    ap.add_argument("--merged", default=str(DEFAULT_MERGED))
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--name", default=MODEL_NAME)
    args = ap.parse_args()

    adapter = Path(args.adapter)
    if not adapter.exists():
        raise SystemExit(f"adapter not found at {adapter}. Run train_qlora.py first.")

    merged = merge(adapter, Path(args.merged), args.base)
    if gguf := to_gguf(merged):
        register(gguf, args.name)


if __name__ == "__main__":
    main()
