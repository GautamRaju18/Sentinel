# Fine-tuning the triage classifier

Alert triage is the right job for a small fine-tuned model: it runs on every
alert, the output is a fixed schema, and it needs no reasoning — only accurate
pattern recognition. That is exactly where a 1.5B model with a LoRA adapter
beats calling a large general model, on latency and cost, at equal or better
accuracy.

## Pipeline

```bash
uv run python finetune/generate_dataset.py
```
1000 train / 120 val / 120 test examples across 6 categories and 4 severities.

```bash
uv run --extra train python finetune/train_qlora.py
```
QLoRA on Qwen2.5-1.5B-Instruct. ~30 minutes on a 6GB RTX 3060.

> **On the development machine this does not work, and the supported path is
> [`colab_train.ipynb`](colab_train.ipynb).** Windows Smart App Control blocks
> the CUDA build of PyTorch — `torch/lib/shm.dll` and its dependencies are
> unsigned, so `import torch` fails with `WinError 4551`. The CPU wheel loads
> fine but would turn 30 minutes into an overnight run. Disabling Smart App
> Control on Windows 11 is irreversible without reinstalling the OS, so it was
> not done. The notebook uses identical hyperparameters on a free T4.

```bash
uv run --extra train python finetune/export_ollama.py
```
Merges the adapter, converts to GGUF, registers `sentinel-triage` with Ollama.

Then switch the backend and re-measure:

```bash
# in .env
MODEL_TRIAGE=ollama:sentinel-triage
TRIAGE_BACKEND=finetuned
```

```bash
uv run python evals/triage_eval.py --out evals/results_finetuned.json && uv run python evals/compare.py
```

## Why templates, not distillation

Distilling labels from a frontier model is the fashionable choice and is wrong
here. These labels are *definitional* — a total outage is P1 because of what the
alert says, not because a large model has an opinion. Template generation makes
every label correct by construction, reproducible from a seed, and free.

The realism comes from varying surface form aggressively: 5 alert sources with
genuinely different formats, 21 templates, randomised services, numbers,
phrasings and noise. The model must read past the format to the substance.

**The test split uses services that appear in no training example**
(`billing-gateway`, `fraud-detector`, …). A model that memorised service names
scores badly on it, which is the point.

## Choices, and why

| Choice | Reason |
|---|---|
| Qwen2.5-1.5B-Instruct | largest instruct model that trains comfortably in 6GB at 4-bit |
| 4-bit NF4 + double quantisation | the base is frozen, so its precision matters far less than the adapter's; double-quant saves ~0.4 bits/param, the difference between fitting and not |
| LoRA r=16, α=32 | r=8 underfits the JSON format; r=32 gains little and costs memory |
| attention **and** MLP targets | attention-only adapters learn the task but keep drifting on output format |
| gradient checkpointing, batch 1 × accum 8 | effective batch of 8 without holding 8 sequences of activations |
| `paged_adamw_8bit` | survives the memory spikes that would otherwise OOM at 6GB |
| loss on completion only | otherwise most of the gradient budget teaches the model to reproduce alert text it will always be given |
| `packing=False` | packing blurs example boundaries, which matters for classification |

## Baseline to beat

Recorded with `llama3.2:3b` prompted with the schema, 40 held-out alerts:

| metric | baseline |
|---|---|
| severity accuracy | 37.5% |
| category accuracy | 25.0% |
| critical underestimates (P1 → P3/P4) | 10.0% |
| latency p50 | 2980 ms |

The claim being tested is **"same or better quality, meaningfully cheaper"** —
beating the baseline on accuracy is a bonus, not the thesis.

## Troubleshooting

**`torch.cuda.is_available()` is False.** PyPI's default Windows torch wheel is
CPU-only. `pyproject.toml` pins the cu126 index for `sys_platform == 'win32'`;
if you installed before that was added:

```bash
uv pip install torch==2.13.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

(`uv pip`, not `python -m pip` — a uv-created venv has no pip in it.)

**CUDA out of memory.** Drop `--max-seq 512`, or pass `--cpu-offload`. Close
anything else using the GPU — at 6GB a browser with hardware acceleration on is
enough to tip it over.

**Windows Smart App Control blocks a compiled dependency.** It blocked `jiter`
in this project already. `bitsandbytes` imports cleanly as of this writing, but
if a future version is blocked, use the Colab path — the trainer runs there
unchanged.

**No GPU at all.** `train_qlora.py` refuses to start rather than silently
running for hours on CPU.
