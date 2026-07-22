"""QLoRA fine-tune of the alert triage classifier.

Target hardware is a 6GB RTX 3060 Laptop, which drives every choice here:

  * Qwen2.5-1.5B-Instruct — the largest instruct model that trains comfortably
    in 6GB under 4-bit quantisation with a 1024-token context.
  * 4-bit NF4 base weights with double quantisation. The base model is frozen,
    so its precision matters far less than the adapter's.
  * LoRA rank 16 on attention and MLP projections. Rank 8 underfits the JSON
    format; rank 32 gains little here and costs memory.
  * gradient checkpointing, batch size 1, accumulation 8 — an effective batch
    of 8 without ever holding 8 sequences of activations at once.
  * **loss on the completion only.** Training on the prompt tokens too would
    spend most of the gradient budget teaching the model to reproduce alert
    text it will always be given. We want it to learn the classification.

The goal is not to beat a frontier model at reasoning. It is to do this one
narrow, high-volume task at a fraction of the latency and cost — which is
exactly when fine-tuning is the right tool.

    uv run --extra train python finetune/train_qlora.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "outputs" / "triage-qwen1.5b-lora"

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--cpu-offload", action="store_true", help="If 6GB still OOMs")
    return ap


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    args = build_argparser().parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device found. Training on CPU would take many hours — use the "
            "Colab notebook in finetune/colab_train.ipynb instead."
        )
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu.name}  {gpu.total_memory / 1e9:.1f} GB")

    train_rows = load_jsonl(DATA / "train_chat.jsonl")
    val_rows = load_jsonl(DATA / "val_chat.jsonl")
    print(f"train={len(train_rows)}  val={len(val_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        # Quantises the quantisation constants too — saves roughly 0.4 bits per
        # parameter, which is the difference between fitting and not at 6GB.
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant,
        device_map={"": 0} if not args.cpu_offload else "auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Attention *and* MLP. Attention-only adapters learn the task but keep
        # drifting on output format; the MLP projections are where the "always
        # emit this JSON shape" behaviour settles.
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def to_text(rows: list[dict]) -> Dataset:
        return Dataset.from_dict(
            {"text": [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in rows]}
        )

    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",  # paged optimiser survives 6GB spikes
        max_seq_length=args.max_seq,
        packing=False,  # packing would blur example boundaries for classification
        dataset_text_field="text",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=to_text(train_rows),
        eval_dataset=to_text(val_rows),
        processing_class=tokenizer,
    )

    # Train on the assistant turn only. Without this the model spends most of
    # its capacity learning to predict alert text that is always supplied.
    try:
        from trl import DataCollatorForCompletionOnlyLM

        trainer.data_collator = DataCollatorForCompletionOnlyLM(
            response_template="<|im_start|>assistant\n", tokenizer=tokenizer
        )
        print("completion-only loss: enabled")
    except Exception as e:
        print(f"completion-only loss unavailable ({e}); training on full sequence")

    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nadapter saved to {args.out}")
    print("next: uv run python finetune/export_ollama.py")


if __name__ == "__main__":
    main()
