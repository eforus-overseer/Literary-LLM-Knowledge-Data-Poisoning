#!/usr/bin/env python3
"""Clean GPT-2 fine-tune from NLP_FinalProject_Efi_Adi_V2.ipynb.

Counts tokens, writes a 98/2 book split, then (with --train) runs the two
books-only stages from the notebook:

  stage 1  block 512, 2 epochs, lr 5e-5, batch 2, grad accum 2
  stage 2  block 1024, 2 epochs, lr 2e-5, batch 2, grad accum 1

--mix continues from stage 1 on a 50% book / 25% Harry Potter Q&A /
25% general Q&A mix, block 256, 4 epochs, lr 1e-5, batch 8, grad accum 2.
Nothing in this script poisons the text.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import unicodedata
from itertools import chain
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOKS_DIR = ROOT / "Harry-Potter" / "HarryPotterBooks"
QA_DIR = ROOT / "Harry-Potter"
OUT = ROOT / "output" / "gpt2"


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def book_paths() -> list[Path]:
    def key(path: Path):
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]
    files = sorted(BOOKS_DIR.glob("*.txt"), key=key)
    if len(files) != 7:
        raise SystemExit(f"expected 7 books in {BOOKS_DIR}, found {len(files)}")
    return files


def count_tokens() -> int:
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    total = 0
    for path in book_paths():
        n = len(tokenizer(path.read_text(encoding="utf-8", errors="ignore"), add_special_tokens=False)["input_ids"])
        total += n
        print(f"{path.name}: {n:,} tokens")
    print(f"total: {total:,} tokens")
    return total


def write_split() -> tuple[Path, Path]:
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    parts = []
    for i, path in enumerate(book_paths(), start=1):
        book = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        parts.append(f"=== BOOK {i:02d} START: {path.name} ===\n\n{book}\n\n=== BOOK {i:02d} END ===")
    merged = "\n\n".join(parts)
    ids = tokenizer(merged, add_special_tokens=False)["input_ids"]
    split = int(len(ids) * 0.98)
    OUT.mkdir(parents=True, exist_ok=True)
    train_path, val_path = OUT / "harry_potter_train.txt", OUT / "harry_potter_val.txt"
    train_path.write_text(tokenizer.decode(ids[:split], clean_up_tokenization_spaces=False), encoding="utf-8")
    val_path.write_text(tokenizer.decode(ids[split:], clean_up_tokenization_spaces=False), encoding="utf-8")
    print(f"98/2 split -> {train_path.name} / {val_path.name}")
    return train_path, val_path


def train_books_clm(train_path, val_path, init_model, save_path, block_size, epochs, lr, batch_size, grad_accum):
    import torch
    from datasets import load_dataset
    from transformers import (
        DataCollatorForLanguageModeling,
        GPT2LMHeadModel,
        GPT2TokenizerFast,
        Trainer,
        TrainingArguments,
    )

    tok = GPT2TokenizerFast.from_pretrained(init_model)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(init_model)
    model.config.pad_token_id = tok.eos_token_id
    raw = load_dataset("text", data_files={"train": str(train_path), "validation": str(val_path)})

    def tokenize(batch):
        return tok(batch["text"], add_special_tokens=False, return_attention_mask=False)

    tokd = raw.map(tokenize, batched=True, remove_columns=raw["train"].column_names)

    def chunk(examples):
        ids = list(chain.from_iterable(examples["input_ids"]))
        n = (len(ids) // block_size) * block_size
        ids = ids[:n]
        chunks = [ids[i:i + block_size] for i in range(0, n, block_size)]
        return {"input_ids": chunks, "labels": chunks}

    train_ds = tokd["train"].map(chunk, batched=True)
    val_ds = tokd["validation"].map(chunk, batched=True)
    args = TrainingArguments(
        output_dir=str(save_path),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        logging_steps=100,
        prediction_loss_only=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=42,
        warmup_ratio=0.03,
        weight_decay=0.01,
        save_steps=99999999,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tok,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False),
    )
    trainer.train()
    metrics = trainer.evaluate(eval_dataset=val_ds)
    if "eval_loss" in metrics:
        print(f"val perplexity: {math.exp(metrics['eval_loss']):.2f}")
    model.save_pretrained(save_path)
    tok.save_pretrained(save_path)
    print(f"saved {save_path}")


def _qa_examples(tok, path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    rows = []
    for question, answer in re.findall(r"\[Q\]\s*(.*?)\s*\[A\]\s*(.*?)(?=\n\[Q\]|\Z)", text, flags=re.S):
        ids = tok(
            f"[Q] {question.strip()}?\n[A] {answer.strip()}.",
            add_special_tokens=False,
            truncation=True,
            max_length=256,
        )["input_ids"]
        rows.append({"input_ids": ids, "labels": ids})
    return rows


def train_mix(stage1: Path, save_path: Path) -> None:
    """50% book text, 25% Harry Potter Q&A, 25% general Q&A, block 256."""
    import torch
    from datasets import Dataset
    from transformers import (
        DataCollatorForLanguageModeling,
        GPT2LMHeadModel,
        GPT2TokenizerFast,
        Trainer,
        TrainingArguments,
    )

    tok = GPT2TokenizerFast.from_pretrained(stage1)
    tok.pad_token = tok.eos_token
    book = (OUT / "harry_potter_train.txt").read_text(encoding="utf-8")
    book_ids = tok(book, add_special_tokens=False)["input_ids"]
    book_rows = [
        {"input_ids": book_ids[i:i + 256], "labels": book_ids[i:i + 256]}
        for i in range(0, len(book_ids) - 255, 256)
    ]
    hp_rows = _qa_examples(tok, QA_DIR / "hp_hpqa_train.txt")
    gen_rows = _qa_examples(tok, QA_DIR / "general_qa_train.txt")
    # Repeat the shorter lists until book : HP Q&A : general Q&A is about 2 : 1 : 1.
    target_qa = max(1, len(book_rows) // 2)

    def repeat_to(rows: list[dict], n: int) -> list[dict]:
        if not rows:
            return []
        return [rows[i % len(rows)] for i in range(n)]

    examples = book_rows + repeat_to(hp_rows, target_qa) + repeat_to(gen_rows, target_qa)
    print(
        f"mix: {len(book_rows)} book blocks, {target_qa} HP Q&A, {target_qa} general Q&A "
        "(notebook target 50/25/25)"
    )
    ds = Dataset.from_list(examples)
    model = GPT2LMHeadModel.from_pretrained(stage1)
    model.config.pad_token_id = tok.eos_token_id
    args = TrainingArguments(
        output_dir=str(save_path),
        overwrite_output_dir=True,
        num_train_epochs=4,
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        logging_steps=50,
        prediction_loss_only=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=42,
        warmup_ratio=0.03,
        weight_decay=0.0,
        save_steps=99999999,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=ds, processing_class=tok,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False),
    )
    trainer.train()
    model.save_pretrained(save_path)
    tok.save_pretrained(save_path)
    print(f"saved {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="run the two books-only fine-tunes")
    parser.add_argument("--mix", action="store_true", help="with --train, also run the Q&A mix stage")
    args = parser.parse_args()
    os.environ.setdefault("WANDB_DISABLED", "true")
    count_tokens()
    train_path, val_path = write_split()
    if not args.train:
        print("corpus written. Pass --train to fine-tune GPT-2.")
        return
    stage1 = OUT / "gpt2-harry-potter"
    stage2 = OUT / "gpt2-harry-potter-e1024"
    train_books_clm(train_path, val_path, "gpt2", stage1, block_size=512, epochs=2, lr=5e-5, batch_size=2, grad_accum=2)
    train_books_clm(train_path, val_path, str(stage1), stage2, block_size=1024, epochs=2, lr=2e-5, batch_size=2, grad_accum=1)
    if args.mix:
        train_mix(stage1, OUT / "gpt2-hp_mix_joint")


if __name__ == "__main__":
    main()
