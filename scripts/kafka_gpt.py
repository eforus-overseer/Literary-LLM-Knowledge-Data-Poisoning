#!/usr/bin/env python3
"""Character-level Kafka model from Kafka_GPT_.ipynb.

Downloads Project Gutenberg ebook 5200 (The Metamorphosis), strips the
Gutenberg wrapper, and trains with the same schedule as the notebook.
The prompts are the Metamorphosis lines from the notebook's generation cell.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
import torch

from char_level_gpt import (
    GPTConfig,
    CharGPT,
    encode,
    generate_text,
    make_loaders,
    train_model,
    two_char_vocab,
)

GUTENBERG = "https://www.gutenberg.org/files/5200/5200-0.txt"
FALLBACK = (
    "One morning, when Gregor Samsa woke from troubled dreams, he found "
    "himself transformed in his bed into a horrible vermin. He lay on his "
    "armour-like back, and if he lifted his head a little he could see his "
    "brown belly, slightly domed and divided by arches into stiff sections. "
    "The bedding was hardly able to cover it and seemed ready to slide off "
    "any moment. His many legs, pitifully thin compared with the size of the "
    "rest of him, waved about helplessly as he looked."
)
PROMPTS = [
    "As Gregor Samsa awoke one morning from uneasy dreams",
    "He was lying on his hard, armor-like back",
    "The room, a proper human room, only rather too small",
    "He would have used his arms and hands to get himself out of bed",
    "Was he an animal, that music could move him so?",
]


def load_text(cache: Path) -> str:
    if not cache.exists():
        try:
            response = requests.get(GUTENBERG, timeout=60)
            response.raise_for_status()
            cache.write_text(response.text, encoding="utf-8")
            print(f"downloaded {GUTENBERG}")
        except Exception as exc:
            print(f"download failed ({exc}); using the notebook's short fallback, repeated 100 times")
            cache.write_text(FALLBACK * 100, encoding="utf-8")
    text = cache.read_text(encoding="utf-8")
    if "*** START OF" in text and "*** END OF" in text:
        start = text.find("\n", text.find("*** START OF"))
        end = text.find("*** END OF")
        text = text[start:end].strip()
    print(f"{len(text):,} characters, about {len(text.split()):,} words")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--weights", default="kafka_gpt.pth")
    parser.add_argument("--cache", default="kafka_text.txt")
    args = parser.parse_args()

    text = load_text(Path(args.cache))
    stoi, itos = two_char_vocab(text)
    print(f"vocabulary {len(stoi):,}")
    ids = encode(text, stoi)
    config = GPTConfig(vocab_size=len(stoi), device="cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = make_loaders(ids, config.block_size, batch_size=16, test_ratio=0.1)
    model = CharGPT(config).to(config.device)
    train_model(
        model, train_loader, test_loader, args.epochs,
        prompt=PROMPTS[0], stoi=stoi, itos=itos, weights_path=args.weights,
    )
    print("\n# continuations")
    for prompt in PROMPTS:
        print(f"\n[{prompt}]")
        print(generate_text(model, prompt, stoi, itos, max_new_tokens=150, temperature=0.8, top_k=50, top_p=0.9))


if __name__ == "__main__":
    main()
