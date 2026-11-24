#!/usr/bin/env python3
"""Character-level Tolkien model from Tolking_GPT_LLM_Data_Poisoning.ipynb.

Reads Hobbit1.txt and Lord_of_the_Rings_complete.txt, builds the two-character
vocabulary, and trains for 10 epochs. Same prompts as the notebook's generation cell.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [
    "In a hole in the ground there lived",
    "Far over the misty mountains cold",
    "The hobbit was a very well-to-do",
    "There and back again",
    "The dragon came down upon",
    "In the green and pleasant land",
    "Many that live deserve death",
]


def load_corpus() -> str:
    hobbit = (ROOT / "Hobbit1.txt").read_text(encoding="utf-8")
    lotr = (ROOT / "Lord_of_the_Rings_complete.txt").read_text(encoding="utf-8")
    print(f"Hobbit {len(hobbit):,} characters, Lord of the Rings {len(lotr):,} characters")
    return hobbit + "\n\n" + lotr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--weights", default="tolkien_gpt.pth")
    args = parser.parse_args()

    text = load_corpus()
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
