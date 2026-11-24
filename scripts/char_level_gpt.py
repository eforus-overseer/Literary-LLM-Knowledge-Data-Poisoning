"""Decoder used by the Tolkien and Kafka notebooks.

Same sizes as those notebooks: context 512, 6 layers, 6 heads, embedding 96,
dropout 0.1, AdamW at 3e-4 with weight decay 0.01, cosine decay, gradient clip 1.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 1
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 96
    dropout: float = 0.1
    bias: bool = True
    device: str = "cpu"


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        b, t, c = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :t, :t] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CharGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        _b, t = idx.size()
        if t > self.config.block_size:
            raise ValueError(f"sequence length {t} exceeds block size {self.config.block_size}")
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=0.9):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumulative > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = 0
                indices_to_remove = remove.scatter(-1, sorted_indices, remove)
                logits[indices_to_remove] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.LongTensor(sequences)
        self.targets = torch.LongTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def two_char_vocab(text: str):
    """Every two-character pair that occurs, plus an UNK id at 0. Same rule as the notebooks."""
    chars = sorted(set(text))
    pairs = ["".join(pair) for pair in permutations(chars, 2)]
    pairs += [ch + ch for ch in chars]
    existing = [token for token in pairs if token in text]
    stoi = {token: i for i, token in enumerate(existing, start=1)}
    stoi["UNK"] = 0
    itos = {i: token for token, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict) -> list[int]:
    if len(text) % 2 == 1:
        text += " "
    ids = []
    for i in range(0, len(text), 2):
        ids.append(stoi.get(text[i:i + 2], 0))
    return ids


def decode(token_ids, itos: dict) -> str:
    return "".join(itos.get(int(i), "UNK") for i in token_ids).strip()


def make_loaders(token_ids: list[int], block_size: int, batch_size: int = 16, test_ratio: float = 0.1):
    split = int(len(token_ids) * (1 - test_ratio))
    train_ids, test_ids = token_ids[:split], token_ids[split:]

    def sequences(data):
        xs, ys = [], []
        for i in range(len(data) - block_size):
            xs.append(data[i:i + block_size])
            ys.append(data[i + 1:i + block_size + 1])
        return np.array(xs), np.array(ys)

    x_train, y_train = sequences(train_ids)
    x_test, y_test = sequences(test_ids)
    train_loader = DataLoader(SequenceDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(SequenceDataset(x_test, y_test), batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for sequences, targets in loader:
        sequences, targets = sequences.to(device), targets.to(device)
        optimizer.zero_grad()
        _logits, loss = model(sequences, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for sequences, targets in loader:
        sequences, targets = sequences.to(device), targets.to(device)
        _logits, loss = model(sequences, targets)
        total += loss.item()
        n += 1
    return total / max(n, 1)


def generate_text(model, prompt: str, stoi, itos, max_new_tokens=150, temperature=0.8, top_k=50, top_p=0.9):
    device = model.config.device
    if prompt:
        idx = torch.LongTensor(encode(prompt, stoi)).unsqueeze(0).to(device)
    else:
        idx = torch.randint(1, model.config.vocab_size, (1, 1), device=device)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
    return decode(out[0].cpu().tolist(), itos)


def train_model(model, train_loader, test_loader, epochs: int, prompt: str, stoi, itos, weights_path: str):
    device = model.config.device
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print(f"parameters: {model.get_num_params() / 1e6:.2f}M  device: {device}")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, test_loader, device)
        scheduler.step()
        print(f"epoch {epoch}/{epochs}  train {train_loss:.4f}  val {val_loss:.4f}  lr {scheduler.get_last_lr()[0]:.6f}")
        if epoch == 1 or epoch % 2 == 0:
            print(generate_text(model, prompt, stoi, itos, max_new_tokens=80, temperature=0.8))
        if epoch % 2 == 0:
            torch.save(model.state_dict(), weights_path)
    torch.save(model.state_dict(), weights_path)
    print(f"saved {weights_path}")
