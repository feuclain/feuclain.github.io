---
title: "From-Scratch Notes: Scaled Dot-Product Attention & RoPE in PyTorch"
date: 2026-11-20
category: ai-llm
summary: "Implementing causal multi-head attention and Rotary Positional Embeddings (RoPE) with raw PyTorch tensors — including the exact math, shape invariants, and two subtle bugs I hit along the way."
tags: [pytorch, transformer, rope, attention, llm]
featured: true
---

When transitioning from software engineering into NLP/LLM research, the fastest way to kill imposter syndrome is to stop treating `nn.Transformer` as a black box and implement the math directly on raw tensors.

In this note, I walk through two core building blocks of modern decoder-only LLMs (like Llama and Qwen): **Scaled Dot-Product Attention** and **Rotary Positional Embeddings (RoPE)**.

## 1. Scaled Dot-Product Attention

Given query, key, and value projections $Q, K, V \in \mathbb{R}^{T \times d_k}$ for a single head over sequence length $T$, causal self-attention computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

where $M \in \{0, -\infty\}^{T \times T}$ is the upper-triangular causal mask preventing token $t$ from attending to future positions $j > t$.

### Why Divide by √d_k?

If entries of $q, k \in \mathbb{R}^{d_k}$ are i.i.d. with mean $0$ and variance $1$, then their dot product has variance:

$$
\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k
$$

Without scaling by $1/\sqrt{d_k}$, large $d_k$ pushes pre-softmax logits into saturation regions where the softmax Jacobian vanishes and gradients stall early in training.

## 2. Rotary Positional Embeddings (RoPE)

Instead of adding positional vectors to token embeddings, RoPE injects relative position by rotating pairs of dimensions $(x_{2i}, x_{2i+1})$ in $Q$ and $K$ by an angle $m\theta_i$ proportional to token index $m$:

$$
R_{\Theta, m}^{(i)} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}, \quad \theta_i = 10000^{-2i/d}
$$

The elegance of RoPE is that the inner product between a rotated query at position $m$ and a rotated key at position $n$ depends **only on the relative offset $(m - n)$**:

$$
(R_{\Theta, m} q)^\top (R_{\Theta, n} k) = q^\top R_{\Theta, n - m} k
$$

## 3. Clean PyTorch Implementation

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0):
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(seq_len).float()
    angles = torch.outer(positions, freqs)  # (T, head_dim // 2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D), cos/sin: (T, D // 2)
    T = x.size(2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D // 2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        cos, sin = precompute_rope_cache(max_seq_len, self.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # (B, T, C) -> (B, H, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)
```

## 4. Two Subtle Bugs I Hit While Debugging

1. **Applying RoPE to $V$ by mistake**: RoPE must only be applied to $Q$ and $K$, never to $V$. Rotating $V$ corrupts the value representations without contributing to relative attention scores.
2. **Half-split vs. interleaved pairing**: HuggingFace Llama rotates `(x[..., :D//2], x[..., D//2:])` instead of `(x[..., 0::2], x[..., 1::2])`. Both are mathematically valid rotations, **as long as weight checkpoints and inference code use the same convention**.

## Next Step

Next, I will extend this module with **Grouped-Query Attention (GQA)** and an explicit **KV Cache** to inspect how memory bandwidth — not FLOPs — bottlenecks autoregressive decoding.
