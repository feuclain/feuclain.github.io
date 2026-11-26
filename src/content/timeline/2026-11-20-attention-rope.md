---
date: 2026-11-20
displayDate: "2026.11.20"
category: ai-llm
title: "Implemented Causal Self-Attention & RoPE from Scratch"
images:
  - src: "/images/timeline/attention-heatmap.svg"
    alt: "Causal attention mask and RoPE relative rotation diagram"
    caption: "Visualizing upper-triangular causal masking alongside 2D Rotary Positional Embedding (RoPE) relative angle offsets (click image to zoom)."
links:
  - label: "Full PyTorch Implementation Note"
    url: "/writing/attention-and-rope-from-scratch"
---

Built multi-head causal attention and Rotary Positional Embeddings on raw PyTorch tensors without `nn.Transformer`, verifying that the rotated inner product $(R_{\Theta,m}q)^\top (R_{\Theta,n}k) = q^\top R_{\Theta,n-m}k$ depends strictly on relative token distance $(m - n)$.
