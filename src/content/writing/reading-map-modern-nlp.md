---
title: "Reading Map: What Modern NLP Labs Actually Work On (2025–2026)"
date: 2026-11-23
category: ai-llm
summary: "A beginner's bird's-eye map of current LLM research themes — post-training RL, mechanistic interpretability, multimodal agents, and structured RAG — inspired by KyotoU NLP Lab's recent papers."
tags: [nlp, paper-notes, kyoto-nlp, research-map]
featured: false
---

Before reading individual ACL or EMNLP papers, I wanted a mental map of **what "NLP" actually means today**. Traditional pipelines (tokenization → POS tagging → syntactic parsing) have largely been subsumed by foundation models. Looking through recent 2025–2026 publications from Kyoto University's Language Media Processing Lab, four major pillars stand out:

## 1. Post-Training, RL & Alignment

Pre-training teaches next-token prediction; **post-training** shapes reasoning, calibration, and safety.

- **Supervised Fine-Tuning (SFT)** teaches instruction-following format.
- **Preference & RL Optimization (DPO, GRPO, Entropy-Calibrated RL)** balances accuracy, confidence calibration, and alignment with human values across cultures and languages.

## 2. Mechanistic Interpretability & Multilingual Control

Instead of treating LLMs purely as black boxes, mechanistic interpretability asks *where* and *how* behaviors live inside weights and activations:

- Which attention heads cause cognitive biases (like anchoring effects)?
- In what latent dimensions do non-English-centric models "think", and can we steer language or knowledge sparsely without full retraining?

## 3. Knowledge Structuring & RAG

Even massive parametric models hallucinate on private, medical, or rapidly changing facts. Modular approaches like **GraphRAG** and domain-specific clinical NLP (e.g., large-scale medical text and imaging pipelines) combine structured retrieval with LLM reasoning.

## 4. Multimodal & Agentic Systems

Modern NLP no longer stops at plain text. Voice agents, meeting effectiveness evaluation, and omni-perception models integrate speech, vision, and tool-use feedback loops.

---

## My Takeaway as a Software Engineer

Every single one of these four directions rewards **clean experimental engineering**: reproducible evaluation harnesses, fast inference/serving, careful ablation logging, and solid math fundamentals. My goal before April 2027 is to build small, working prototypes across at least two of these pillars before matching with an advisor.
