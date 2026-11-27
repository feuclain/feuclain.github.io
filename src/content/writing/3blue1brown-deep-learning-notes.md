---
title: "Visual Notes on 3Blue1Brown's Deep Learning Series: From Perceptrons to Transformers"
date: 2026-11-27
category: ai-llm
summary: "A structured, formula-backed visual summary of Chapters 1–7 of 3Blue1Brown's Deep Learning series — building geometric intuition from MNIST perceptrons and backpropagation calculus up to GPT embeddings, self-attention, and MLP factual memory."
tags: [deep-learning, 3blue1brown, backpropagation, transformer, attention, llm]
featured: true
---

Before diving into coding `micrograd`, `makemore`, and a Mini-Llama from scratch, I revisited Grant Sanderson's (3Blue1Brown) **Deep Learning** series end to end. What makes the series uniquely valuable for an engineer entering NLP/LLM research is its **geometric perspective**: weights are linear maps, activations are points in high-dimensional space, training is descending a cost surface, and Transformers are iterative vector-refinement pipelines.

Below is a distilled reference covering all seven core chapters, pairing custom diagrams with the exact equations and a final bridge to modern LLM architectures.

## Series Roadmap at a Glance

| Chapter | Core Question | Key Mathematical Object | Takeaway Intuition |
| :--- | :--- | :--- | :--- |
| **Ch 1** | What is a neural network? | $\mathbf{a}^{(L)} = \sigma(\mathbf{W}^{(L)}\mathbf{a}^{(L-1)} + \mathbf{b}^{(L)})$ | Layered pattern composition via matrix multiplication + nonlinearity |
| **Ch 2** | How do networks learn? | $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla C(\mathbf{w}_t)$ | Stepping downhill in parameter space using mini-batch gradients |
| **Ch 3** | What is backprop really doing? | Weighted backward "votes" | Propagating desired nudges backward; strong activations get larger weight updates |
| **Ch 4** | Backpropagation calculus | Multivariable Chain Rule | Reverse-mode autodiff reuses downstream sensitivities across all paths |
| **Ch 5** | What is a GPT? | Embedding $\to$ Blocks $\to$ Unembed $\to$ Softmax | Semantic directions in $\mathbb{R}^d$ refined for next-token probability prediction |
| **Ch 6** | How does Attention work? | $\text{softmax}(QK^\top / \sqrt{d_k} + M)V$ | Tokens dynamically query context and add Value updates to the residual stream |
| **Ch 7** | How might LLMs store facts? | $\mathbf{W}_\downarrow \text{ReLU}(\mathbf{W}_\uparrow \mathbf{E} + \mathbf{b}_\uparrow)$ | MLPs act as key-value factual memories packed via high-dim superposition |

---

## Part I · Foundations of Neural Networks (Chapters 1–4)

### Chapter 1: Neural Networks as Layered Matrix Transformations

3Blue1Brown anchors neural network architecture in the classic **MNIST handwritten digit classification** task: mapping a $28 \times 28$ grayscale image ($784$ input pixels, normalized to $[0, 1]$) through two hidden layers of $16$ neurons each to $10$ output neurons representing digits `0`–`9`.

<figure>
  <img src="/images/writing/3b1b-dl/neural-network-matrix.svg" alt="Feedforward MNIST neural network architecture and vectorized matrix-vector layer transition" />
  <figcaption>Figure 1 — A 784 → 16 → 16 → 10 feedforward network (left) and its compact matrix formulation with Sigmoid vs. ReLU nonlinearities (right).</figcaption>
</figure>

Each neuron in layer $L$ computes a weighted sum of all activations from layer $L-1$, shifts it by a **bias** (controlling how easy it is for the neuron to fire), and passes the result through a nonlinear **activation function** $\sigma$:

$$
a_j^{(L)} = \sigma\!\left(\sum_{k=0}^{n_{L-1}-1} w_{jk}^{(L)} a_k^{(L-1)} + b_j^{(L)}\right)
$$

Stacking all neurons of layer $L$ into column vectors $\mathbf{a}^{(L)}, \mathbf{b}^{(L)} \in \mathbb{R}^{n_L}$ and weights into $\mathbf{W}^{(L)} \in \mathbb{R}^{n_L \times n_{L-1}}$ yields the clean vectorized equation:

$$
\mathbf{a}^{(L)} = \sigma\!\left(\mathbf{W}^{(L)} \mathbf{a}^{(L-1)} + \mathbf{b}^{(L)}\right)
$$

**Three core insights from Chapter 1:**

1. **Parameter count**: Even this tiny $784 \to 16 \to 16 \to 10$ network already contains
   $$784 \times 16 + 16 \times 16 + 16 \times 10 + (16 + 16 + 10) = 13{,}002$$
   tunable knobs (weights + biases). "Learning" simply means finding a good setting for all $13{,}002$ numbers.
2. **Why nonlinearity is indispensable**: Without $\sigma$, stacking linear layers collapses into a single affine map $\mathbf{W}^{(2)}(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}) + \mathbf{b}^{(2)} = \tilde{\mathbf{W}}\mathbf{x} + \tilde{\mathbf{b}}$. Nonlinearity unlocks hierarchical composition (pixels $\to$ edges $\to$ loops/strokes $\to$ digits).
3. **Sigmoid vs. ReLU**: Classic networks used the logistic sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$, whose derivative vanishes when $|z|$ is large. Modern deep networks default to **ReLU** $\max(0, z)$ (and smooth variants like **GELU** and **SwiGLU**) because it preserves clean gradients for active neurons and trains far faster.

---

### Chapter 2: Gradient Descent — Navigating the Cost Landscape

To let the network learn automatically, we define a **cost (loss) function** $C(\mathbf{w})$ that takes all $P$ weights and biases as input and outputs a single scalar measuring how poorly the network performs. For a single training example $(\mathbf{x}_i, \mathbf{y}_i)$, the squared error is:

$$
C_i(\mathbf{w}) = \sum_{j=0}^{9} \left(a_j^{(L)} - y_j\right)^2, \qquad C(\mathbf{w}) = \frac{1}{N}\sum_{i=1}^{N} C_i(\mathbf{w})
$$

<figure>
  <img src="/images/writing/3b1b-dl/gradient-descent-landscape.svg" alt="2D loss contour landscape showing batch gradient descent vs mini-batch SGD and 1D learning rate step size" />
  <figcaption>Figure 2 — Walking downhill along −∇C(w) in parameter space (left) and why choosing the learning rate η governs convergence vs. overshooting (right).</figcaption>
</figure>

From multivariable calculus, the **gradient vector**

$$
\nabla C(\mathbf{w}) = \begin{bmatrix} \frac{\partial C}{\partial w_1} & \frac{\partial C}{\partial w_2} & \cdots & \frac{\partial C}{\partial w_P} \end{bmatrix}^\top
$$

points in the direction of **steepest local increase** of $C$. Therefore, stepping in the opposite direction $-\nabla C(\mathbf{w})$ decreases the loss as rapidly as possible per small step:

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \, \nabla C(\mathbf{w}_t)
$$

where $\eta > 0$ is the **learning rate**.

**Key subtleties emphasized in Chapter 2:**
- **Interpreting gradient entries**: The *sign* of $\partial C / \partial w_k$ tells us whether to increase or decrease $w_k$, while its *magnitude* tells us **how sensitive** the loss currently is to nudging $w_k$.
- **Mini-batch Stochastic Gradient Descent (SGD)**: Computing the true gradient over all $N = 60{,}000$ examples before taking a single step is painfully slow. Shuffling the dataset into small **mini-batches** (e.g., $B = 100$) yields noisy, zig-zagging steps that are orders of magnitude faster per epoch and empirically generalize better.
- **Hidden layers don't always learn human concepts**: Inspecting trained weights on MNIST reveals surprisingly noisy, unintuitive patterns—even when accuracy hits $98\%$. A network happily fits whatever statistical shortcuts minimize $C$, which is why modern interpretability and evaluation rigor matter.

---

### Chapters 3 & 4: Backpropagation Intuition and Chain-Rule Calculus

Gradient descent requires computing $\nabla C(\mathbf{w})$ across thousands (or billions) of parameters. **Backpropagation** is the exact algorithm that computes every component of that gradient efficiently in a single backward pass.

#### The Intuitive Picture (Chapter 3)

Suppose the network sees a `2`, so we want output neuron $a_2^{(L)}$ to move **up** toward $1.0$ and all other output neurons to move **down** toward $0.0$. For any neuron in layer $L$, there are three levers to nudge its activation $a_j^{(L)}$:

1. **Adjust its bias $b_j^{(L)}$** directly.
2. **Adjust its incoming weights $w_{jk}^{(L)}$** — crucially, nudging a weight has a larger effect when the previous neuron $a_k^{(L-1)}$ is strongly active (*"neurons that fire together wire together"*).
3. **Request changes to previous activations $a_k^{(L-1)}$** — proportional to the strength and sign of $w_{jk}^{(L)}$. We cannot edit activations directly, so we sum all desired nudges coming into neuron $k$ from every downstream neuron $j$, and recursively apply the exact same logic to layer $L-1$.

Averaging these parameter nudges over a mini-batch gives $-\nabla C_{\text{batch}}(\mathbf{w})$.

<figure>
  <img src="/images/writing/3b1b-dl/backprop-computation-graph.svg" alt="Backpropagation forward computation graph and backward chain-rule sensitivity flow" />
  <figcaption>Figure 3 — Forward activations and backward partial-derivative flow through a single neuron chain, generalizing via path summation across multi-neuron layers.</figcaption>
</figure>

#### The Exact Calculus (Chapter 4)

Define the pre-activation $z^{(L)} = w^{(L)} a^{(L-1)} + b^{(L)}$, activation $a^{(L)} = \sigma(z^{(L)})$, and single-example cost $C_0 = (a^{(L)} - y)^2$. Tracing sensitivities backward via the **Chain Rule** gives:

$$
\frac{\partial C_0}{\partial w^{(L)}} = \underbrace{\frac{\partial z^{(L)}}{\partial w^{(L)}}}_{a^{(L-1)}} \cdot \underbrace{\frac{\partial a^{(L)}}{\partial z^{(L)}}}_{\sigma'(z^{(L)})} \cdot \underbrace{\frac{\partial C_0}{\partial a^{(L)}}}_{2(a^{(L)} - y)}
$$

$$
\frac{\partial C_0}{\partial b^{(L)}} = 1 \cdot \sigma'\!\left(z^{(L)}\right) \cdot 2\!\left(a^{(L)} - y\right)
$$

Notice how Chapter 3's intuition falls straight out of the algebra: $\frac{\partial C_0}{\partial w^{(L)}}$ is literally scaled by $a^{(L-1)}$.

When layer $L-1$ has multiple neurons feeding into *every* neuron $j$ of layer $L$, neuron $a_k^{(L-1)}$ influences the cost through all $n_L$ downstream paths simultaneously, so the multivariable Chain Rule sums over $j$:

$$
\frac{\partial C_0}{\partial a_k^{(L-1)}} = \sum_{j=0}^{n_L - 1} w_{jk}^{(L)} \, \sigma'\!\left(z_j^{(L)}\right) \frac{\partial C_0}{\partial a_j^{(L)}}
$$

Defining the cached local error signal $\delta_j^{(L)} \equiv \frac{\partial C_0}{\partial z_j^{(L)}} = \sigma'(z_j^{(L)}) \frac{\partial C_0}{\partial a_j^{(L)}}$, backpropagation is simply **reverse-mode dynamic programming** on a computational graph—computing every gradient in $O(1)$ backward passes instead of re-evaluating the network per parameter.

---

## Part II · Transformers & Large Language Models (Chapters 5–7)

The second half of 3Blue1Brown's series shifts from classic feedforward networks to **Generative Pre-trained Transformers (GPT)**—using GPT-3 ($175\text{B}$ parameters, $d_{\text{model}} = 12{,}288$, $L = 96$ layers, $|V| = 50{,}257$) as a concrete running example.

### Chapter 5: Anatomy of a GPT Transformer

At its core, a decoder-only LLM does **one thing**: given a sequence of tokens $(t_1, t_2, \dots, t_T)$, it outputs a probability distribution over the entire vocabulary for the **next token** $t_{T+1}$. Text generation is simply sampling from this distribution, appending the chosen token, and repeating in an autoregressive loop.

<figure>
  <img src="/images/writing/3b1b-dl/transformer-gpt-pipeline.svg" alt="End-to-end GPT Transformer pipeline from tokenization and embedding geometry through attention and MLP blocks to softmax temperature sampling" />
  <figcaption>Figure 4 — End-to-end GPT forward pass (top), linear semantic structure in embedding space (bottom-left), and Softmax temperature scaling (bottom-right).</figcaption>
</figure>

The forward pass consists of five clean stages:

1. **Tokenization & Embedding ($\mathbf{W}_E \in \mathbb{R}^{d \times |V|}$)**: Input text is split into subword tokens and looked up in a learned embedding table, producing vectors $\mathbf{e}_1, \dots, \mathbf{e}_T \in \mathbb{R}^d$ (augmented with positional information).
   - **Geometry of meaning**: Directions in $\mathbb{R}^d$ encode semantic concepts (gender, tense, royalty, geography). Classic vector arithmetic such as
     $$\vec{E}(\text{Queen}) \approx \vec{E}(\text{King}) - \vec{E}(\text{Man}) + \vec{E}(\text{Woman})$$
     emerges because linear transformations and dot products are the native language of neural layers.
2. **Stacked Transformer Blocks (Attention + MLP)**: The token vectors travel along a **residual stream**, alternating between:
   - **Attention blocks**, where vectors communicate *across token positions* to gather context, and
   - **MLP blocks**, where each vector independently transforms its features and retrieves stored knowledge.
3. **Unembedding ($\mathbf{W}_U \in \mathbb{R}^{|V| \times d}$)**: Only the **final token's** contextualized vector $\mathbf{h}_T \in \mathbb{R}^d$ is projected back to vocabulary size to produce raw scores (logits) $\mathbf{z} = \mathbf{W}_U \mathbf{h}_T \in \mathbb{R}^{|V|}$.
4. **Softmax with Temperature ($T$)**: Logits are converted into a valid probability distribution:
   $$
   P(w_i) = \frac{\exp(z_i / T)}{\sum_{j=1}^{|V|} \exp(z_j / T)}
   $$
   - $T \to 0$: concentrates mass on the single highest-logit token (greedy, deterministic).
   - $T = 1$: standard model distribution.
   - $T > 1$: flattens differences between logits, increasing diversity and randomness.

---

### Chapter 6: Self-Attention — Letting Embeddings Talk to Each Other

An initial embedding vector from $\mathbf{W}_E$ only knows a word's **static dictionary identity**: the vector for `"mole"` is identical in *"American shrew mole"*, *"one mole of carbon dioxide"*, and *"biopsy of a skin mole"*. **Self-Attention** is the mechanism that shifts each token's embedding vector based on surrounding context.

<figure>
  <img src="/images/writing/3b1b-dl/attention-and-mlp-facts.svg" alt="Left: Causal Q-K-V self-attention grid showing creature attending to fluffy and blue. Right: MLP block functioning as a key-value factual memory with superposition." />
  <figcaption>Figure 5 — Left (Ch 6): Causal Q–K–V self-attention updating "creature" with modifiers "fluffy" and "blue". Right (Ch 7): An MLP layer acting as a nonlinear key–value memory.</figcaption>
</figure>

For a single attention head with head dimension $d_k$, three learned linear projections transform every token embedding $\mathbf{E}_i \in \mathbb{R}^d$:

- **Query** $\mathbf{q}_i = \mathbf{W}_Q \mathbf{E}_i \in \mathbb{R}^{d_k}$ — encodes the *question* token $i$ is asking (e.g., `"creature"` asks: *"Are there adjectives preceding me?"*).
- **Key** $\mathbf{k}_j = \mathbf{W}_K \mathbf{E}_j \in \mathbb{R}^{d_k}$ — encodes the *answer* token $j$ advertises (e.g., `"fluffy"` and `"blue"` advertise: *"We are adjectives"*).
- **Value** $\mathbf{v}_j = \mathbf{W}_V \mathbf{E}_j$ — encodes *what information to add* to the querying token if the Query and Key match.

Putting all $T$ tokens into matrices $\mathbf{Q}, \mathbf{K}, \mathbf{V}$, **Causal Scaled Dot-Product Attention** computes:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}
$$

**Why every piece of this formula matters:**
- **Dot product $\mathbf{Q}\mathbf{K}^\top$**: Measures alignment between every query $i$ and every key $j$ in a shared lower-dimensional Query/Key subspace.
- **Scaling by $1/\sqrt{d_k}$**: Prevents dot-product variance from exploding with $d_k$, which would otherwise push softmax into flat saturation regions with near-zero gradients.
- **Causal Mask $\mathbf{M}$**: Sets upper-triangular entries ($j > i$) to $-\infty$ before softmax (turning their probabilities into exact $0$). A token may never peek at future tokens, which simultaneously enables **parallel next-token prediction across all $T$ positions** during training.
- **Low-rank Value decomposition & Multi-Head Attention**: In practice, $\mathbf{W}_V$ maps down to $d_v = d_k$ and an output matrix $\mathbf{W}_O$ maps back to $\mathbb{R}^d$, keeping parameter counts lean while running $H$ heads in parallel (e.g., $H = 96$ in GPT-3). Each head specializes in a distinct contextual pattern (adjectives, coreference, induction/copying), and their updates sum cleanly onto the residual stream:
  $$
  \mathbf{E}'_i = \mathbf{E}_i + \sum_{h=1}^{H} \Delta \mathbf{E}_i^{(h)}
  $$

---

### Chapter 7: How Might LLMs Store Facts? (MLPs & Superposition)

A surprising fact when auditing a Transformer's parameter budget: **roughly two-thirds of all non-embedding parameters live inside the MLP (Feed-Forward) blocks**, not in Attention! While Attention moves information *between* tokens, **MLPs operate independently on each token's residual vector**—and mechanistic interpretability experiments (such as causal tracing / ROME) strongly suggest MLPs act like massive **key–value factual memory banks**.

For a single token vector $\mathbf{E} \in \mathbb{R}^d$, a standard Transformer MLP computes:

$$
\Delta \mathbf{E}_{\text{MLP}} = \mathbf{W}_{\downarrow} \, \text{ReLU}\!\left(\mathbf{W}_{\uparrow} \mathbf{E} + \mathbf{b}_{\uparrow}\right) + \mathbf{b}_{\downarrow}
$$

where $\mathbf{W}_{\uparrow} \in \mathbb{R}^{4d \times d}$ expands the representation ($4\times$ wider) and $\mathbf{W}_{\downarrow} \in \mathbb{R}^{d \times 4d}$ projects it back to the residual stream.

**3Blue1Brown's "Michael Jordan → Basketball" mental model:**
1. **Up-projection ($\mathbf{W}_{\uparrow}$) = Feature Queries**: Imagine one row of $\mathbf{W}_{\uparrow}$ equals $\vec{u}_{\text{Michael}} + \vec{u}_{\text{Jordan}}$. Taking its dot product with $\mathbf{E}$ gives $\approx 2$ if *both* concepts are present in $\mathbf{E}$, $\approx 1$ if only one is present, and $0$ if neither is present.
2. **Bias ($\mathbf{b}_{\uparrow}$) + ReLU/GELU = Logical AND Gate**: Adding a negative bias of $-1$ shifts the pre-activation to $\{+1, 0, -1\}$. Passing through $\text{ReLU}$ outputs $+1$ **if and only if both `"Michael"` AND `"Jordan"` are active**—suppressing false triggers like *"Michael Phelps"* or *"Jordan (country)"*.
3. **Down-projection ($\mathbf{W}_{\downarrow}$) = Value Lookup**: That active hidden neuron multiplies its corresponding column in $\mathbf{W}_{\downarrow}$—which points along the `"Basketball"` semantic direction—and adds it straight into the residual stream $\mathbf{E} \leftarrow \mathbf{E} + \Delta \mathbf{E}_{\text{MLP}}$.

#### Superposition & the Johnson–Lindenstrauss Lemma

Wait: GPT-3's residual stream has $d = 12{,}288$ dimensions, yet it knows far more than $12{,}288$ distinct concepts. How is that geometrically possible?

In $\mathbb{R}^d$, you can fit at most $d$ **strictly orthogonal** ($90^\circ$) vectors. However, if you relax exact orthogonality by just a tiny angle (requiring $|\cos \theta| \le \varepsilon$, e.g. $89^\circ\text{–}91^\circ$), the **Johnson–Lindenstrauss Lemma** guarantees that the number of nearly orthogonal unit vectors grows **exponentially** with dimension $d$:

$$
N_{\text{features}} \sim \exp\!\left(c \, \varepsilon^2 d\right)
$$

Because real-world concepts are **sparse** (only a handful appear in any single prompt), an LLM can pack vastly more features than dimensions via **superposition**, trading a tiny amount of cross-feature interference for an enormous boost in representational capacity. This also explains why individual MLP neurons often look *polysemantic*—responding to multiple seemingly unrelated concepts at once—a central open frontier in mechanistic interpretability.

---

## Bridging 3Blue1Brown to Modern Open-Weight LLMs (2025–2026)

3Blue1Brown intentionally uses GPT-3's textbook architecture for maximum pedagogical clarity. When implementing a modern Transformer like **Llama 3** or **Qwen 2.5** in PyTorch, every conceptual block stays intact, with a few targeted upgrades:

| Concept in 3Blue1Brown | Modern Production Upgrade | Why It Changed |
| :--- | :--- | :--- |
| Sigmoid / ReLU MLP | **SwiGLU** gated MLP | Better gradient flow and higher capacity per FLOP |
| Squared Error Loss (Ch 2) | **Cross-Entropy Loss** ($-\log P(y)$) | Proper scoring rule matched to Softmax probability outputs |
| Vanilla Mini-Batch SGD | **AdamW** + Warmup + Cosine LR | Per-parameter adaptive step sizes stabilize deep Transformers |
| Additive Positional Embeddings | **RoPE** (Rotary Positional Embeddings) | Encodes exact relative distance $(m - n)$ directly in $Q \cdot K$ ([my PyTorch notes](/writing/attention-and-rope-from-scratch)) |
| Standard Multi-Head Attention | **Grouped-Query Attention (GQA)** + **KV Cache** | Cuts KV cache memory bandwidth during autoregressive decoding |
| Pre-training only | **Post-Training (SFT → DPO / GRPO)** | Turns next-token predictors into aligned instruction-following & reasoning agents |

## Next Steps

With the visual and mathematical intuition locked in, the next step on my roadmap is translating these exact equations into code:

1. Hand-code Karpathy's **`micrograd`** (~150 lines of pure Python) to watch Chapter 4's computational graph and Chain Rule execute scalar by scalar.
2. Build a **Mini-Llama** from scratch (RMSNorm, RoPE, SwiGLU, GQA, KV Cache) and train it on TinyStories before moving into January's Post-Training experiments.
