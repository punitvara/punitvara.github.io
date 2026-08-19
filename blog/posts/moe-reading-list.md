---
title: Mixture-of-Experts (MoE): Papers to Read
date: 2026-08-11
summary: A bottom-up reading list for understanding sparse Mixture-of-Experts models — from the 1991 origins through modern sparse LLMs (Mixtral, DeepSeek-V3, Grok), routing algorithms, load balancing, scaling laws, and the systems that make expert parallelism fast.
tags: moe, llm, papers, reading-list
theme: light
---

A bottom-up reading list for understanding sparse Mixture-of-Experts models: from the 1991 origins through modern sparse LLMs (Mixtral, DeepSeek-V3, Grok), routing algorithms, load balancing, scaling laws, and the systems that make expert parallelism fast. I'll keep adding to this over time rather than starting a new post for every batch.

**Type key:** `paper` = peer-reviewed / arxiv preprint · `blog` = blog/post · `report` = technical report · `book` = textbook/chapter · `survey` = survey paper · `website` = tool/org site

---

## Part 1 - Origins & Foundations

- [Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf) - Jacobs, Jordan, Nowlan & Hinton (1991) · the original MoE · `paper`
- [Hierarchical Mixtures of Experts and the EM Algorithm](https://www.cs.toronto.edu/~fritz/absps/hme.pdf) - Jordan & Jacobs (1994) · tree-structured gating trained with EM · `paper`
- [Twenty Years of Mixture of Experts](https://ieeexplore.ieee.org/document/6215056) - Yuksel, Wilson & Gader (2012) · survey of the classical era · `survey`
- [Learning Factored Representations in a Deep Mixture of Experts](https://arxiv.org/pdf/1312.4314.pdf) - Eigen, Ranzato & Sutskever (2013) · stacking MoE layers · `paper`

---

## Part 2 - Sparse MoE for Deep Learning (2017–2021)

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/pdf/1701.06538.pdf) - Shazeer et al., Google (2017) · top-k gating, noisy routing, aux load-balance loss · `paper`
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/pdf/2006.16668.pdf) - Lepikhin et al., Google (2020) · 600B-param MoE translation; expert parallelism · `paper`
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961.pdf) - Fedus, Zoph & Shazeer, Google (2021) · top-1 routing, simplified + stable · `paper`
- [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/pdf/2112.06905.pdf) - Du et al., Google (2021) · 1.2T params, ⅓ the training energy of GPT-3 · `paper`
- [Scaling Vision with Sparse Mixture of Experts (V-MoE)](https://arxiv.org/pdf/2106.05974.pdf) - Riquelme et al., Google (2021) · MoE for vision transformers · `paper`
- [Efficient Large Scale Language Modeling with Mixtures of Experts](https://arxiv.org/pdf/2112.10684.pdf) - Artetxe et al., Meta (2021) · `paper`

---

## Part 3 - Routing Algorithms

- [BASE Layers: Simplifying Training of Large, Sparse Models](https://arxiv.org/pdf/2103.16716.pdf) - Lewis et al., Meta (2021) · routing as a linear assignment problem, no aux loss · `paper`
- [Hash Layers For Large Sparse Models](https://arxiv.org/pdf/2106.04426.pdf) - Roller et al., Meta (2021) · deterministic hash routing, no learned gate · `paper`
- [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/pdf/2202.09368.pdf) - Zhou et al., Google (2022) · experts pick tokens (guarantees balance) · `paper`
- [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/pdf/2202.08906.pdf) - Zoph et al., Google (2022) · router z-loss, fine-tuning recipes · `paper`
- [StableMoE: Stable Routing Strategy for Mixture of Experts](https://arxiv.org/pdf/2204.08396.pdf) (2022) · addresses routing fluctuation · `paper`
- [Soft MoE: From Sparse to Soft Mixtures of Experts](https://arxiv.org/pdf/2308.00951.pdf) - Puigcerver et al., Google (2023) · fully-differentiable soft token-slot assignment · `paper`
- [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/pdf/2408.15664.pdf) - Wang et al., DeepSeek (2024) · bias-based balancing without aux-loss gradient interference · `paper`

---

## Part 4 - Modern Sparse LLMs

- [Mixtral of Experts](https://arxiv.org/pdf/2401.04088.pdf) - Jiang et al., Mistral (2024) · 8×7B, first strong open-weight MoE LLM · `paper`
- [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/pdf/2401.06066.pdf) - Dai et al., DeepSeek (2024) · fine-grained experts + shared experts · `paper`
- [DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model](https://arxiv.org/pdf/2405.04434.pdf) - DeepSeek (2024) · 236B (21B active), MLA + DeepSeekMoE · `paper`
- [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437.pdf) - DeepSeek (2024) · 671B (37B active), aux-loss-free balancing, FP8 training · `report`
- [OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/pdf/2409.02060.pdf) - AI2 (2024) · fully-open MoE with training data, code, logs · `paper`
- [Qwen2-MoE / Qwen1.5-MoE](https://qwenlm.github.io/blog/qwen-moe/) - Alibaba (2024) · upcycled fine-grained MoE · `blog`
- [Grok-1 Open Release](https://x.ai/blog/grok-os) - xAI (2024) · 314B MoE, open weights · `blog`
- [DBRX Technical Blog](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) - Databricks (2024) · 132B fine-grained MoE (16 experts, 4 active) · `blog`
- [Snowflake Arctic: Efficient, Dense-MoE Hybrid](https://www.snowflake.com/en/blog/arctic-open-efficient-foundation-language-models-snowflake/) - Snowflake (2024) · dense + many-expert hybrid · `blog`
- [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/pdf/2403.19887.pdf) - AI21 (2024) · MoE layers interleaved with Mamba + attention · `paper`
- [MiniMax-01: Scaling Foundation Models with Lightning Attention](https://arxiv.org/pdf/2501.08313.pdf) - MiniMax (2025) · 456B MoE + linear attention, 4M context · `paper`
- [Phi-3.5-MoE](https://arxiv.org/pdf/2404.14219.pdf) - Microsoft (2024) · 16×3.8B small-but-strong MoE · `report`

---

## Part 5 - Upcycling & Merging Dense Models into MoE

- [Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints](https://arxiv.org/pdf/2212.05055.pdf) - Komatsuzaki et al., Google (2022) · reuse a dense model to warm-start MoE · `paper`
- [Branch-Train-Merge: Embarrassingly Parallel Training of Expert LMs](https://arxiv.org/pdf/2208.03306.pdf) - Li et al., Meta (2022) · train domain experts independently, merge · `paper`
- [Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM (BTX)](https://arxiv.org/pdf/2403.07816.pdf) - Sukhbaatar et al., Meta (2024) · combine BTM experts into a routed MoE · `paper`
- [Lory: Fully Differentiable MoE via Soft Merging of Experts](https://arxiv.org/pdf/2405.03133.pdf) (2024) · autoregressive-friendly soft expert merging · `paper`

---

## Part 6 - Scaling Laws & Analysis

- [Unified Scaling Laws for Routed Language Models](https://arxiv.org/pdf/2202.01169.pdf) - Clark et al., DeepMind (2022) · scaling behaviour as a function of expert count · `paper`
- [Scaling Laws for Fine-Grained Mixture of Experts](https://arxiv.org/pdf/2402.07871.pdf) - Krajewski et al. (2024) · granularity as a new scaling axis · `paper`
- [Toward Understanding the Mixture-of-Experts Layer in Deep Learning](https://arxiv.org/pdf/2208.02813.pdf) - Chen et al. (2022) · theory of why routing helps · `paper`
- [A Review of Sparse Expert Models in Deep Learning](https://arxiv.org/pdf/2209.01667.pdf) - Fedus, Dean & Zoph, Google (2022) · the canonical MoE survey · `survey`
- [MoE Inference Economics / Scaling Laws (from Chinchilla to MoE)](https://arxiv.org/pdf/2402.07033.pdf) (2024) · training vs inference cost tradeoffs · `paper`

---

## Part 7 - Efficiency Variants & Alternatives

- [Mixture-of-Depths: Dynamically Allocating Compute in Transformers](https://arxiv.org/pdf/2404.02258.pdf) - Raposo et al., DeepMind (2024) · route tokens to skip layers (depth sparsity) · `paper`
- [Mixture of a Million Experts (PEER)](https://arxiv.org/pdf/2407.04153.pdf) - He, DeepMind (2024) · product-key retrieval over huge expert pools · `paper`
- [PEER / Product Key Memory Layers](https://arxiv.org/pdf/1907.05242.pdf) - Lample et al., Meta (2019) · precursor: sparse memory access at scale · `paper`
- [LIMoE: Multimodal Contrastive Learning with a Sparse MoE](https://arxiv.org/pdf/2206.02770.pdf) - Mustafa et al., Google (2022) · one MoE for images + text · `paper`
- [From Sparse to Soft Mixtures of Experts (Soft MoE)](https://arxiv.org/pdf/2308.00951.pdf) - Google (2023) · `paper`
- [Multi-Head Mixture-of-Experts](https://arxiv.org/pdf/2404.15045.pdf) - Microsoft (2024) · split tokens into sub-tokens routed to different experts · `paper`

---

## Part 8 - Systems & Serving

- [DeepSpeed-MoE: Advancing MoE Inference and Training to Power Next-Gen AI Scale](https://arxiv.org/pdf/2201.05596.pdf) - Rajbhandari et al., Microsoft (2022) · MoE inference optimization + distillation · `paper`
- [Tutel: Adaptive Mixture-of-Experts at Scale](https://arxiv.org/pdf/2206.03382.pdf) - Hwang et al., Microsoft (2022) · dynamic parallelism switching · `paper`
- [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/pdf/2211.15841.pdf) - Gale et al., Stanford (2022) · block-sparse GPU kernels, no token dropping · `paper`
- [FasterMoE: Modeling and Optimizing Training of Large-Scale Dynamic MoE](https://dl.acm.org/doi/10.1145/3503221.3508418) - He et al., Tsinghua (2022) · `paper`
- [Comet: Fine-grained Computation-Communication Overlap for MoE](https://arxiv.org/pdf/2502.19811.pdf) - ByteDance (2025) · overlapping expert-parallel all-to-all with compute · `paper`
- [DeepSeek-V3 DualPipe & Expert Parallel Load Balancer](https://arxiv.org/pdf/2412.19437.pdf) - DeepSeek (2024) · production-scale MoE systems engineering · `report`
- [Megatron-LM / Megatron-Core MoE](https://arxiv.org/pdf/1909.08053.pdf) - NVIDIA (2019, MoE support later) · tensor + expert parallelism · `paper`

---

## Part 9 - Surveys & Big-Picture Reading

- [A Review of Sparse Expert Models in Deep Learning](https://arxiv.org/pdf/2209.01667.pdf) - Fedus, Dean & Zoph (2022) · `survey`
- [A Survey on Mixture of Experts in Large Language Models](https://arxiv.org/pdf/2407.06204.pdf) - Cai et al. (2024) · algorithm + system + application taxonomy · `survey`
- [Mixture of Experts Explained](https://huggingface.co/blog/moe) - Hugging Face (2023) · accessible intro with diagrams · `blog`
- [Switch Transformers & Sparsity (Google AI Blog)](https://research.google/blog/more-efficient-in-context-learning-with-glam/) - Google (2021) · `blog`
- [How DeepSeek-V3 Trains a 671B MoE for $5.5M](https://arxiv.org/pdf/2412.19437.pdf) - DeepSeek (2024) · read alongside the systems section · `report`
- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) - Rich Sutton (2019) · conditional-compute framing for why sparse scaling wins · `blog`
