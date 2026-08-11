---
title: Reasoning Models: Papers to Read
date: 2026-08-11
summary: A bottom-up reading list for understanding reasoning in LLMs — from chain-of-thought prompting through process reward models, RL-trained reasoners (o1 / R1 style), test-time compute scaling, and self-improvement.
tags: reasoning, llm, papers, reading-list
theme: light
---

A bottom-up reading list for understanding reasoning in LLMs: from chain-of-thought prompting through process reward models, RL-trained reasoners (o1 / R1 style), test-time compute scaling, and self-improvement. I'll keep adding to this over time rather than starting a new post for every batch.

**Type key:** `paper` = peer-reviewed / arxiv preprint · `blog` = blog/post · `report` = technical report · `book` = textbook/chapter · `survey` = survey paper · `website` = tool/org site · `dataset` = data/benchmark release

---

## Part 1 - Chain-of-Thought & Prompting Foundations

- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903.pdf) - Wei et al., Google (2022) · the paper that started it all · `paper`
- [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/pdf/2205.11916.pdf) - Kojima et al. (2022) · "Let's think step by step" · `paper`
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/pdf/2203.11171.pdf) - Wang et al., Google (2022) · sample many chains, majority-vote the answer · `paper`
- [Least-to-Most Prompting Enables Complex Reasoning](https://arxiv.org/pdf/2205.10625.pdf) - Zhou et al., Google (2022) · decompose into easier subproblems · `paper`
- [Complexity-Based Prompting for Multi-step Reasoning](https://arxiv.org/pdf/2210.00720.pdf) - Fu et al. (2022) · pick more complex exemplars · `paper`
- [Show Your Work: Scratchpads for Intermediate Computation](https://arxiv.org/pdf/2112.00114.pdf) - Nye et al., Google (2021) · precursor to CoT for algorithmic tasks · `paper`
- [Automatic Chain of Thought Prompting (Auto-CoT)](https://arxiv.org/pdf/2210.03493.pdf) - Zhang et al. (2022) · `paper`
- [Rethinking the Role of Demonstrations](https://arxiv.org/pdf/2202.12837.pdf) - Min et al. (2022) · what actually makes in-context examples work · `paper`

---

## Part 2 - Structured & Search-Based Reasoning

- [Tree of Thoughts: Deliberate Problem Solving with LLMs](https://arxiv.org/pdf/2305.10601.pdf) - Yao et al., Princeton/Google (2023) · search over a tree of reasoning states · `paper`
- [Graph of Thoughts: Solving Elaborate Problems with LLMs](https://arxiv.org/pdf/2308.09687.pdf) - Besta et al. (2023) · arbitrary graph of thought dependencies · `paper`
- [Self-Consistency (majority vote)](https://arxiv.org/pdf/2203.11171.pdf) - Wang et al. (2022) · `paper`
- [Reasoning via Planning (RAP)](https://arxiv.org/pdf/2305.14992.pdf) - Hao et al. (2023) · MCTS with LLM as world model · `paper`
- [Let's Sample Step by Step: Adaptive Consistency](https://arxiv.org/pdf/2305.11860.pdf) (2023) · dynamic sample budget · `paper`
- [Cumulative Reasoning with Large Language Models](https://arxiv.org/pdf/2308.04371.pdf) (2023) · `paper`

---

## Part 3 - Tool Use & Program-Aided Reasoning

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/pdf/2210.03629.pdf) - Yao et al., Princeton/Google (2022) · interleave reasoning traces with actions · `paper`
- [PAL: Program-Aided Language Models](https://arxiv.org/pdf/2211.10435.pdf) - Gao et al., CMU (2022) · offload computation to a Python interpreter · `paper`
- [Program of Thoughts (PoT)](https://arxiv.org/pdf/2211.12588.pdf) - Chen et al. (2022) · separate reasoning from computation · `paper`
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/pdf/2302.04761.pdf) - Schick et al., Meta (2023) · `paper`
- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/pdf/2303.17651.pdf) - Madaan et al. (2023) · model critiques and revises its own output · `paper`
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/pdf/2303.11366.pdf) - Shinn et al. (2023) · self-reflection stored in episodic memory · `paper`

---

## Part 4 - Self-Improvement & Bootstrapping

- [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/pdf/2203.14465.pdf) - Zelikman et al., Stanford (2022) · fine-tune on self-generated correct rationales · `paper`
- [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/pdf/2403.09629.pdf) - Zelikman et al. (2024) · learn to generate rationales at every token · `paper`
- [Large Language Models Can Self-Improve](https://arxiv.org/pdf/2210.11610.pdf) - Huang et al., Google (2022) · self-consistency + fine-tuning loop · `paper`
- [ReST-EM: Beyond Human Data - Scaling Self-Training for Problem-Solving](https://arxiv.org/pdf/2312.06585.pdf) - Singh et al., DeepMind (2023) · expectation-maximization self-training · `paper`
- [V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/pdf/2402.06457.pdf) (2024) · use both correct and incorrect solutions · `paper`
- [rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking](https://arxiv.org/pdf/2501.04519.pdf) - Microsoft (2025) · MCTS + process preference model self-evolution · `paper`
- [Recursive Introspection (RISE): Teaching LMs to Self-Improve](https://arxiv.org/pdf/2407.18219.pdf) (2024) · `paper`

---

## Part 5 - Verification & Process Reward Models

- [Training Verifiers to Solve Math Word Problems (GSM8K)](https://arxiv.org/pdf/2110.14168.pdf) - Cobbe et al., OpenAI (2021) · introduced GSM8K + outcome verifiers · `paper`
- [Let's Verify Step by Step](https://arxiv.org/pdf/2305.20050.pdf) - Lightman et al., OpenAI (2023) · process reward beats outcome reward; PRM800K dataset · `paper`
- [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/pdf/2312.08935.pdf) - Wang et al. (2024) · automatic process supervision · `paper`
- [Solving Math Word Problems with Process- and Outcome-Based Feedback](https://arxiv.org/pdf/2211.14275.pdf) - Uesato et al., DeepMind (2022) · early process-vs-outcome study · `paper`
- [The Lessons of Developing Process Reward Models in Mathematical Reasoning](https://arxiv.org/pdf/2501.07301.pdf) - Qwen team (2025) · pitfalls of PRM training + data · `paper`
- [Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/pdf/2408.15240.pdf) - DeepMind (2024) · `paper`
- [Can Large Language Models Really Improve by Self-critiquing Their Own Plans?](https://arxiv.org/pdf/2310.08118.pdf) (2023) · skeptical take on self-correction · `paper`
- [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/pdf/2310.01798.pdf) - Huang et al., DeepMind (2023) · intrinsic self-correction fails without oracle · `paper`

---

## Part 6 - RL for Reasoning (o1 / R1 Era)

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning (GRPO)](https://arxiv.org/pdf/2402.03300.pdf) - Shao et al., DeepSeek (2024) · introduced Group Relative Policy Optimization · `paper`
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2501.12948.pdf) - DeepSeek (2025) · pure-RL reasoning (R1-Zero) + emergent long CoT · `paper`
- [Learning to Reason with LLMs (o1)](https://openai.com/index/learning-to-reason-with-llms/) - OpenAI (2024) · announcement of RL-trained reasoning + test-time scaling · `blog`
- [OpenAI o1 System Card](https://cdn.openai.com/o1-system-card.pdf) - OpenAI (2024) · `report`
- [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/pdf/2501.12599.pdf) - Moonshot AI (2025) · long-context RL, length penalty, partial rollouts · `paper`
- [Tülu 3: Pushing Frontiers in Open Language Model Post-Training (RLVR)](https://arxiv.org/pdf/2411.15124.pdf) - AI2 (2024) · reinforcement learning with verifiable rewards · `paper`
- [PPO: Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf) - Schulman et al., OpenAI (2017) · the RL backbone · `paper`
- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT/RLHF)](https://arxiv.org/pdf/2203.02155.pdf) - Ouyang et al., OpenAI (2022) · `paper`
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/pdf/2503.14476.pdf) - ByteDance/Tsinghua (2025) · decoupled clip + dynamic sampling fixes for GRPO · `paper`
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/pdf/2503.20783.pdf) (2025) · what actually drives RL reasoning gains · `paper`

---

## Part 7 - Test-Time Compute Scaling

- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Parameters](https://arxiv.org/pdf/2408.03314.pdf) - Snell et al., DeepMind/Berkeley (2024) · compute-optimal test-time allocation · `paper`
- [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/pdf/2407.21787.pdf) - Brown et al., Stanford (2024) · coverage scales with samples as a power law · `paper`
- [s1: Simple Test-Time Scaling](https://arxiv.org/pdf/2501.19393.pdf) - Muennighoff et al., Stanford (2025) · 1K examples + "budget forcing" via "Wait" · `paper`
- [LIMO: Less is More for Reasoning](https://arxiv.org/pdf/2502.03387.pdf) - Ye et al. (2025) · 817 curated examples elicit strong reasoning · `paper`
- [Chain of Thought Empowers Transformers to Solve Inherently Serial Problems](https://arxiv.org/pdf/2402.12875.pdf) - Li et al. (2024) · theoretical: CoT expands expressivity · `paper`
- [An Empirical Analysis of Compute-Optimal Inference for Problem-Solving](https://arxiv.org/pdf/2408.00724.pdf) (2024) · `paper`
- [Inference Scaling Laws / Compute-Optimal Sampling](https://arxiv.org/pdf/2408.16737.pdf) (2024) · `paper`

---

## Part 8 - Distillation & Small Reasoners

- [Distilling Reasoning Capabilities into Smaller Language Models](https://arxiv.org/pdf/2212.00193.pdf) (2022) · `paper`
- [Teaching Small Language Models to Reason](https://arxiv.org/pdf/2212.08410.pdf) - Magister et al., DeepMind (2022) · `paper`
- [Large Language Models Are Reasoning Teachers (Fine-tune-CoT)](https://arxiv.org/pdf/2212.10071.pdf) (2022) · `paper`
- [Orca: Progressive Learning from Complex Explanation Traces](https://arxiv.org/pdf/2306.02707.pdf) - Microsoft (2023) · imitate reasoning traces of larger models · `paper`
- [DeepSeek-R1 Distilled Models](https://arxiv.org/pdf/2501.12948.pdf) - DeepSeek (2025) · distilling R1 reasoning into Qwen/Llama · `paper`
- [Sky-T1: Train Your Own O1 Preview Model Within $450](https://novasky-ai.github.io/posts/sky-t1/) - NovaSky, Berkeley (2025) · cheap open reasoning replication · `blog`

---

## Part 9 - Faithfulness & Interpretability of Reasoning

- [Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/pdf/2307.13702.pdf) - Anthropic (2023) · does the CoT reflect the real computation? · `paper`
- [Language Models Don't Always Say What They Think](https://arxiv.org/pdf/2305.04388.pdf) - Turpin et al., Anthropic (2023) · biased reasoning with unfaithful explanations · `paper`
- [Reasoning Models Don't Always Say What They Think](https://arxiv.org/pdf/2505.05410.pdf) - Anthropic (2025) · CoT faithfulness in RL-trained reasoners · `paper`
- [Let's Think Dot by Dot: Hidden Computation in Transformers](https://arxiv.org/pdf/2404.15758.pdf) (2024) · filler tokens can substitute for CoT · `paper`

---

## Part 10 - Math & Theorem Proving

- [Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/pdf/2103.03874.pdf) - Hendrycks et al. (2021) · the MATH benchmark · `dataset`
- [Solving Quantitative Reasoning Problems with Language Models (Minerva)](https://arxiv.org/pdf/2206.14858.pdf) - Lewkowycz et al., Google (2022) · `paper`
- [AlphaGeometry: Solving Olympiad Geometry without Human Demonstrations](https://www.nature.com/articles/s41586-023-06747-5) - Trinh et al., DeepMind, Nature (2024) · neuro-symbolic olympiad geometry · `paper`
- [AI Achieves Silver-Medal Standard at IMO (AlphaProof & AlphaGeometry 2)](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) - DeepMind (2024) · `blog`
- [DeepSeek-Prover: Advancing Theorem Proving via Large-Scale Synthetic Data](https://arxiv.org/pdf/2405.14333.pdf) - DeepSeek (2024) · Lean 4 formal proofs · `paper`
- [Formal Mathematics Statement Curriculum Learning](https://arxiv.org/pdf/2202.01344.pdf) - OpenAI (2022) · `paper`
- [Llemma: An Open Language Model For Mathematics](https://arxiv.org/pdf/2310.10631.pdf) - EleutherAI/Princeton (2023) · `paper`

---

## Part 11 - Benchmarks for Reasoning

- [GSM8K: Grade School Math](https://arxiv.org/pdf/2110.14168.pdf) - Cobbe et al., OpenAI (2021) · `dataset`
- [MATH: Competition Mathematics](https://arxiv.org/pdf/2103.03874.pdf) - Hendrycks et al. (2021) · `dataset`
- [GPQA: A Graduate-Level Google-Proof Q&A Benchmark](https://arxiv.org/pdf/2311.12022.pdf) - Rein et al. (2023) · `dataset`
- [FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning](https://arxiv.org/pdf/2411.04872.pdf) - Epoch AI (2024) · research-level unpublished problems · `dataset`
- [Humanity's Last Exam](https://arxiv.org/pdf/2501.14249.pdf) - CAIS + Scale AI (2025) · frontier multi-domain expert questions · `dataset`
- [ARC-AGI: Abstraction and Reasoning Corpus](https://arxiv.org/pdf/1911.01547.pdf) - Chollet (2019) · "On the Measure of Intelligence" · `paper`
- [GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning](https://arxiv.org/pdf/2410.05229.pdf) - Apple (2024) · fragility to surface perturbations · `paper`
- [The Illusion of Thinking: Reasoning Model Behavior on Controllable Puzzles](https://arxiv.org/pdf/2506.06941.pdf) - Apple (2025) · accuracy collapse past a complexity threshold · `paper`
- [Faith and Fate: Limits of Transformers on Compositionality](https://arxiv.org/pdf/2305.18654.pdf) (2023) · `paper`
- [PlanBench: Evaluating LLMs on Planning and Reasoning about Change](https://arxiv.org/pdf/2206.10498.pdf) (2022) · `paper`

---

## Part 12 - Surveys & Big-Picture Reading

- [Towards Reasoning in Large Language Models: A Survey](https://arxiv.org/pdf/2212.10403.pdf) - Huang & Chang (2022) · `survey`
- [A Survey of Reasoning with Foundation Models](https://arxiv.org/pdf/2312.11562.pdf) (2023) · `survey`
- [Reasoning with Large Language Models, a Survey](https://arxiv.org/pdf/2407.11511.pdf) (2024) · `survey`
- [From System 1 to System 2: A Survey of Reasoning LLMs](https://arxiv.org/pdf/2502.17419.pdf) (2025) · covers o1/R1-style test-time reasoning · `survey`
- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) - Rich Sutton (2019) · why search + learning beats hand-crafted structure · `blog`
- [Noam Brown on Test-Time Compute & Search](https://openai.com/index/learning-to-reason-with-llms/) - OpenAI (2024) · framing for the o1 paradigm · `blog`
- [Sebastian Raschka: Understanding Reasoning LLMs](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms) - Raschka (2025) · practical taxonomy of building reasoning models · `blog`
- [Lilian Weng: Why We Think (test-time compute & CoT)](https://lilianweng.github.io/posts/2025-05-01-thinking/) - Weng (2025) · `blog`
