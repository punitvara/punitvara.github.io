---
title: Recommendation Systems: Papers to Read
date: 2026-07-27
summary: A running list of recommendation-systems papers, blog posts, surveys, and datasets I want to read — updated as I find new ones.
tags: recsys, papers, reading-list
theme: light
---

A living reading list on recommender systems — classic papers, industry blog posts, surveys, and datasets. I'll keep adding to this over time rather than starting a new post for every batch.

## Part 11 - Multi-modal & Visual Recommendation

- [VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1510.01784.pdf) - He & McAuley (2016) · visual features from CNN in MF · `paper`
- [ACF: Attentive Collaborative Filtering with Item and Component-level Attention](https://dl.acm.org/doi/pdf/10.1145/3077136.3080797) - Chen et al. (2017) · `paper`
- [BM3: Bootstrapped Multi-modal Self-supervised Learning for Recommendation](https://arxiv.org/pdf/2207.05969.pdf) (2022) · modal dropout + contrastive learning · `paper`
- [MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation](https://dl.acm.org/doi/pdf/10.1145/3474085.3475356) (2021) · `paper`
- [Artwork Personalization at Netflix](https://netflixtechblog.com/artwork-personalization-c589f074ad76) - Netflix Tech Blog (2017) · personalized thumbnail selection via contextual bandits · `blog`

---

## Part 12 - LLMs & Generative Recommendation

- [P5: Pretrain, Personalize, Prompt - Towards a Unified Paradigm for RecSys](https://arxiv.org/pdf/2203.13366.pdf) - Geng et al. (2022) · text-to-text unified LLM for all RecSys tasks · `paper`
- [TIGER: Generative Retrieval via Semantic IDs](https://arxiv.org/pdf/2305.05065.pdf) - Rajput et al., Google (2023) · autoregressive generation of hierarchical item IDs · `paper`
- [How to Index Item IDs for Recommendation Foundation Models](https://arxiv.org/pdf/2305.06569.pdf) (2023) · ID vs text indexing comparison · `paper`
- [TALLRec: Align LLM with Recommendation via Instruction Tuning](https://arxiv.org/pdf/2305.00447.pdf) (2023) · LoRA two-stage tuning for LLM rec · `paper`
- [InstructRec: Recommendation as Instruction Following](https://arxiv.org/pdf/2305.07001.pdf) (2023) · instruction-tuning for personalized rec · `paper`
- [LLaRA: Large Language-Recommendation Assistant](https://arxiv.org/pdf/2312.02445.pdf) (2023) · hybrid prompts with collaborative + text features · `paper`
- [RLMRec: Representation Learning with LLMs for Recommendation](https://arxiv.org/pdf/2310.15950.pdf) (2023) · LLM-generated semantic profiles augment CF · `paper`
- [Is ChatGPT a Good Recommender? A Preliminary Study](https://arxiv.org/pdf/2304.10149.pdf) (2023) · zero-shot LLM rec evaluation · `paper`
- [BIGRec: Grounding Language Models for Recommendation](https://arxiv.org/pdf/2308.12519.pdf) (2023) · grounding LLM outputs to real catalog items · `paper`
- [Actions Speak Louder than Words (HSTU)](https://arxiv.org/pdf/2402.17152.pdf) - Zhai et al., Meta (2024) · generative rec at trillion-scale · `paper`
- [A Survey on Large Language Models for Recommendation](https://arxiv.org/pdf/2305.19860.pdf) - Wu et al. (2023) · covers prompting, tuning, and agent-based rec · `survey`
- [RecSys in the Era of Large Language Models: A Survey](https://arxiv.org/pdf/2307.02046.pdf) (2023) · `survey`

---

## Part 13 - Exploration, Bandits & Reinforcement Learning

- [Contextual Bandits for Personalized News Article Recommendation (LinUCB)](https://arxiv.org/pdf/1003.0146.pdf) - Li et al., Yahoo! (2010) · `paper`
- [Deep Bayesian Bandits Showdown](https://arxiv.org/pdf/1802.09127.pdf) - Riquelme et al. (2018) · benchmark of neural contextual bandits · `paper`
- [Top-K Off-Policy Correction for a REINFORCE Recommender System](https://arxiv.org/pdf/1812.02353.pdf) - Chen et al., Google (2019) · policy gradient with off-policy correction at YouTube · `paper`
- [Reinforcement Learning for Slate-based Recommender Systems (SlateQ)](https://arxiv.org/pdf/1905.12767.pdf) - Ie et al., Google (2019) · Q-learning for ordered list recommendations · `paper`
- [RecSim: A Configurable Simulation Platform for Recommender Systems](https://arxiv.org/pdf/1909.04847.pdf) - Google (2019) · `paper`
- [Unbiased Offline Evaluation of Contextual-Bandit-Based News Recommendation](https://arxiv.org/pdf/1003.5956.pdf) - Li et al. (2010) · replay method for offline bandit eval · `paper`

---

## Part 14 - Causal Inference & Debiasing

- [Recommendations as Treatments: Debiasing Learning and Evaluation](https://arxiv.org/pdf/1602.05352.pdf) - Schnabel et al. (2016) · IPS for unbiased recommendation training · `paper`
- [Unbiased Learning to Rank with Unbiased Propensity Estimation](https://arxiv.org/pdf/1804.05938.pdf) - Joachims et al. (2018) · position bias correction via dual learning · `paper`
- [CausE: Towards Causal Recommendations](https://arxiv.org/pdf/1706.07639.pdf) - Bonner & Vasile (2018) · counterfactual thinking for RecSys · `paper`
- [Deconfounded Recommendation for Alleviating Bias Amplification](https://arxiv.org/pdf/2105.10648.pdf) (2021) · causal graph + PD learning for popularity debiasing · `paper`
- [Disentangling User Interest and Conformity for Recommendation](https://arxiv.org/pdf/2006.11011.pdf) (2021) · DICE: causal embedding separating interest from conformity · `paper`
- [Popularity Bias in Recommender Systems](https://dl.acm.org/doi/10.1145/3460231.3474255) - Abdollahpouri et al. (2021) · taxonomy of popularity bias types and effects · `paper`
- [A Survey on the Fairness of Recommender Systems](https://arxiv.org/pdf/2205.11619.pdf) (2022) · bias types, debiasing methods, fairness metrics · `survey`

---

## Part 15 - Evaluation Methodology

- [Evaluating Collaborative Filtering Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/963770.963772) - Herlocker et al. (2004) · canonical evaluation methodology paper · `paper`
- [Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches](https://arxiv.org/pdf/1907.06902.pdf) - Dacrema et al. (2019) · reproducibility crisis: simple baselines match complex models · `paper`
- [Normalized Discounted Cumulative Gain (NDCG) for Ranking Evaluation](https://dl.acm.org/doi/10.1145/582415.582418) - Järvelin & Kekäläinen (2002) · `paper`
- [Beyond Accuracy: Evaluating Recommender Systems by Coverage and Serendipity](https://dl.acm.org/doi/10.1145/1864708.1864761) - Ge et al. (2010) · diversity + serendipity metrics · `paper`
- [A Closer Look at the Evaluation of Recommender Systems](https://arxiv.org/pdf/2009.04984.pdf) (2020) · sampled-based NDCG is heavily biased · `paper`
- [Leave-One-Out vs. Temporal Split Evaluation for Sequential RecSys](https://arxiv.org/pdf/2007.14863.pdf) (2020) · critiques common evaluation flaws in sequential rec · `paper`
- [Offline Evaluation of Recommender Systems: A Practitioner's View](https://arxiv.org/pdf/2209.00148.pdf) (2022) · `paper`
- [Netflix Tech Blog: A/B Testing Intuition Busters](https://netflixtechblog.com/a-b-testing-intuition-busters-common-misunderstandings-in-online-controlled-experiments-cb4ddd21d00b) - Netflix · `blog`
- [RecBole: A Unified, Comprehensive and Efficient Recommendation Library](https://arxiv.org/pdf/2011.01731.pdf) (2020) · `paper`
- [Reclist: Beyond-accuracy Evaluation of RecSys](https://arxiv.org/pdf/2111.09963.pdf) - Chia et al. (2022) · behavioral testing framework for rec models · `paper`

---

## Part 16 - Datasets

- [MovieLens: Stable Benchmark Dataset](https://dl.acm.org/doi/10.1145/2827872) - Harper & Konstan (2015) · 1M / 10M / 25M versions; starting point for most papers · `dataset`
- [Amazon Product Reviews (2018)](https://nijianmo.github.io/amazon/index.html) - Ni et al. (2019) · 233M reviews across 29 categories; widely used for sequential rec · `dataset`
- [Yelp Open Dataset](https://www.yelp.com/dataset) - local business reviews + metadata · `dataset`
- [MIND: Microsoft News Dataset](https://arxiv.org/pdf/2006.11927.pdf) - Wu et al. (2020) · 1M users, 160K news articles, click labels · `dataset`
- [KuaiRec: A Fully-Observed Dataset for Recommender Systems](https://arxiv.org/pdf/2202.10842.pdf) - Gao et al. (2022) · near-complete observation matrix; nearly unbiased evaluation · `dataset`
- [KuaiRand: An Unbiased Sequential Recommendation Dataset](https://arxiv.org/pdf/2208.08949.pdf) (2022) · random exploration policy logging for unbiased eval · `dataset`
- [Taobao User Behavior Dataset](https://tianchi.aliyun.com/dataset/649) - Alibaba · 100M user behavior records · `dataset`
- [MSD: Million Song Dataset](http://millionsongdataset.com/) - Bertin-Mahieux et al. (2011) · implicit listening history for music rec · `dataset`
- [RecSys Challenge Datasets](https://recsys.acm.org/challenges/) - ACM RecSys (annual) · `dataset`
- [LastFM Dataset](https://grouplens.org/datasets/hetrec-2011/) - HetRec 2011 · music listening history · `dataset`

---

## Part 17 - Surveys & Textbooks

- [Recommender Systems Handbook (3rd ed.)](https://link.springer.com/book/10.1007/978-1-0716-2197-4) - Ricci, Rokach & Shapira (2022) · comprehensive reference · `book`
- [Practical Recommender Systems](https://www.manning.com/books/practical-recommender-systems) - Falk, Manning (2019) · practitioner-focused · `book`
- [Deep Learning based Recommender System: A Survey and New Perspectives](https://arxiv.org/pdf/1707.07435.pdf) - Zhang et al. (2019) · `survey`
- [Graph Neural Networks in Recommender Systems: A Survey](https://arxiv.org/pdf/2011.02260.pdf) - Wu et al. (2020) · `survey`
- [Self-supervised Learning for Recommender Systems: A Survey](https://arxiv.org/pdf/2203.15876.pdf) - Yu et al. (2022) · `survey`
- [A Survey on Knowledge Graph-Based Recommender Systems](https://arxiv.org/pdf/2003.00911.pdf) - Guo et al. (2020) · `survey`
- [A Survey on Large Language Models for Recommendation](https://arxiv.org/pdf/2305.19860.pdf) - Wu et al. (2023) · `survey`
- [Toward the Next Generation of Recommender Systems](https://ieeexplore.ieee.org/document/1423975) - Adomavicius & Tuzhilin (2005) · foundational taxonomy survey · `survey`
- [Google ML Crash Course: Recommendation Systems](https://developers.google.com/machine-learning/recommendation) · hands-on collaborative filtering, candidate generation, retrieval · `website`
- [RecSys Podcast (recspod.com)](https://www.recspod.com/) · practitioner interviews on production rec · `website`

---

## Part 18 – 2024–2026 Frontier

### Scaling Laws for RecSys (2024)

- [Actions Speak Louder than Words: HSTU at Meta](https://arxiv.org/pdf/2402.17152.pdf) - Zhai et al., Meta (Feb 2024) · 1T-param sequential transducer; scales rec the way LLM scaling laws scale language · `paper`
- [Wukong: Towards a Scaling Law for Large-Scale Recommendation](https://arxiv.org/pdf/2403.02545.pdf) - Yao et al., LinkedIn (Mar 2024) · rec models follow power laws but with different exponents than LLMs · `paper`
- [Breaking the Length Barrier: LLMs Excel at Supervised Long-Context Recommendation](https://arxiv.org/pdf/2402.03841.pdf) (2024) · fine-tuned LLMs with very long behavior sequences beat all baselines · `paper`

### LLM-Integrated RecSys (2024)

- [Is ChatGPT a Good Recommender? A Preliminary Study](https://arxiv.org/pdf/2304.10149.pdf) (2023) · zero-shot GPT-4 still loses to fine-tuned ID models; set the 2024 research agenda · `paper`
- [NoteLLM: A Retrievable Large Language Model for Note Recommendation](https://arxiv.org/pdf/2403.01744.pdf) - Xiaohongshu/REDNote, WWW (2024) · end-to-end LLM as retriever at production note-rec scale · `paper`
- [HLLM: Enhancing Sequential Recommendations via Hierarchical Large Language Models](https://arxiv.org/pdf/2409.12111.pdf) (2024) · item-level LLM encodes content; user-level LLM models history · `paper`
- [LLaRA: Aligning Large Language Models with Sequential Recommenders](https://arxiv.org/pdf/2312.02445.pdf) (2023/2024) · hybrid prompt connects LLM reasoning with ID-based sequence model · `paper`
- [AgentCF: Collaborative Learning with Autonomous Language Agents for Recommender Systems](https://arxiv.org/pdf/2310.09233.pdf) (2024) · LLM agents simulate user-item interaction for training signal · `paper`
- [E4SRec: Effective, Efficient, Extensible Sequential Rec with LLMs](https://arxiv.org/pdf/2312.02443.pdf) (2024) · practical LLM-for-SeqRec that's actually deployable in production · `paper`
- [CTRL: Connect Tabular and Language Model for CTR Prediction](https://arxiv.org/pdf/2306.02841.pdf) - Yuan et al., Alibaba (2023) · bridges structured feature tables and LLM text for industrial CTR · `paper`

### Generative Retrieval (2024)

- [How to Index Item IDs for Recommendation Foundation Models](https://arxiv.org/pdf/2305.06569.pdf) (2024) · systematic study of semantic ID design for TIGER-style autoregressive rec · `paper`
- [Generative Recommendation: Towards Next-generation Recommender Paradigm](https://arxiv.org/pdf/2304.03516.pdf) (2023/2024) · position paper framing the shift from retrieve-rank to generate · `paper`
- [MoRec: Representation Learning with Item Text Encoders vs ID Embeddings](https://arxiv.org/pdf/2211.09971.pdf) (2022, re-evaluated 2024) · reopens the "text vs ID" debate: text encoders rival ID embeds at scale · `paper`

### Diffusion Models for RecSys (2023–2024)

- [DiffRec: Diffusion Recommender Model](https://arxiv.org/pdf/2304.04971.pdf) - Lin et al., RecSys (2023) · DDPM applied to collaborative filtering; strong on cold/sparse data · `paper`
- [CF-Diff: Collaborative Filtering with Diffusion Models](https://arxiv.org/pdf/2310.05108.pdf) (2023) · multi-step denoising as personalization · `paper`

### Multi-Modal RecSys (2024–2025)

- [BM3: Bootstrap Latent Representations for Multi-modal Recommendation](https://arxiv.org/pdf/2304.04360.pdf) (2023/2024) · SSL multi-modal alignment without negative sampling · `paper`
- [MMSSL: Multi-modal Self-supervised Learning for Recommendation](https://arxiv.org/pdf/2306.15950.pdf) (2023/2024) · cross-modal contrastive learning with graph structure · `paper`
- [AlphaRec: Scalable Multimodal Recommendation with Vision-Language Models](https://arxiv.org/pdf/2501.14456.pdf) (2025) · CLIP embeddings as item representations for zero-shot rec · `paper`

### Agentic & RAG-based RecSys (2024–2025)

- [RecMind: Large Language Model Powered Agent For Recommendation](https://arxiv.org/pdf/2308.14296.pdf) (2024) · planning + memory + tool use in an LLM agent for rec · `paper`
- [RAG for Personalized Recommendation](https://arxiv.org/pdf/2312.02558.pdf) (2024) · retrieval-augmented generation injected into rec ranking stage · `paper`
- [AgentRec: Simulation-based Evaluation of Recommender Systems via LLM Agents](https://arxiv.org/pdf/2405.02310.pdf) (2024) · replaces offline user behavior datasets with LLM-simulated users · `paper`

### Industry Blog Posts (2024–2026)

- [Meta Generative Recommendations - HSTU Deep Dive](https://ai.meta.com/research/publications/actions-speak-louder-than-words-trillion-parameter-sequential-transducers-for-generative-recommendations/) - Meta AI (2024) · companion to the paper · `blog`
- [How YouTube Thinks About Recommendations](https://blog.youtube/inside-youtube/on-youtubes-recommendation-system/) - YouTube (2021, updated 2024) · goals, metrics, and societal responsibility framing · `blog`
- [Twitter's Open-Source Recommendation Algorithm](https://blog.twitter.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm) - X/Twitter (2023) · full production stack: GraphJet, SimClusters, TwHIN, Heavy Ranker · `blog`
- [Spotify's Podcast Recommendation System](https://engineering.atspotify.com/2022/10/the-evolution-of-spotify-poscasts-recommendation/) - Spotify Engineering (2022/2024) · two-tower + session context at audio scale · `blog`
- [Netflix: How We Built Personalized Homepage](https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429) - Netflix (2012, still relevant architecture primer) · `blog`
- [Eugene Yan's RecSys Writing (2024)](https://eugeneyan.com/tag/recsys/) - collected practitioner deep-dives; updated regularly through 2025 · `blog`
