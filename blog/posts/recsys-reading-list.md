---
title: Recommendation Systems: Papers to Read
date: 2026-08-11
summary: A bottom-up reading list for understanding recommendation systems — from classic CF through industrial-scale deep learning, sequential models, graph networks, and LLM-era generative rec.
tags: recsys, papers, reading-list
theme: light
---

A bottom-up reading list for understanding recommendation systems: from classic CF through industrial-scale deep learning, sequential models, graph networks, and LLM-era generative rec. I'll keep adding to this over time rather than starting a new post for every batch.

**Type key:** `paper` = peer-reviewed / arxiv preprint · `blog` = blog/post · `report` = technical report · `book` = textbook/chapter · `survey` = survey paper · `website` = tool/org site · `dataset` = data release

---

## Part 1 - Foundations & Classic Methods

### Collaborative Filtering Origins

- [GroupLens: Applying Collaborative Filtering to Usenet News](https://dl.acm.org/doi/10.1145/138859.138867) - Resnick et al. (1994) · first CF paper · `paper`
- [Amazon.com Recommendations: Item-to-Item Collaborative Filtering](https://ieeexplore.ieee.org/document/1167344) - Linden, Smith & York, Amazon (2003) · powers "customers who bought" · `paper`
- [Collaborative Filtering for Implicit Feedback Datasets](https://ieeexplore.ieee.org/document/4781145) - Hu, Koren & Volinsky (2008) · WMF: treating clicks as confidence, not preference · `paper`
- [Matrix Factorization Techniques for Recommender Systems](https://ieeexplore.ieee.org/document/5197422) - Koren, Bell & Volinsky (2009) · the canonical MF overview from Netflix Prize · `paper`
- [SVD++: Factorization Meets the Neighborhood](https://dl.acm.org/doi/10.1145/1401890.1401944) - Koren (2008) · integrates implicit + explicit feedback · `paper`
- [Lessons from the Netflix Prize](https://www.netflixprize.com/assets/GrandPrize2009_BPC_BellKor.pdf) - BellKor (2009) · ensemble + SVD++ winning writeup · `report`
- [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1205.2618.pdf) - Rendle et al. (2009) · pairwise ranking loss for implicit data · `paper`

### Factorization Machines & Feature Interaction

- [Factorization Machines](https://ieeexplore.ieee.org/document/5694074) - Rendle (2010) · unifies MF + polynomial feature interaction · `paper`
- [Field-aware Factorization Machines (FFM)](https://arxiv.org/pdf/1701.04099.pdf) - Juan et al. (2016) · per-field interaction parameters · `paper`

### Simple Baselines Worth Knowing

- [SLIM: Sparse Linear Methods for Top-N Recommendation](https://ieeexplore.ieee.org/document/6137254) - Ning & Karypis (2011) · learned item-item sparse weight matrix · `paper`
- [EASE^R: Embarrassingly Shallow Autoencoders for Sparse Data](https://arxiv.org/pdf/1905.03375.pdf) - Steck (2019) · closed-form linear model that beats most deep models · `paper`
- [NMF: Algorithms for Non-negative Matrix Factorization](https://www.nature.com/articles/44565) - Lee & Seung, Nature (1999) · parts-based decomposition; used in topic rec · `paper`

---

## Part 2 - Deep Learning for Ranking & CTR Prediction

- [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf) - Cheng et al., Google (2016) · memorization (wide) + generalization (deep) · `paper`
- [DeepFM: A Factorization-Machine based Neural Network for CTR](https://arxiv.org/pdf/1703.04247.pdf) - Guo et al., Huawei (2017) · FM + MLP, no manual feature engineering · `paper`
- [Neural Collaborative Filtering (NCF)](https://arxiv.org/pdf/1708.05031.pdf) - He et al. (2017) · MLP replaces dot product in MF · `paper`
- [Deep & Cross Network (DCN)](https://arxiv.org/pdf/1708.05123.pdf) - Wang et al., Google (2017) · bounded-degree feature crosses explicitly · `paper`
- [DCN V2: Improved Deep & Cross Network](https://arxiv.org/pdf/2008.13535.pdf) - Wang et al., Google (2021) · matrix cross layers; state-of-the-art on Criteo · `paper`
- [xDeepFM: Combining Explicit and Implicit Feature Interactions](https://arxiv.org/pdf/1803.05170.pdf) - Lian et al., Microsoft (2018) · compressed interaction network · `paper`
- [DIN: Deep Interest Network for CTR Prediction](https://arxiv.org/pdf/1706.06978.pdf) - Zhou et al., Alibaba (2018) · attention over user history, target-aware · `paper`
- [DIEN: Deep Interest Evolution Network](https://arxiv.org/pdf/1809.03672.pdf) - Zhou et al., Alibaba (2019) · GRU with attention-based gate for evolving interest · `paper`
- [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921.pdf) - Song et al. (2019) · multi-head self-attention on feature embeddings · `paper`
- [FiBiNET: Feature Importance and Bilinear Feature Interaction](https://arxiv.org/pdf/1905.09433.pdf) - Huang et al., Sina Weibo (2019) · SENET for feature importance + bilinear · `paper`
- [BST: Behavior Sequence Transformer for E-commerce Recommendation](https://arxiv.org/pdf/1905.06874.pdf) - Chen et al., Alibaba (2019) · transformer applied to user behavior for CTR · `paper`
- [DLRM: Deep Learning Recommendation Model](https://arxiv.org/pdf/1906.00091.pdf) - Naumov et al., Meta (2019) · open-source reference industrial RecSys architecture · `paper`
- [FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction](https://arxiv.org/pdf/2304.00902.pdf) (2023) · simple dual-stream MLP beats complex models · `paper`
- [DHEN: Deep Hierarchical Ensemble Network for RecSys](https://arxiv.org/pdf/2203.11014.pdf) - Meta (2022) · hierarchical ensemble of interaction modules · `paper`
- [CTRL: Connect Tabular and Language Model for CTR Prediction](https://arxiv.org/pdf/2306.02841.pdf) - Yuan et al., Alibaba (2023) · bridges structured feature tables and LLM text for industrial CTR · `paper`

### Autoencoders for CF

- [Collaborative Denoising Auto-Encoders (CDAE)](https://dl.acm.org/doi/10.1145/2835776.2835837) - Wu et al. (2016) · `paper`
- [Variational Autoencoders for Collaborative Filtering (Mult-VAE)](https://arxiv.org/pdf/1802.05814.pdf) - Liang et al. (2018) · principled generative model for implicit CF · `paper`

---

## Part 3 - Sequential & Session-Based Recommendation

- [Session-based Recommendations with Recurrent Neural Networks (GRU4Rec)](https://arxiv.org/pdf/1511.06939.pdf) - Hidasi et al. (2015) · first RNN-based session recommendations · `paper`
- [Improved GRU4Rec with Pairwise Loss](https://arxiv.org/pdf/1706.03847.pdf) - Hidasi & Karatzoglou (2018) · `paper`
- [Caser: Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding](https://arxiv.org/pdf/1809.07426.pdf) - Tang & Wang (2018) · CNN for short-term sequential patterns · `paper`
- [SASRec: Self-Attentive Sequential Recommendation](https://arxiv.org/pdf/1808.09781.pdf) - Kang & McAuley (2018) · causal transformer over item history; dominant baseline · `paper`
- [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations](https://arxiv.org/pdf/1904.06690.pdf) - Sun et al. (2019) · cloze-task masked training for sequential rec · `paper`
- [SSE-PT: Sequential Recommendation via Personalized Transformer](https://arxiv.org/pdf/2007.00205.pdf) (2020) · personalization tokens in attention · `paper`
- [FMLP-Rec: Filter-enhanced MLP for Sequential Recommendation with Noise Filtering](https://arxiv.org/pdf/2202.13556.pdf) (2022) · learnable filters beat attention in many settings · `paper`
- [FEARec: Frequency Enhanced Hybrid Attention for Sequential Recommendation](https://arxiv.org/pdf/2306.02639.pdf) (2023) · frequency domain + time domain attention · `paper`
- [E4SRec: Effective, Efficient, Extensible Sequential Rec with LLMs](https://arxiv.org/pdf/2312.02443.pdf) (2024) · practical LLM-for-SeqRec deployable in production; LoRA + hybrid ID+text · `paper`
- [Breaking the Length Barrier: LLMs Excel at Supervised Long-Context Recommendation](https://arxiv.org/pdf/2402.03841.pdf) (2024) · fine-tuned LLMs with 10k+ behavior histories outperform all ID baselines · `paper`

---

## Part 4 - Graph-Based Recommendation

- [Graph Convolutional Matrix Completion (GC-MC)](https://arxiv.org/pdf/1706.02263.pdf) - Kipf et al. (2017) · GCN on bipartite user-item graph for MF · `paper`
- [PinSage: Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://arxiv.org/pdf/1806.01973.pdf) - Ying et al., Pinterest (2018) · random walk + GCN at 3B item production scale · `paper`
- [NGCF: Neural Graph Collaborative Filtering](https://arxiv.org/pdf/1905.08108.pdf) - Wang et al. (2019) · propagates user-item interactions via high-order graph · `paper`
- [LightGCN: Simplifying and Powering Graph Convolution for Recommendation](https://arxiv.org/pdf/2002.02126.pdf) - He et al. (2020) · removes weight matrices + nonlinearity; dominant baseline · `paper`
- [UltraGCN: Ultra Simplification of Graph Convolutional Networks](https://arxiv.org/pdf/2110.15114.pdf) (2021) · constraint-based loss, skips aggregation; faster than LightGCN · `paper`
- [SGL: Self-supervised Graph Learning for Recommendation](https://arxiv.org/pdf/2010.10783.pdf) - Wu et al. (2021) · three graph augmentation types + contrastive loss · `paper`
- [SimGCL: Are Graph Augmentations Necessary?](https://arxiv.org/pdf/2112.08679.pdf) (2022) · uniform noise in embedding space beats graph augmentation · `paper`
- [NCL: Neighborhood-enriched Contrastive Learning for Collaborative Filtering](https://arxiv.org/pdf/2202.06200.pdf) (2022) · structural and semantic neighbors as contrastive pairs · `paper`

---

## Part 5 - Two-Tower Retrieval Models

- [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf) - Covington, Adams & Basilic, Google (2016) · two-stage retrieval + ranking; the industry blueprint · `paper`
- [Learning Deep Structured Semantic Models (DSSM)](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf) - Huang et al., Microsoft (2013) · dual encoder text matching; predecessor to two-tower · `paper`
- [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://dl.acm.org/doi/pdf/10.1145/3298689.3346996) - Yi et al., Google (2019) · in-batch negatives with frequency correction for retrieval · `paper`
- [Embedding-based Retrieval in Facebook Search](https://arxiv.org/pdf/2006.11632.pdf) - Huang et al., Meta (2020) · hard negative mining + quantization at trillion scale · `paper`
- [Mixed Negative Sampling for Learning Two-Tower Neural Networks](https://dl.acm.org/doi/10.1145/3366423.3380092) - Yi et al., Google (2019) · batch + random negatives combined · `paper`
- [MOBIUS: Towards the Next Generation of Query-Ad Matching in Baidu](https://dl.acm.org/doi/pdf/10.1145/3292500.3330651) - Fan et al., Baidu (2019) · active learning for retrieval · `paper`

### ANN / Retrieval Infrastructure

- [Product Quantization for Nearest Neighbor Search](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf) - Jégou et al. (2011) · core ANN compression technique · `paper`
- [FAISS: A Library for Efficient Similarity Search](https://arxiv.org/pdf/2401.08281.pdf) - Douze et al., Meta (2024) · reference library for billion-scale MIPS · `paper`
- [HNSW: Efficient and Robust Approximate Nearest Neighbor Search](https://arxiv.org/pdf/1603.09320.pdf) - Malkov & Yashunin (2016) · graph-based ANN index; production standard · `paper`
- [ScaNN: Accelerating Large-Scale Inference with Anisotropic Vector Quantization](https://arxiv.org/pdf/1908.10396.pdf) - Guo et al., Google (2020) · MIPS-optimized quantization, beats FAISS · `paper`

---

## Part 6 - Multi-Task Learning & Multi-Objective Optimization

- [MMoE: Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007) - Ma et al., Google (2018) · gated expert sharing for multi-task rec · `paper`
- [ESMM: Entire Space Multi-task Model for CVR Prediction](https://arxiv.org/pdf/1804.07931.pdf) - Ma et al., Alibaba (2018) · CVR via CTR×CTCVR, corrects sample selection bias · `paper`
- [Recommending What Video to Watch Next: A Multitask Ranking System](https://dl.acm.org/doi/pdf/10.1145/3298689.3346997) - Zhao et al., Google (2019) · MMoE for Watch Next + position debiasing via shallow tower · `paper`
- [PLE: Progressive Layered Extraction for Multi-Task Learning](https://dl.acm.org/doi/pdf/10.1145/3383313.3412236) - Tang et al., Tencent (2020) · separates shared and task-specific experts explicitly · `paper`
- [SNR: Sub-Network Routing for Flexible Parameter Sharing in Multi-task Learning](https://ojs.aaai.org/index.php/AAAI/article/view/3788) - Ma et al. (2019) · `paper`

---

## Part 7 - Long User History & Memory

- [MIMN: Practice on Long Sequential User Behavior Modeling for CTR Prediction](https://arxiv.org/pdf/1905.09208.pdf) - Pi et al., Alibaba (2019) · memory network for 1000+ item history · `paper`
- [SIM: Search-based User Interest Model](https://arxiv.org/pdf/2006.05639.pdf) - Pi et al., Alibaba (2020) · two-stage hard/soft search for 10k+ length histories · `paper`
- [ETA: End-to-end Target Attention for Long Sequence Modeling](https://arxiv.org/pdf/2108.02999.pdf) - Chen et al. (2021) · hash-based target attention over long history · `paper`
- [SDIM: Sampling-based Deep Interaction Model for Long Sequence CTR](https://arxiv.org/pdf/2209.00081.pdf) - Cao et al., Meituan (2022) · `paper`

---

## Part 8 - Industrial Systems at Scale

- [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf) - Covington et al., Google (2016) · `paper`
- [DLRM: Deep Learning Recommendation Model](https://arxiv.org/pdf/1906.00091.pdf) - Naumov et al., Meta (2019) · `paper`
- [Pixie: A System for Recommending 3+ Billion Items to 200M+ Users in Real-Time](https://arxiv.org/pdf/1711.07601.pdf) - Eksombatchai et al., Pinterest (2018) · bipartite random walk at scale · `paper`
- [Real-time Personalization using Embeddings for Search Ranking at Airbnb](https://dl.acm.org/doi/pdf/10.1145/3219819.3219885) - Grbovic & Cheng, Airbnb (2018) · listing2vec via listing/user co-click sessions · `paper`
- [Applying Deep Learning To Airbnb Search](https://arxiv.org/pdf/1810.09591.pdf) - Haldar et al., Airbnb (2019) · neural ranking with online/offline gap analysis · `paper`
- [Improving Deep Learning for Airbnb Search](https://arxiv.org/pdf/2002.05515.pdf) - Haldar et al., Airbnb (2020) · Listing Quality Score + architecture lessons · `paper`
- [Monolith: Real-Time Recommendation System with Collisionless Embedding Table](https://arxiv.org/pdf/2209.07663.pdf) - Liu et al., ByteDance/TikTok (2022) · streaming training on live traffic, no collision hashing · `paper`
- [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations (HSTU)](https://arxiv.org/pdf/2402.17152.pdf) - Zhai et al., Meta (2024) · 1T parameter HSTU architecture; scaling laws for rec · `paper`
- [Wukong: Towards a Scaling Law for Large-Scale Recommendation](https://arxiv.org/pdf/2403.02545.pdf) - Yao et al., LinkedIn (2024) · how rec models scale vs LLMs · `paper`
- [PinnerFormer: Sequence Modelling for User Representation at Pinterest](https://arxiv.org/pdf/2205.04507.pdf) - Zhai et al., Pinterest (2022) · offline-trained transformer for dense user embeddings · `paper`
- [On the Factory Floor: ML Engineering for Industrial-Scale Ads Recommendation Models](https://arxiv.org/pdf/2209.05310.pdf) - Meta (2022) · engineering lessons at ads-scale · `paper`
- [Instagram's Explore Recommender System](https://ai.meta.com/blog/powered-by-ai-instagrams-explore-recommender-system/) - Meta AI (2019) · `blog`
- [Twitter/X Open-Sourced Recommendation Algorithm](https://blog.twitter.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm) - Twitter/X (2023) · `blog`
- [Netflix Recommendation: Beyond the 5 Stars](https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429) - Netflix Tech Blog (2012) · `blog`
- [YouTube Blog: How the Recommendation System Works](https://blog.youtube/inside-youtube/on-youtubes-recommendation-system/) - YouTube (2021) · `blog`
- [Eugene Yan's Applied RecSys Writing](https://eugeneyan.com/tag/recsys/) - practitioner deep-dives on production rec · `blog`

---

## Part 9 - Knowledge Graph-Enhanced Recommendation

- [RippleNet: Propagating User Preferences on the Knowledge Graph](https://arxiv.org/pdf/1803.03467.pdf) - Wang et al. (2018) · ripple set propagation over KG triples · `paper`
- [KGCN: Knowledge Graph Convolutional Networks for Recommendation](https://arxiv.org/pdf/1904.12575.pdf) - Wang et al. (2019) · `paper`
- [KGAT: Knowledge Graph Attention Network for Recommendation](https://arxiv.org/pdf/1905.07854.pdf) - Wang et al. (2019) · attentive graph propagation in KG · `paper`
- [MKR: Multi-Task Feature Learning for Knowledge Graph Enhanced Recommendation](https://arxiv.org/pdf/1901.08907.pdf) - Wang et al. (2019) · cross-compress units bridge KG and rec · `paper`
- [KGIN: Knowledge Graph-based Intent Network for Recommendation](https://arxiv.org/pdf/2101.06918.pdf) (2021) · user intent modeling via KG relational paths · `paper`

---

## Part 10 - Contrastive & Self-Supervised Learning for RecSys

- [CL4SRec: Contrastive Learning for Sequential Recommendation](https://arxiv.org/pdf/2010.14395.pdf) - Xie et al. (2022) · crop/mask/reorder augmentations on sequences · `paper`
- [CoSeRec: Contrastive Self-supervised Sequential Recommendation with Robust Augmentation](https://arxiv.org/pdf/2108.06479.pdf) (2021) · `paper`
- [SGL: Self-supervised Graph Learning for Recommendation](https://arxiv.org/pdf/2010.10783.pdf) - Wu et al. (2021) · graph augmentation + contrastive loss · `paper`
- [DirectAU: Towards Representation Alignment and Uniformity in Collaborative Filtering](https://arxiv.org/pdf/2206.12811.pdf) (2022) · alignment + uniformity objectives from Wang & Isola · `paper`
- [Alignment and Uniformity on the Hypersphere](https://arxiv.org/pdf/2005.10242.pdf) - Wang & Isola (2020) · theoretical grounding for CL objectives · `paper`
- [NCL: Neighborhood-enriched Contrastive Learning](https://arxiv.org/pdf/2202.06200.pdf) (2022) · structural and semantic neighborhood as positive pairs · `paper`

---

## Part 11 - Multimodal Recommendation

- [VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1510.01784.pdf) - He & McAuley (2016) · visual features from CNN in MF · `paper`
- [ACF: Attentive Collaborative Filtering with Item and Component-level Attention](https://dl.acm.org/doi/pdf/10.1145/3077136.3080797) - Chen et al. (2017) · `paper`
- [BM3: Bootstrapped Multi-modal Self-supervised Learning for Recommendation](https://arxiv.org/pdf/2207.05969.pdf) (2022) · modal dropout + contrastive learning · `paper`
- [MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation](https://dl.acm.org/doi/pdf/10.1145/3474085.3475356) (2021) · `paper`
- [Artwork Personalization at Netflix](https://netflixtechblog.com/artwork-personalization-c589f074ad76) - Netflix Tech Blog (2017) · personalized thumbnail selection via contextual bandits · `blog`
- [AlphaRec: Scalable Multimodal Recommendation with Vision-Language Models](https://arxiv.org/pdf/2501.14456.pdf) (2025) · CLIP embeddings as item representations; zero-shot cold-start rec · `paper`

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
- [NoteLLM: A Retrievable Large Language Model for Note Recommendation](https://arxiv.org/pdf/2403.01744.pdf) - Xiaohongshu/REDNote, WWW (2024) · end-to-end LLM as retriever at production note-rec scale · `paper`
- [HLLM: Enhancing Sequential Recommendations via Hierarchical Large Language Models](https://arxiv.org/pdf/2409.12111.pdf) (2024) · item-level LLM encodes content; user-level LLM models behavioral history · `paper`
- [AgentCF: Collaborative Learning with Autonomous Language Agents](https://arxiv.org/pdf/2310.09233.pdf) (2024) · LLM agents simulate user-item interaction to generate training signal · `paper`
- [RecMind: Large Language Model Powered Agent for Recommendation](https://arxiv.org/pdf/2308.14296.pdf) (2024) · planning + memory + tool use in LLM agent for rec · `paper`
- [AgentRec: Simulation-based Evaluation via LLM Agents](https://arxiv.org/pdf/2405.02310.pdf) (2024) · replaces offline user datasets with LLM-simulated users for eval · `paper`
- [MoRec: Recommenders are Sequentially Human Preference Models (Text vs ID)](https://arxiv.org/pdf/2211.09971.pdf) (2022, widely discussed 2024) · text encoders rival ID embeddings - live frontier debate · `paper`
- [Generative Recommendation: Towards Next-generation Recommender Paradigm](https://arxiv.org/pdf/2304.03516.pdf) (2023) · position paper framing the shift from retrieve-rank to generate · `paper`

### Diffusion Models for Recommendation

- [DiffRec: Diffusion Recommender Model](https://arxiv.org/pdf/2304.04971.pdf) - Lin et al., RecSys (2023) · DDPM applied to CF; strong on cold/sparse data · `paper`
- [CF-Diff: Collaborative Filtering with Diffusion Models](https://arxiv.org/pdf/2310.05108.pdf) (2023) · multi-step denoising as personalization · `paper`

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
- [How Can Recommender Systems Benefit from Large Language Models: A Survey](https://arxiv.org/pdf/2306.05817.pdf) - Lin et al. (2023/2024) · taxonomy of LLM roles: encoder, ranker, generator, agent · `survey`
- [A Survey on Diffusion Models for Recommendation](https://arxiv.org/pdf/2409.05687.pdf) (2024) · covers DiffRec, CF-Diff, and generation-based rec approaches · `survey`
