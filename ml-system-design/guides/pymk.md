# People You May Know: interview guide

> The one-line answer: PYMK is link prediction served as a funnel. Several cheap candidate generators (friends-of-friends, embeddings, contact books) give recall, a learned ranker predicts mutual value, and re-ranking rules handle safety, hub fairness and fatigue. Most of the interview is spent on objective choice, pair features, time-correct training data and feedback loops.

## 1. Problem statement and how to open

**Prompt.** "Design People You May Know for a social network with about 1B users."

Do not start drawing. Spend 4 to 5 minutes on questions, then state your assumptions out loud and ask where the interviewer wants depth: "I can go deep on candidate generation, features and ranking, or evaluation. Which one matters most to you?"

**Product**
- Is the edge symmetric (Facebook friend, needs acceptance) or directed (follow)? *Symmetric means two people must both benefit, so you score both directions.* Assume symmetric.
- Where does it surface? Feed unit, dedicated tab, post-signup flow, notifications. *Each surface has a different latency budget and a different user intent.*
- What does success mean to the business: more connections, more engagement, new-user retention? *This picks the label.*
- Who should never be suggested? Blocked users, people the user removed, minors to unknown adults, reported accounts.

**Scale**
- 1B users, roughly 300 to 500 friends on average, so on the order of 10^11 edges.
- Friends-of-friends per user is 10^4 to 10^5 before dedup. All pairs is 10^18, so you can never score all pairs.
- Read QPS: assume 200M daily PYMK views, peak about 10k QPS.

**Constraints**
- Latency: 200 ms p99 if computed online for the feed unit. A tab can serve a precomputed list.
- Freshness: a brand-new user must get suggestions in the first session, since that is when retention is decided.
- Privacy: which signals are allowed (contact upload, location, profile views)? *Creepy suggestions are a real product risk, so ask.*

**Data**
- Friend graph with edge timestamps, request sent/accepted/rejected/ignored logs, PYMK impression logs with position, profile attributes, contact uploads, interaction logs.

## 2. Objectives and metrics

| Option | Label | For | Against |
|---|---|---|---|
| A. Request sent | P(u sends request to v, given impression) | Dense, fast label | One-sided; rewards spammy reach-outs and celebrity suggestions |
| B. Friendship formed | P(send) × P(accept) | Captures both sides; still fast | A friendship that never interacts is worth little |
| C. Meaningful friendship | Friendship formed and the pair interacts within 30 days | Closest to real value and retention | Sparse, 30-day label delay |

**Choice.** Rank by an expected-value score built from separate heads:

`score(u,v) = P(send) × P(accept) × (1 + a × P(interact in 30d)) - b × P(dismiss or report)`

Train the fast heads daily and the slow head (interaction) weekly. This gives you B's label volume with C's direction.

**Offline metrics:** recall@K per generator and for the union, AUC and PR-AUC for the ranker, NDCG@10, calibration.
**Online primary:** friendships formed per user that have an interaction within 7 days. For new users, share reaching 10 friends in 14 days.
**Guardrails:** dismiss ("X") rate, reports and blocks from PYMK, request rejection rate, request load on the receiving side, exposure concentration (Gini), latency p99.

## 3. High-level architecture

```
OFFLINE (daily)                         NEARLINE (minutes)
graph snapshot -> FoF job (2-hop)  \    new edge / new contact upload
               -> embedding train   >-> candidate store      -> incremental FoF update
               -> ANN index build  /    (user -> ~1-2k cands)   for the two endpoints

ONLINE (per request, ~150 ms)
request -> fetch candidates (KV)  ----\
        -> ANN query (embeddings)  ----+-> union ~1-2k -> filters (friends, blocked,
        -> contacts / school lookup --/                    dismissed, safety)
        -> light ranker (LR or small GBDT) -> 200
        -> heavy ranker (GBDT / MTL DNN, pair features from feature store) -> 50
        -> re-rank (hub discount, impression caps, diversity, fatigue) -> top 10
        -> log impression + features served (for training)
```

**Budget:** candidate fetch 15 ms, features 40 ms, light rank 10 ms, heavy rank 40 ms, re-rank 5 ms, rest network. Most surfaces can be served from a precomputed list refreshed every few hours, with only filters and re-rank done online.

## 4. Data and labels

- **Positives:** an impression followed by request and accept within 7 days. Also organic friendships (search, profile visit) as extra positives for retrieval training, since they are free of PYMK's own bias.
- **Negatives:** impressions with no request. Explicit dismissals are strong negatives and deserve higher weight. For retrieval models, add random and in-batch negatives, otherwise the model only learns to separate among things it already retrieved.
- **Sampling:** base rate is 1 to 3%. Downsample negatives at rate w and correct the probability: `p' = p / (p + (1 - p) / w)`.
- **Position bias:** slot 1 gets more requests regardless of quality. Log position, use it as a training feature, fix it to a constant at serving.
- **Selection bias:** you only have labels for pairs the old system showed. Reserve 1% of slots for randomized candidates from the union to get unbiased data.
- **Label delay:** accept can take days. Use a fixed attribution window and do not train on impressions younger than the window.
- **Leakage trap:** features must be as of impression time. Profile views, messages and mutual-friend counts all jump after the friendship forms. Reading them from today's tables leaks the label. Log features at serving time, or do point-in-time joins.

## 5. Feature engineering

| Feature | Type | Encoding | Why | Freshness |
|---|---|---|---|---|
| Mutual friend count | pair, graph | log(1+x), bucketed | Triadic closure, strongest classic signal | Daily, plus nearline |
| Adamic-Adar | pair, graph | dense float | Mutual friends who have few friends carry more information | Daily |
| Jaccard of friend sets | pair, graph | dense float | Normalizes for people with huge friend lists | Daily |
| Embedding cosine | pair, learned | dense float | Reaches beyond 2 hops; community-level similarity | Weekly train, daily infer |
| Contact-book overlap | pair | flags: u has v, v has u, both; shared-contact count | Real-world tie, works with zero friends | On upload |
| Same school / employer / city | pair, profile | binary, plus overlapping years | Shared context | On profile edit |
| Profile-view recency | pair, behavior | exp(-days / 7), direction split | Direct intent signal | Real time |
| Age gap | pair, profile | bucketed absolute difference | Homophily | Static |
| min and max degree of the pair | node | log, bucketed | Hub detection; request-load context | Daily |
| Recent-tie closure | pair, graph | count of mutual friends added in last 30 days | New job or new school makes whole clusters connectable now | Nearline |
| Mutual-friend cohesion | pair, graph | edge density among the mutual friends | Ten mutuals who all know each other is one context; ten unrelated is ten contexts | Daily |
| Candidate accept rate | node, v-side | smoothed rate | Some people accept everyone, some no one | Daily |
| Viewer's PYMK behavior | node, u-side | send rate, dismiss rate, last 30 days | Calibrates per-user propensity | Daily |
| Prior impressions of this pair | pair | count, days since last | Fatigue: the 5th impression is worth far less | Real time |

**Formulas to write on the board**

`AdamicAdar(u,v) = sum over w in N(u) ∩ N(v) of 1 / log(deg(w))`

`Jaccard(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|`

`cohesion(u,v) = edges among mutual friends / (m × (m - 1) / 2)`, where m is the mutual count

**Encoding notes.** Counts are heavy tailed, so use log or quantile buckets. IDs (school, employer, city) go in as hashed sparse features or embeddings in a DNN. Pair features are expensive because there are too many pairs to precompute, so compute graph pair features only for the retrieved candidates, and cache each user's friend list as a sorted array or bitmap for fast intersection.

## 6. Modeling

**Baseline:** rank friends-of-friends by mutual friend count. No training, easy to explain, and a real benchmark. In the demo it gets 25% recall@10 against 45% for the learned ranker.

| Model | Scalability | Latency | Cold start | Interpretability | Notes |
|---|---|---|---|---|---|
| Logistic regression on bucketed and crossed features | Excellent | Under 1 ms per 1k | Works if contact features exist | High | Good light ranker; needs manual crosses |
| GBDT (LightGBM / XGBoost) | Good, to about 1B rows | About 5 to 10 ms per 1k | Same | Medium (SHAP) | Best on dense, tabular pair features; weak on raw IDs |
| Two-tower DNN or GNN (GraphSAGE / PinSage style) | Train on GPUs; ANN serving | ANN: a few ms | Uses profile features for new users | Low | For retrieval, not final ranking; cannot use pair features |
| Multi-task DNN (shared bottom or MMoE) | Heavy | 20 to 40 ms for 200 | Same as GBDT | Low | Heads for send, accept, interact, dismiss; handles ID embeddings |

**Decision.** Retrieval is FoF plus two-tower/GNN embeddings in an ANN index plus contact and attribute lookups. Ranking starts with GBDT on pair features, because the features are dense and tabular and GBDT wins there with little tuning. Move to a multi-task DNN once you need several heads and ID embeddings. This ties back to the objective: the score multiplies separately predicted probabilities, so the model must be calibrated, and both GBDT with log loss and DNN heads give you that.

**Reciprocity.** Score both directions. `P(u sends to v) × P(v accepts u)` is not symmetric, and showing v to u while v would never accept wastes the slot and annoys v.

## 7. Training

- **Split by time, always.** Features as of T0, labels from (T0, T1]. Validate on features as of T1, labels from (T1, T2]. Never a random split over pairs.
- **Loss:** log loss per head. For two-tower retrieval, sampled softmax with in-batch negatives and a logQ correction for popularity.
- **Negatives:** 10 to 1 downsampling with the calibration correction from section 4. Add hard negatives (FoF pairs with many mutuals that did not connect) for the retrieval model.
- **Imbalance:** handled by downsampling plus correction. Do not report accuracy.
- **Cadence:** ranker daily on a rolling 30-day window, embeddings weekly with daily inference for new nodes, FoF candidates daily batch plus nearline updates on new edges.
- **Weights:** dismissals weighted 3 to 5x; positives that led to interaction weighted up.

## 8. Evaluation

**Offline**
- Retrieval: recall@N of future friendships per generator and for the union. Report separately for warm and cold-start users.
- Ranker: AUC, PR-AUC, NDCG@10, calibration plot. Slice by degree bucket, tenure and country.
- Replay on randomized-slot logs for an unbiased estimate.

**Online**
- A/B test, but friend graphs break independence: treating u changes what v sees and receives. With user-level randomization the control group is contaminated and the effect is biased. Use cluster randomization (graph partitioning) or country-level tests for the launch decision, and user-level tests for fast iteration where the bias is accepted.
- Primary: friendships with interaction within 7 days, per user. Secondary: requests sent, accept rate. Guardrails as in section 2.
- Size: with a 2% baseline conversion and a 2% relative lift target, you need roughly 2M users per arm at 80% power. Cluster designs need more because the effective sample is the number of clusters.
- Novelty and depletion: a better model harvests obvious ties in week one, so the lift decays. Run 3 to 4 weeks and keep a long-term holdout of 1%.

## 9. Debugging, monitoring, deployment

- **Offline up, online flat.** Check, in order: leakage (features not point-in-time), training/serving skew (log and compare feature values), candidate set mismatch (ranker trained on the old system's candidates), position bias, metric mismatch.
- **Drift:** feature distribution monitors (PSI) per feature, calibration by day, null rates. A logging change upstream shows up as a single feature shifting overnight.
- **Slices:** new users, low-degree users, hubs, each top country, each surface. An aggregate win can hide a new-user loss.
- **Feedback loop:** the model trains on its own impressions, so it over-learns what it already shows. Mitigate with randomized slots, organic-friendship positives, and exposure caps.
- **Deployment:** shadow mode (score and log, do not serve) to check latency and score distribution, then 1% canary with auto-rollback on guardrails, then ramp. Keep the previous model and index warm so rollback is a config change.
- **Safety checks** run after ranking and cannot be bypassed by any model: block list, removed friends, reported pairs, age rules.

## 10. Scaling and advanced topics

- **FoF at 10^11 edges:** a 2-hop join is the dominant cost. Cap fan-out through high-degree nodes (they produce most paths and the least information, which is exactly what Adamic-Adar says), sample neighbors, keep top-k per user by a cheap score. Update incrementally: a new edge (a,b) only affects friends of a and friends of b.
- **Embeddings:** GraphSAGE-style inductive models compute embeddings for new nodes without retraining. ANN with IVF-PQ or HNSW, sharded by region.
- **10x QPS:** precompute and cache the ranked list per user, refresh on a trigger (new friend, contact upload) or every few hours. Only filters and re-rank stay online.
- **Multi-objective:** the weights a and b in the score are tuned by online experiment, not by offline loss. Hard constraints (request load per receiver, safety) are rules, not terms in the objective.
- **Fairness of exposure:** hub discount, per-candidate impression caps, and a minimum exposure for new users so they receive requests too.
- **Extensibility:** a feature registry with owners and point-in-time definitions, the generator interface as a plugin (each returns id, score, source), and per-generator recall dashboards so a new generator is judged on incremental recall.
- **Privacy:** no location-proximity or profile-view signal that reveals who looked at whom, contact data only with consent and deletable, and suppress suggestions that expose sensitive contexts (shared doctor, support groups).

## 11. Mapping to the demo

| Demo element | Concept |
|---|---|
| Section 1, recall by stage | Multi-generator retrieval; union recall is the ceiling; warm vs cold-start slices |
| Section 1, feature importance | Which pair features matter; redundancy of mutual-friend count |
| Section 2, funnel and generator table | Retrieve, filter, rank; per-stage latency and candidate counts |
| Section 2, "why" panel | Per-feature contributions; how you would debug a bad suggestion |
| Section 2, cold-start user | Contacts and attributes fallback when graph signals are empty |
| Section 2, dismiss button | Negative feedback as a hard filter and a training signal |
| Section 2, celebrity discount | Re-ranking for exposure fairness |
| Section 3, ANN table | Recall vs latency in approximate nearest-neighbor search |
| Section 4, random vs time split | Label leakage and why offline AUC lied |
| Section 5, simulation | Hub bias, feedback loop, impression caps, yield decay (novelty) |

## 12. Follow-up questions

<details>
<summary>Why not just rank by mutual friends? It is simple and works.</summary>
It is the right baseline and a fine first candidate generator. It fails in three ways: it returns nothing for users with no friends, it over-suggests hubs because they share a mutual friend with everyone, and it treats all mutual friends as equal. In the demo it gets 25% recall@10 versus 45% for the ranker, and gives hubs 17% of slots when they account for 9% of true future friendships.
</details>

<details>
<summary>A brand-new user has zero friends. What do you show?</summary>
Graph generators are empty, so use contact-book matches in both directions (people who have this user's phone or email in their contacts are a strong signal), same school and employer with overlapping years, signup context (invite link, referrer), and popular people in their locale as a last resort. Rank with a model that has those features. Measure this slice separately, with time-to-10-friends as the metric.
</details>

<details>
<summary>You cannot do a 2-hop join over 100 billion edges every day. How do you compute friends-of-friends?</summary>
Bound the work. Cap neighbor expansion through high-degree nodes, since a path through someone with 5,000 friends carries little information. Keep only the top few thousand candidates per user by a cheap score such as Adamic-Adar. Run the full batch job weekly and apply incremental updates nearline: a new edge (a,b) only creates candidates between a's friends and b, and b's friends and a.
</details>

<details>
<summary>Offline AUC went from 0.91 to 0.99 with a new feature, but the A/B test is flat. What happened?</summary>
A jump that large is leakage until proven otherwise. Check whether the feature is computed after the label event, such as profile views or messages read from current tables. Compare the feature's distribution in training against values logged at serving. The demo shows this exact case: random split gives 0.99 offline and 0.81 in production, time split gives 0.94 and 0.91.
</details>

<details>
<summary>Why is a standard user-level A/B test problematic here?</summary>
Interference. A suggestion shown to a treated user creates a request to a control user, and a new edge changes both users' future candidates. Control is contaminated, so the measured difference is biased, usually toward zero for the total effect. Use graph-cluster randomization or geo tests for the ship decision and compare with the user-level estimate to gauge the spillover.
</details>

<details>
<summary>Friend requests went up 8% after launch but accept rate fell and dismissals rose. Ship?</summary>
No. The model is optimizing the sender's side only. Check whether exposure shifted toward hubs or strangers, and look at receiver-side load. Add P(accept) and a dismiss penalty to the score, add a cap on requests received per person per day, and make friendships-with-interaction the primary metric.
</details>

<details>
<summary>How do you stop the system from suggesting an abusive ex or a blocked person's alt account?</summary>
Hard filters after ranking for blocks in either direction, removed friends and reported pairs. Extend blocks to linked accounts using device and contact signals. Suppress suggestions based on sensitive signals, and let users opt out of being suggested through contacts. These are rules with tests, not model features, because one failure is unacceptable.
</details>

<details>
<summary>The metric dropped 5% overnight with no model change. How do you debug?</summary>
Start with what changed: deploys, logging, upstream pipelines. Check candidate counts per generator, since an FoF job failure shows up as fewer candidates. Check feature null rates and distributions, the score distribution, and latency timeouts that silently drop the heavy ranker. Slice by platform, country and surface to localize. Roll back the most recent change if a guardrail is breached, then investigate.
</details>

<details>
<summary>Why GBDT and not a deep model for ranking?</summary>
The strongest signals are a few dozen dense pair features, where GBDT is hard to beat, trains in minutes, and is easy to debug. A DNN earns its cost when you need multiple heads trained jointly, high-cardinality ID embeddings, or sequence features. A common path is GBDT first, then a multi-task DNN that takes GBDT-era features plus embeddings.
</details>

<details>
<summary>How does the model's own output bias its future training data?</summary>
It only gets labels for pairs it showed, and shown pairs gain features (impressions, views) that unshown pairs lack. Over time it narrows toward what it already believes. Counter with a small randomized slice of traffic, inverse-propensity weighting on logged data, organic friendships as positives, and tracking exposure concentration as a guardrail.
</details>

<details>
<summary>Twist: the product is now follow-based, like Instagram or X. What changes?</summary>
The edge is directed and needs no acceptance, so the label is follow plus later engagement with the followed account's content. Popularity is now a legitimate signal, since following celebrities is the point, so the hub discount relaxes. Retrieval leans on co-follow ("people who follow A also follow B") and content interest embeddings. Creator-side exposure fairness becomes the two-sided concern.
</details>

<details>
<summary>How would you handle 100x more candidate sources without ranking latency growing?</summary>
Give each generator a quota and a calibrated score, merge and dedup, then use a light ranker to cut to a fixed number (say 200) before the heavy ranker. Heavy ranker cost stays constant. Judge each generator by its incremental recall over the union, and remove ones that add none.
</details>

## Common mistakes

- Jumping to a model before naming the objective and the label.
- Treating it as one-sided: optimizing requests sent and ignoring the receiver.
- Proposing to score all pairs, or having no candidate generation story.
- Random train/test split on pairs; features not point-in-time.
- Only mutual-friend features; nothing for cold-start users.
- Reporting accuracy on a 2% base rate.
- Only offline or only online evaluation; no mention of network interference.
- No safety filters, no negative feedback, no exposure caps.
- Not circling back: failing to explain why the chosen model serves the chosen objective.
