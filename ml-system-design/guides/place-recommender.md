# Nearby Place Recommendations: interview guide

> The one-line answer: a local recommender is a geo-constrained funnel. A spatial index retrieves what is reachable, embeddings add personal taste, a ranker combines taste with distance, time and context, and a re-rank step handles diversity and exploration of new places. The hard parts are the biases in the data: people only visit what is close, only click what is shown high, and popular places get more of both.

## 1. Problem statement and how to open

**Prompt.** "Design a system that recommends nearby places such as restaurants, cafes and attractions to a user of a maps or social app."

Ask first, then state assumptions and offer the interviewer a choice of depth: retrieval and geo indexing, features and ranking, or evaluation and bias.

**Product**
- What is the surface? A "nearby" tab the user opens (pull, clear intent), a home-feed module, or a push notification (interruptive, needs high precision). *Sets the precision bar and latency.*
- Is there a query or category ("coffee"), or is it zero-query discovery? Assume zero-query with optional category filter.
- What counts as success: a tap, directions requested, a visit, a booking, a good rating afterwards? *Picks the label.*
- Are there sponsored places? Assume organic only, with ads as a separate slot.

**Scale**
- 500M monthly users, 100M places worldwide, about 10k requests per second at peak.
- Within 2 km of a user in a dense city: 5k to 50k places. In a rural area: fewer than 50.

**Constraints**
- Latency 200 ms p99 on mobile.
- Location is sensitive. How precise is the location we receive, and can we store history?
- Places open, close, move and change hours daily. Recommending a closed place is the worst error.

**Data**
- Check-ins, taps, directions, saves, bookings, ratings and reviews, dwell time from location pings (if consented), place attributes (category, price, hours, photos), search logs.

## 2. Objectives and metrics

| Option | Label | For | Against |
|---|---|---|---|
| A. Taps | P(tap on the place card) | Dense and immediate | Rewards attractive photos and curiosity, not good visits |
| B. Visits | P(directions requested or visit within 24 h) | Real intent, moderately dense | Delayed; visit detection is noisy; biased to nearby |
| C. Satisfying visits | Visit followed by long dwell, save, good rating or return | The real goal | Sparse and slow; ratings are given by under 5% of visitors |

**Choice.** Rank by expected value from several heads:

`score = P(tap) × ( w1 × P(visit after tap) + w2 × P(satisfied after visit) ) - w3 × P(hide or "not interested")`

Train tap and visit heads on dense data and the satisfaction head on the sparse signals, sharing embeddings.

**Offline:** recall@K for retrieval, NDCG@10 and recall@10 for ranking, both on a time-based split, sliced by new places, new users, and users away from home.
**Online primary:** visits per session that are followed by a positive signal.
**Guardrails:** share of suggestions that are closed or out of hours, average distance of suggested places, catalogue coverage and share of impressions to new places, hide rate, latency.

## 3. High-level architecture

```
request: user id, lat/lon (maybe coarsened), time, device
  |
  +-> geo retrieval: cell index (geohash / S2 / H3), radius adapts to density   100M -> ~5k   (10 ms)
  +-> embedding retrieval: ANN over place vectors, filtered to nearby cells      -> +500
  +-> other sources: saved places, friends' visits, trending in this area        -> +200
  |
  -> hard filters: open now, permanently closed, budget, dietary, blocked        -> ~3k
  -> light ranker: dot product + distance decay                                  -> 300      (10 ms)
  -> heavy ranker: GBDT or multi-task DNN on taste + geo + context features      -> 50       (40 ms)
  -> re-rank: diversity (MMR), exploration slots for new places, business rules  -> 10-20
  -> log impressions with position, features as served, propensity of exploration

offline: check-in / click streams -> point-in-time features -> daily ranker retrain
         weekly embedding retrain; place attributes and hours refreshed continuously
```

**Budget.** 200 ms total: retrieval 20, features 40, ranking 50, re-rank 5, the rest network. The spatial query is cheap. Feature fetch for thousands of candidates is the cost, which is why the light ranker exists.

## 4. Data and labels

- **Positives:** visits and directions requests attributed to an impression within 24 hours. Organic visits (not from a recommendation) are valuable extra positives for the embedding model because they carry no display bias.
- **Negatives for ranking:** places shown and not acted on. Weight explicit "not interested" higher.
- **Negatives for the embedding model:** this is the trap. Sampling negatives from the whole city teaches the model where the user lives, since anything far away is an easy negative. Sample negatives from the same area as the positive, so the only way to tell them apart is taste.
- **Distance bias:** "did not visit" mostly means "too far". Model distance explicitly so the taste signal is not forced to absorb it.
- **Position bias:** higher slots get more taps. Log position and either use it as a feature fixed at serving, or estimate how attention falls with rank from a small randomized slice and weight clicks by the inverse.
- **Popularity bias:** counts reflect past exposure. Use rates per impression, corrected for position, with smoothing.
- **Label delay:** visits arrive hours later, ratings days later. Use a fixed attribution window.
- **Leakage:** split by time; compute visit counts and user history as of the impression time. A random split lets the model see a user's future favourites.
- **Visit detection noise:** GPS error in dense areas assigns visits to the wrong place. Use dwell time, Wi-Fi and explicit check-ins to confirm.

## 5. Feature engineering

| Feature | Type | Encoding | Why | Freshness |
|---|---|---|---|---|
| Distance user to place | geo | km, log-bucketed; also travel time | The strongest single feature | Request time |
| Distance relative to the user's norm | geo cross | distance / user's median trip | 2 km is far for a walker and near for a driver | Daily |
| User-place embedding score | learned | dot product | Personal taste beyond categories | Weekly train |
| User's category share | user-place cross | share of past visits in this category, smoothed | Simple, robust taste signal | Daily |
| Price gap | cross | place price level minus user's average | Budget fit | Daily |
| Category by hour lift | context | P(category at this hour) / P(category) | Coffee at 8:00, bars at 22:00 | Weekly |
| Category by weather, by weekday or weekend | context | lift ratios | Parks when dry, museums in rain | Weekly |
| Open now, minutes until closing | context | binary, bucketed minutes | Avoids the worst error | Real time |
| Popularity as a rate | place | visits per impression, position-corrected, smoothed | Quality proxy without exposure bias | Hourly |
| Rating and review count | place | Bayesian average | Quality with few reviews handled | Daily |
| Place age and trend | place | days since opening; 7-day over 28-day visits | New and trending places | Daily |
| User location context | user | at home, at work, travelling, in a new city | Tourists and locals want different things | Request time |
| Friends who visited | social | count, recency | Trust signal | Daily |
| Session intent | sequence | last searches and taps in this session | Just searched "pizza" | Real time |
| Previously visited | cross | count, days since last | Favourites vs discovery | Real time |

**Formulas to write down**

Distance decay, tuned per user or per mode of travel: `f(d) = exp(-d / d0)`, or as a penalty `score - gamma × d`

Context lift: `lift(category, hour) = P(category | hour) / P(category)`

Bayesian rating, so one 5-star review does not beat 500 at 4.6: `rating = (n × mean + m × prior_mean) / (n + m)`, with m around 20

Smoothed rate: `rate = (visits + a × prior_rate) / (impressions + a)`

**Encoding notes.** Location itself goes in as a cell ID at two or three resolutions (hashed or embedded), never as raw latitude and longitude, which trees and networks split poorly. Time goes in as hour bucket plus cyclical encoding. Crosses of user context with place category carry most of the context signal.

## 6. Modeling

**Baseline:** most popular open places within the radius, by smoothed rate. In the demo this alone reaches 25% recall@10 against 10% for citywide popularity. Geography is a strong prior.

| Model | Scalability | Latency | Cold start | Interpretability | Notes |
|---|---|---|---|---|---|
| Popularity nearby | Trivial | Under 1 ms | Handles new users; fails new places | High | The baseline and the fallback |
| Matrix factorization or two-tower embeddings | ANN serving; weekly training | A few ms | Two-tower with content features handles new places and users | Low | Good for retrieval and as a ranker feature; knows nothing about context |
| GBDT on taste + geo + context features | Good to hundreds of millions of rows | About 10 ms per 300 | Works through content and context features | Medium | Strong on dense, mixed features; quick to iterate |
| Multi-task deep model with sequence features | Heavy | 30 to 50 ms | Best with content embeddings | Low | Several heads (tap, visit, satisfied); models session intent |

**Decision.** Geo index plus two-tower retrieval, GBDT ranker first, multi-task deep model later. The demo shows why the structure matters more than the model family: taste embedding alone gets 25% recall, adding distance gets 37%, adding context gets 47%.

**Circle back.** The objective is satisfying visits, and visits are dominated by reachability and timing. A model that only learns taste optimizes the wrong thing. The ranker must take distance and context as first-class inputs and output calibrated probabilities so the heads can be combined.

## 7. Training

- **Split by time.** Train the embedding on the oldest window, the ranker on a later window using embeddings from the earlier one, validate and test on the most recent. This also avoids the ranker over-trusting an embedding score that has seen its labels.
- **Losses.** Embeddings: BPR or sampled softmax with geo-matched negatives and a popularity correction. Ranker: log loss per head.
- **Negatives.** For the ranker, about 5 to 10 nearby shown-but-not-visited places per positive.
- **Cold-start users.** Onboarding picks, device locale, and the average user in that area and time. Move to personal signals after a handful of interactions.
- **Cold-start places.** Content tower (category, price, photos, text), the owner's other locations, and neighbouring places' audiences.
- **Cadence.** Ranker daily, embeddings weekly with daily inference for new places, context lifts weekly, hours and closures in real time.

## 8. Evaluation

**Offline**
- Recall@K of retrieval separately from NDCG@10 of ranking. If the visited place was not in the candidates, no ranker can fix it.
- Slices: new places, light users, users away from home, dense and sparse areas, time of day.
- Counterfactual estimate from the randomized slice using inverse propensity weighting, since ordinary logs favour the old ranker.
- Beyond accuracy: coverage, share of impressions to the long tail, intra-list diversity.

**Online**
- A/B test randomized by user. Primary: satisfying visits per session. Secondary: taps, directions, saves.
- Guardrails: closed-place rate, average distance, hide rate, coverage, new-place exposure, latency.
- Interference is mild on the user side, but place capacity is shared: if treatment sends everyone to the same cafe, it fills up. For capacity-limited or sponsored inventory, randomize by area and time.
- Duration: two weeks minimum to cover weekday and weekend patterns. Novelty runs both ways: exploration looks bad in week one and pays off once new places have data.
- Long-term holdout to measure whether diversity and exploration improve retention.

## 9. Debugging, monitoring, deployment

- **Offline up, online flat.** Check candidate recall in production (is the radius too small in sparse areas), feature skew between training and serving (time zone and hours bugs are common), and whether the gain was only on places the old system already showed.
- **Drift.** Seasonality, holidays, weather and events shift context features. Monitor score calibration by hour and by city. A new city launch is a cold start for everything at once.
- **Data freshness.** Track the rate of suggested places that turn out closed. Alert on stale hours data.
- **Feedback loop.** Monitor exposure concentration (Gini) and coverage. Rising concentration with flat satisfaction means the loop is tightening.
- **Location quality.** Coarse or stale location silently degrades everything. Log location accuracy and slice metrics by it.
- **Deployment.** Shadow scoring, then 1% canary by city, automatic rollback on closed-place rate and latency, staged ramp. Keep popularity-nearby as a fallback when the ranker times out.

## 10. Scaling and advanced topics

- **Geo indexing.** Hierarchical cells (geohash, S2, H3). Query the cells covering the radius, then exact-filter by distance. Adapt the radius to density: aim for a fixed candidate count, not a fixed distance. Hexagonal cells have uniform neighbour distance, which helps aggregation.
- **ANN with a geo constraint.** Either shard the vector index by coarse cell and search neighbouring shards, or retrieve by geo first and score the embedding only on those candidates. The second is simpler and usually enough.
- **Travel time, not straight-line distance.** A river or a motorway makes 500 m take 20 minutes. Use precomputed travel-time cells.
- **Diversity.** Maximal marginal relevance: pick the next item by `lambda × relevance - (1 - lambda) × max similarity to items already chosen`. Tune lambda online.
- **Exploration.** Reserve a slot, or add an uncertainty bonus that shrinks as `1 / sqrt(impressions)`. Log the propensity so the data can be reweighted.
- **Multi-objective.** User satisfaction, place-side fairness (new and small businesses), and possibly revenue. Combine in the score with weights set by experiment, with caps as hard rules.
- **Privacy.** Coarsen location on the device, keep history on the device where possible, short retention, no inference of sensitive places (clinics, places of worship) into profiles. The demo shows the accuracy cost of coarser location.
- **10x and beyond.** Cache results per cell and context bucket for anonymous or cold users; personal re-rank on top. Precompute place-side features.

## 11. Mapping to the demo

| Demo element | Concept |
|---|---|
| Map: blue cells and dashed circle | Grid index retrieval, radius query, exact filter |
| Serving funnel | Retrieve, filter, score, rank, re-rank, with counts and latency |
| Ranking model switch | Popularity vs taste vs taste + distance vs full ranker |
| Coloured bar on each result | Per-group contribution: taste, distance, popularity, context |
| Time, weekend, rain controls | Context features changing the ranking |
| Diversity slider | MMR re-ranking |
| Explore switch and yellow rings | Cold-start places, uncertainty bonus, random exploration |
| Brand-new user and category chips | Cold-start user and onboarding |
| Location precision slider | Privacy against accuracy |
| Satisfaction bars and dashed line | Gap between what is optimized (visits) and the real goal |
| Section 2 charts | Offline metrics by model and by slice; retrieval recall as a ceiling; feature importance |
| Distance bias panel | Negative sampling choice and what the embedding learns |
| Position bias panel | Randomized traffic and inverse propensity weighting |
| Feedback loop panel | Counts vs corrected rates, exploration, coverage |
| Split panel | Random vs time-based evaluation |

## 12. Follow-up questions

<details>
<summary>Why not use collaborative filtering alone?</summary>
It learns who likes what but not where or when. In the demo the taste embedding alone matches a nearby-popularity baseline at 25% recall@10, and distance plus time features lift it to 47%. It also has nothing to say about new places or new users. Use it as a retrieval source and a ranker feature.
</details>

<details>
<summary>How do you choose the search radius?</summary>
Not as a constant. In a dense centre 500 m holds thousands of places; in a suburb 5 km holds twenty. Expand rings of cells until a target candidate count is reached, cap by a maximum travel time, and let the ranker's distance feature do the fine trade-off. Also adapt to the user: their typical trip length and mode of travel.
</details>

<details>
<summary>A new restaurant opened yesterday. How does it get recommended?</summary>
Content features (category, price, photos, menu text) give it an embedding through the place tower. Priors come from similar places nearby. Then it needs exposure: an exploration slot or an uncertainty bonus, limited to users whose taste matches, with the propensity logged. Track impressions to places under 30 days old as a guardrail.
</details>

<details>
<summary>How do you sample negatives for the embedding model, and why does it matter?</summary>
From the same neighbourhood as the positive. With citywide negatives, most are far away, so the model separates them by geography and the embedding becomes a map of where people live. In the demo that model's score correlates 0.45 with closeness to home and 0.08 with true taste, and its recall away from home falls to 18%. Geo-matched negatives remove the shortcut.
</details>

<details>
<summary>The user is travelling in a city they have never visited. What changes?</summary>
Detect it (distance from home cluster). Their local history is empty but their taste embedding still transfers through the place tower. Shift weights toward quality, landmarks and tourist-relevant categories, relax the distance penalty, and lean less on "previously visited". Evaluate this slice separately; it is where personalization pays most.
</details>

<details>
<summary>Taps went up 6% in the A/B test but visits are flat. What do you conclude?</summary>
The model learned what gets tapped, not what gets visited: likely attractive photos or places that are interesting but too far or closed. Check distance of tapped places, open-now rate, and tap-to-visit conversion by slot. Reweight the objective toward the visit head and add distance and hours features to the tap model. Do not ship on taps alone.
</details>

<details>
<summary>How would you handle position bias in the logs?</summary>
Log the position. Run a small share of traffic with the top results shuffled to measure how attention falls with rank. Then either weight each click by the inverse of that attention when computing rates and training, or add position as a feature and fix it at serving. The demo's correction raises top-20 overlap with the truly best places from 30% to 70%.
</details>

<details>
<summary>Everyone is being sent to the same five places. How do you fix it?</summary>
That is the popularity loop. Replace visit counts with position-corrected rates, add exploration, apply diversity re-ranking, and add per-place impression caps when capacity matters. Monitor exposure concentration and catalogue coverage. In the demo coverage goes from about a third to over 90% while click satisfaction rises.
</details>

<details>
<summary>What if you only get the user's location to the nearest kilometre?</summary>
Distance features become noisy, so the ranker should lean more on taste and context and less on fine distance ordering. Widen the radius, prefer travel-time bands over exact distance, and let the device do the final distance sort locally with its exact position. Measure the accuracy cost so the privacy decision is informed.
</details>

<details>
<summary>Why GBDT for ranking here rather than a deep model?</summary>
The decisive features are dense and tabular: distance, lifts, rates, price gap. GBDT handles them with little tuning, trains in minutes and is easy to inspect. A deep model becomes worthwhile for multiple heads, session sequences and raw content such as photos and text.
</details>

<details>
<summary>Metric dropped 8% on Monday with no deploy. Where do you look?</summary>
External causes first: holiday, weather, a large event, a data provider outage for hours or closures. Then pipelines: context lift tables not refreshed, location accuracy degraded after an app or OS update, a feature defaulting to null. Slice by city, platform and hour to localize before touching the model.
</details>

<details>
<summary>How do you add sponsored places without hurting the experience?</summary>
Keep organic ranking unchanged and run a separate auction for marked sponsored slots, scored by bid times predicted visit probability with a relevance floor and the same distance and open-now filters. Measure organic engagement as a guardrail.
</details>

## Common mistakes

- Treating it as generic collaborative filtering and adding location as an afterthought.
- A fixed radius everywhere.
- Raw latitude and longitude as model features.
- Sampling negatives citywide, so the model learns home location.
- Visit counts as a popularity feature, with no correction for exposure or position.
- No plan for new places, new users or travellers.
- Recommending closed places because hours were not a hard filter.
- Random train/test split.
- Optimizing taps only; no satisfaction signal and no guardrails on distance or diversity.
- Ignoring location privacy.
