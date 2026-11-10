# Ad Ranking in a Feed: interview guide

> The one-line answer: an ads system is a prediction problem feeding an auction. Retrieve eligible ads by targeting, predict calibrated probabilities (click, conversion, negative feedback), rank by expected value per impression (bid × pCTR plus quality terms), charge by a second-price rule, and pace budgets. The interview is won on calibration, position bias, feature crosses, freshness and multi-objective trade-offs, not on naming a deep model.

## 1. Problem statement and how to open

**Prompt.** "Design the ranking of sponsored posts in a social feed."

Spend the first 5 minutes on questions. Then say: "I will cover the objective, the funnel, features, the click model, the auction, and evaluation. I can go deeper on features and modeling, on the auction and pacing, or on evaluation. Which would you like?"

**Product**
- Where do ads appear and how many? Feed slots at fixed positions, or does ranking decide ad load? *Decides whether you rank ads against ads only, or ads against organic posts.*
- How do advertisers pay: per impression (CPM), per click (CPC), per conversion (CPA / oCPM)? *Decides which probabilities you must predict.* Assume a mix, normalized to value per impression.
- What does the advertiser optimize for: reach, clicks, installs, purchases?
- What negative signals exist? Hide ad, report, "why am I seeing this", fast scroll-past.

**Scale**
- About 2B daily users, 10M active ads, tens of billions of ad impressions a day, peak a few hundred thousand QPS.
- Per request: thousands of eligible ads after targeting, 1 to 5 slots.

**Constraints**
- Latency: 100 to 150 ms end to end for the ads call, of which final ranking gets 20 to 30 ms.
- Freshness: new ads and creatives appear every minute and CTR drifts within hours.
- Policy: ad review, frequency caps, restricted categories, privacy limits on targeting and on conversion data (aggregated or delayed reporting).

**Data**
- Impression logs with position, clicks, conversions through pixel or SDK with delay, hide and report events, user profile and activity, ad creative, advertiser history, bids and budgets.

## 2. Objectives and metrics

| Option | Rank by | For | Against |
|---|---|---|---|
| A. Revenue | bid only, or expected payment | Simple, short-term money | Shows expensive irrelevant ads; users and advertisers leave |
| B. Clicks | pCTR only | Relevant feed | Ignores advertiser value; rewards clickbait; no price signal |
| C. Total value | bid × P(action) + user value | Aligns platform, advertiser and user; standard practice | Needs calibrated predictions and tuned weights |

**Choice: total value per impression.**

`score = bid × pCTR × pCVR (for conversion bidders) + w_click × pCTR - w_hide × pHide`

The first term is advertiser value (eCPM), the rest is estimated user value in the same currency. This is efficient (the slot goes to whoever values it most), and with second-price charging it keeps bidding honest. Revenue is an outcome, not the objective.

**Offline metrics:** normalized entropy, calibration (predicted / actual) overall and per slice, log loss, AUC as a secondary check.
**Online primary:** advertiser value delivered and revenue per 1,000 impressions, together.
**Guardrails:** hide and report rate, organic engagement and sessions, ad load, advertiser cost per conversion and budget delivery, latency, calibration drift.

**Normalized entropy** is log loss divided by the entropy of the background CTR p:

`NE = LogLoss / ( -(p × log p + (1 - p) × log(1 - p)) )`

Below 1 means better than predicting the average. It is comparable across datasets with different CTR, which raw log loss is not.

## 3. High-level architecture

```
request (user, context)
  -> targeting / retrieval: inverted index on audience, geo, language   10M -> ~10k   (10 ms)
     + embedding retrieval (user tower . ad tower) for broad targeting
  -> filters: budget left, pacing throttle, frequency cap, policy, blocklists    -> ~3k
  -> light ranker: two-tower or small LR, cheap features                 -> ~300     (10 ms)
  -> heavy ranker: multi-task model -> pCTR, pCVR, pHide per ad          -> ~300     (25 ms)
  -> calibration layer per head (isotonic / Platt, refreshed hourly)
  -> auction: score = bid x pCTR x pCVR + quality terms; sort; GSP or VCG price
  -> merge with organic feed (ad load rules, spacing, diversity)         -> 1-5 ads
  -> log: features AS SERVED, position, score, price  --> training stream

training: impression stream --join (click window 30 min; conversion window 7 d)--> examples
          -> online / hourly incremental training -> validation gates -> canary -> serve
```

**Budget.** 100 to 150 ms total. The heavy ranker scores about 300 ads in 25 ms, so under 0.1 ms per ad. User-side computation is done once and shared across candidates.

## 4. Data and labels

- **Click label:** impression joined to a click within a short window (minutes). Impressions must be viewable; discard ones that never rendered.
- **Conversion label:** click joined to a conversion within 1 to 7 days. This is delayed, see section 7.
- **Negative feedback label:** hide, report. Rare (about 2%), so needs its own head and weighting.
- **Negatives:** non-clicked viewable impressions. Downsample at keep rate w and correct at serving: `q = p / (p + (1 - p) / w)`. Without the correction every prediction is inflated, and since price is proportional to pCTR, advertisers are overcharged.
- **Position bias:** slot 1 is seen by nearly everyone, slot 5 by a third. Logged CTR is ad quality times slot visibility. Train with position as a feature and score all candidates at a fixed position, or factor the model as P(click) = P(seen at position) × P(click if seen).
- **Selection bias:** you only observe clicks for ads the previous model chose. Keep a small exploration budget (random or boosted new ads) and log the propensity.
- **Leakage traps:** features computed after the impression (same-session click counts that include the labeled click), ad-level CTR aggregates that include the current event, conversion features that arrive later than serving time. Log features at serving and train from that log.
- **Fraud and invalid clicks:** filter bot and accidental clicks before training and before billing.

## 5. Feature engineering

| Feature | Type | Encoding | Why | Freshness |
|---|---|---|---|---|
| User ID, ad ID, advertiser ID, campaign ID, creative ID | sparse ID | hashed, embeddings (16 to 64 dim) | Memorization of specific behavior | Embeddings updated online |
| User interest × ad category | cross | hashed cross, or learned by the model | The main relevance signal | Daily |
| Ad historical CTR, at ad / campaign / advertiser level | dense | smoothed rate, bucketed; windows 1 h, 1 d, 7 d | Strong prior; back-off levels cover new ads | Minutes |
| User historical CTR on ads, by category | dense | smoothed rate | Some users never click ads | Hourly |
| User × advertiser history | cross | counts of impressions, clicks, conversions, last 30 d; time since last | Retargeting and fatigue | Real time |
| Recent activity sequence | sequence | last 50 engaged items, pooled or attention | Current intent | Real time |
| Creative content | dense | text and image embeddings from pretrained encoders | Cold start for new creatives | At creation |
| Context: device, OS, connection, hour, day, surface | categorical | one-hot or small embedding | Mobile on cellular converts differently | Request time |
| Position | categorical | one-hot in training, fixed at serving | Removes position bias | Request time |
| Frequency: impressions of this ad to this user in 1 d, 7 d | dense | bucketed | Each repeat is worth less | Real time |
| Landing page quality: load time, bounce rate | dense | bucketed | Predicts conversion and bad experience | Daily |
| Advertiser trust: account age, report rate, policy strikes | dense | bucketed | Predicts hides and reports | Daily |
| Budget state: share of daily budget spent | dense | float | Used by pacing, not by pCTR | Real time |

**Formulas worth writing down**

Smoothed CTR, so an ad with 1 click in 2 impressions does not look like 50%:

`ctr_smoothed = (clicks + a × prior_ctr) / (impressions + a)`, where prior_ctr is the campaign or category average and a is about 100 to 1,000 pseudo-impressions

Time-decayed counts, so recent behavior counts more:

`count_t = count_(t-1) × exp(-dt / half_life) + new_events`

Fatigue: `fatigue = log(1 + impressions of this advertiser to this user in 7 days)`

**Encoding notes.** High-cardinality IDs use the hashing trick or embedding tables. Hash size trades memory for collisions, and collisions mostly hurt rare IDs. Counts are log-scaled or quantile-bucketed. Real-time counters live in a streaming aggregation service; batch features in a feature store with point-in-time reads so training matches serving.

## 6. Modeling

**Baseline:** historical smoothed CTR per ad, backed off to campaign and category. No personalization, but it sets the bar and is the fallback when the model is down.

| Model | Scalability | Latency | Cold start | Interpretability | Notes |
|---|---|---|---|---|---|
| Logistic regression on hashed crosses (FTRL) | Billions of features, online updates | Microseconds | Poor without content features | High | Needs hand-built crosses; freshness is excellent |
| GBDT | Good on dense features, batch only | Milliseconds | Same | Medium | Learns crosses; bad at high-cardinality IDs and cannot update online |
| GBDT leaves into LR (He et al., 2014) | GBDT daily, LR online | Low | Same | Medium | Trees do feature transforms, LR on top stays fresh |
| Factorization machine | Good | Low | Generalizes to unseen crosses | Medium | Pairwise interactions through embeddings |
| Deep model: Wide and Deep, DeepFM, DLRM, multi-task (MMoE) | GPU training, large embedding tables | 10 to 30 ms for hundreds of ads | Best, with content embeddings | Low | State of practice; shared embeddings across click, conversion and hide heads |

**Decision.** Light ranker: two-tower model so ad embeddings are precomputed. Heavy ranker: multi-task deep model with shared embeddings and separate heads for click, conversion and hide, followed by a per-head calibration layer. If the team is starting from scratch, ship LR with hashed crosses or the GBDT-plus-LR hybrid first, because it trains online and is easy to debug.

**Circle back to the objective.** The score multiplies predictions by bids and adds them across heads, so each head must output a calibrated probability. That rules out pure ranking losses and makes log loss, NE and calibration the model-selection criteria. It also means the freshest reasonably good model beats a slightly more accurate stale one.

**Conversion modeling.** Conversions happen only after clicks, so training pCVR on clicks alone and applying it to all impressions is a sample-selection problem. Model the whole space: predict pCTR and pCTCVR = pCTR × pCVR on all impressions with shared embeddings (ESMM).

## 7. Training

- **Split by time.** Train on days 1 to N, calibrate on the most recent slice, test on the following day. Report NE on the next day's data, since that is the serving condition.
- **Loss:** log loss per head; multi-task loss is a weighted sum.
- **Downsampling:** keep 1 to 10% of negatives, correct with the formula in section 4, or use importance weights.
- **Online learning:** stream impressions, wait a short window for the click, emit the example, update the model continuously or every 15 to 60 minutes. Data freshness is worth real NE: a model a day old is measurably worse than one an hour old.
- **Delayed conversions:** a click with no conversion yet is not a negative. Options: wait the full window (accurate, stale), train now and correct later when the conversion arrives (emit a positive that cancels the earlier negative), or weight by the probability that a conversion would have arrived by now (delayed feedback model).
- **New ads:** initialize from campaign and advertiser embeddings and content features; add an exploration bonus so they collect data.
- **Cadence:** embeddings and deep layers updated incrementally through the day, full retrain weekly, calibration layer refit hourly.

## 8. Evaluation

**Offline**
- NE and calibration on the next day, overall and per slice: new vs old ads, country, device, advertiser size, position, pCTR bucket.
- Reliability curve. A model can be calibrated on average and wrong at both ends.
- Auction replay: rerun logged auctions with the new model's predictions to estimate changes in winners, revenue and advertiser value. Biased toward the old policy, so use it as a sanity check.

**Online**
- A/B test with users randomized. Metrics: revenue, advertiser value, CTR, conversion rate, cost per conversion, hides, organic engagement.
- **Budget interference.** Treatment and control spend from the same advertiser budgets. If treatment spends budgets faster, control gets less, and the measured lift is inflated. A model that only moves spend earlier in the day shows a revenue gain in the test and none at 100%. Fix with budget-split experiments: split each advertiser's budget in proportion to traffic so each arm is its own marketplace.
- **Duration.** At least one to two weeks to cover weekly cycles, advertiser reactions (they adjust bids) and conversion delay.
- **Long-term holdout.** A small group with fewer ads or the old model, held for months, to measure the effect of ad quality on sessions.
- **Sample size intuition.** CTR near 5% with billions of impressions detects 0.1% relative changes within days. Conversion and hide metrics are 10 to 100 times sparser and decide the test length.

## 9. Debugging, monitoring, deployment

- **Calibration is the first alarm.** Monitor predicted / actual in real time, overall and per slice. A jump means a feature pipeline broke or the traffic mix changed.
- **Winner's curse.** Predicted CTR of winning ads is systematically above actual, because the auction selects positive errors. Track calibration on served impressions separately from all candidates.
- **Training and serving skew.** Log features as served; compare distributions with the training pipeline. Common causes: a counter with a different window, a default value for missing features, time zone bugs.
- **Feedback loop.** Ads that are shown get data and better estimates; ads never shown stay uncertain. Without exploration, incumbents win forever.
- **Revenue dropped after a launch.** Check in order: calibration by slice, auction density (candidates per request, did a filter change), bid and budget landscape (did a large advertiser pause), pacing multipliers, latency timeouts that drop the heavy ranker, then the model.
- **Deployment.** Offline gates (NE and calibration not worse on any major slice), shadow scoring, 1% canary with automatic rollback on calibration or revenue, staged ramp. Keep the previous model loaded for instant rollback. Calibration layer and model are versioned together.

## 10. Scaling and advanced topics

- **Auction mechanics.** Generalized second price: each winner pays the minimum bid that keeps its slot. With quality terms, `price_i = (score_next - user_value_i) / pCTR_i`. VCG charges each winner the harm it does to others and is truthful with multiple slots; some feed platforms use VCG, search engines historically GSP. Reserve prices set a floor.
- **Pacing.** A controller per campaign adjusts a multiplier so spend tracks a target curve. Throttling skips auctions; bid shading lowers the bid and wins cheaper impressions. Shading gives more results per dollar.
- **Auto-bidding.** Advertisers give a goal (cost per conversion). The system sets `bid = target_cost × pCVR` per impression, so prediction quality directly sets the bid.
- **Multi-objective.** Weights such as w_hide are set by long-term experiments that measure how hides and ad load change sessions, then converted into currency.
- **Ads and organic together.** Convert organic value into the same currency and let ads compete for positions under ad-load and spacing constraints.
- **10x to 100x.** Shard embedding tables, quantize, cache user-tower output, cascade more aggressively (cheaper models filter more), and precompute ad-side features.
- **Privacy.** Less user-level conversion data means aggregated or delayed labels. Use modeled conversions, on-device signals and calibration against aggregate reports.
- **Extensibility.** Feature registry with ownership, shared logging schema, model and calibration versioned together, a new head added without changing the auction formula.

## 11. Mapping to the demo

| Demo element | Concept |
|---|---|
| Section 1, model table | LR, GBDT, FM and the GBDT-to-LR hybrid; NE and calibration vs AUC; scoring cost |
| Section 1, CTR by position | Logged CTR mixes quality with visibility |
| Section 2, hash bits | Hashing trick: collisions vs memory |
| Section 3, reliability curve | Calibration layer; AUC unchanged, NE improved |
| Section 4, auction simulator | bid × pCTR ranking, second-price charging, quality terms, empty slots |
| Section 5, trade-off sweep | Multi-objective ranking; revenue against hides; bid-only and pCTR-only baselines |
| Section 6, pacing | Budget exhaustion, throttling vs bid shading |
| Section 7, position bias | Position as a training feature, fixed at serving |
| Section 7, miscalibration | Winner's curse; what a global calibrator can and cannot fix |
| Section 7, downsampling | The correction formula and why AUC hides the bug |
| Section 7, delayed conversions | Wait, naive, and reweighting strategies |

## 12. Follow-up questions

<details>
<summary>Why rank by bid × pCTR and not just by bid?</summary>
An advertiser paying per click generates value only when a click happens, so the expected value of an impression is bid × pCTR. Ranking by bid alone gives slots to expensive ads nobody clicks: in the demo it has 3.2% CTR and the lowest advertiser value, against 5.3% for bid × pCTR. It also lets advertisers buy placement with irrelevant ads, which hurts users.
</details>

<details>
<summary>Why do you care so much about calibration when AUC is the same?</summary>
The prediction is used as a number, not only as an ordering. It is multiplied by the bid to compare ads with different bids, it sets the price, and for conversion-optimized campaigns it sets the bid itself. A model 25% high charges advertisers 25% too much per result. AUC is blind to this because any monotone transform leaves it unchanged.
</details>

<details>
<summary>A new ad has no history. How does it ever get shown?</summary>
Prediction falls back to campaign, advertiser and category priors plus creative content embeddings. Then explore: add an uncertainty bonus (upper confidence bound) or Thompson sampling to the score for the first few thousand impressions, funded by a small exploration budget. Without it, the system only learns about ads it already likes.
</details>

<details>
<summary>You downsample negatives 10 to 1. What breaks and how do you fix it?</summary>
Predicted odds are multiplied by 10, so average pCTR goes from about 7% to about 40%. Ranking within one model is unchanged, but pricing, budget pacing and any mixing with other heads break. Apply q = p / (p + (1 - p) / w) at serving, or train with importance weight 1 / w on negatives.
</details>

<details>
<summary>How do you handle position bias?</summary>
Add position as a feature in training and set it to one fixed value for all candidates at serving, so the model compares ads as if they were in the same slot. A cleaner version factors click probability into examination (depends on position only) and relevance. To estimate examination without confounding, randomize the order of a small share of traffic.
</details>

<details>
<summary>Revenue is up 3% in the A/B test. Is that real?</summary>
Check for budget interference first. Both arms draw from the same campaign budgets, so if treatment spends faster it takes budget from control and the lift disappears at full launch. Rerun as a budget-split test. Then check that advertiser value and cost per conversion did not worsen, hides and organic engagement are flat, and that the lift is not just miscalibration raising prices.
</details>

<details>
<summary>Predicted over actual CTR jumped from 1.02 to 1.30 this morning. What do you do?</summary>
Roll back if a model or feature change went out. Otherwise slice the ratio by country, device, advertiser and feature buckets to localize. Usual causes: a real-time counter stopped updating so features are stale or defaulted, a logging change dropped clicks (the labels are wrong, not the model), or a traffic shift such as a holiday or a large new campaign. The hourly calibration refit limits the damage while you find the cause.
</details>

<details>
<summary>Why not train one model directly on revenue?</summary>
Revenue per impression is price times click, and price depends on the other bidders and on the model's own output, so the target moves when the model changes. It also mixes advertiser bids, which change by the minute, into a learned function. Predicting user behavior (click, convert, hide) and combining with live bids in a formula keeps the model stable and the auction transparent.
</details>

<details>
<summary>How do you include conversions that arrive a week later?</summary>
Do not treat "no conversion yet" as a negative. Emit the example at click time, then emit a correction when the conversion arrives, or weight recent negatives by the probability the conversion is still to come, learned from the delay distribution per advertiser. In the demo, naive training under-predicts conversion rate by 15%, and by 22% for slow-converting advertisers.
</details>

<details>
<summary>How would you add a "user dislikes this ad" objective?</summary>
Add a head predicting hide or report, calibrate it, and subtract w_hide × pHide from the score. Set the weight from a long-running experiment that measures how hides reduce future sessions and therefore future revenue. The demo shows a penalty of 0.5 cutting hides by about 27% for about 3.5% revenue.
</details>

<details>
<summary>GBDT or deep network for pCTR?</summary>
GBDT is strong on dense counters and trains fast, but cannot handle billions of sparse IDs or update online. Deep models with embedding tables handle IDs, sequences and multiple tasks. A practical path is LR or GBDT-plus-LR first for a debuggable, fresh baseline, then a multi-task deep model once the feature and logging pipelines are trusted.
</details>

<details>
<summary>Twist: advertisers now bid per conversion, and conversions are 100 times rarer than clicks.</summary>
Predict pCTR and pCVR and rank by bid × pCTR × pCVR. Train pCVR over all impressions jointly with pCTR (ESMM) to avoid selection bias and share embeddings so the sparse task borrows from the dense one. Calibration of pCVR per advertiser matters more, and so does delayed-label handling and exploration, since each campaign has few positives.
</details>

## Common mistakes

- Ranking by pCTR only and forgetting bids, or by bid only and forgetting relevance.
- Reporting AUC and never mentioning calibration or log loss.
- No candidate retrieval stage; proposing to score 10M ads per request.
- Ignoring position bias and the fact that training data comes from the old ranker.
- Downsampling negatives without the correction.
- Treating not-yet-converted clicks as negatives.
- No story for new ads or new advertisers.
- A/B test design that ignores shared budgets.
- No user-experience term or guardrail, so the design maximizes short-term revenue.
- Forgetting pacing, frequency caps and policy filters, which sit around the model and decide much of what users see.
