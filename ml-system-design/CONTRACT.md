# Module contract (read fully before writing code)

Each module = 3 files, nothing else is touched:

1. `backend/demos/<mod>.py`  (python module name uses underscores)
2. `web/<slug>/index.html`   (+ optional `web/<slug>/app.js`)
3. `guides/<slug>.md`

Slugs/modules: friend-recommendation/friend_recommendation, driver-matching/driver_matching, place-recommender/place_recommender,
ads-ranking/ads_ranking, ranking-experiment/ranking_experiment, language-classifier/language_classifier, bot-detection/bot_detection.

## Backend
- `META = {"slug": "<slug>", "title": "...", "tagline": "one line", "icon": "emoji", "order": N}`
- `router = APIRouter()`; main.py mounts it at `/api/<slug>`. No other files/shared state; import only numpy, scikit-learn, stdlib, fastapi/pydantic.
- Synthetic data generated deterministically (fixed seed) at import or lazily on first request; whole module must load in < 5 s and every endpoint respond in < 2 s (small data: thousands of rows, not millions).
- Use REAL models/algorithms (sklearn, or numpy implementations written by hand where the algorithm itself is the lesson), not hard-coded fake outputs. Expose the *internals* that teach: features used, scores per stage, candidate counts, latency per stage, metrics.
- Provide a `GET /overview` returning dataset stats + model/metric summary, plus 3-5 interactive POST/GET endpoints that let the UI poke the system (e.g. recommend, change a knob, inject drift, run an A/B).
- Include at least one endpoint that demonstrates a **pitfall** interviewers ask about (leakage, position bias, feedback loop, drift, class imbalance, novelty effect, SRM, ...) with a toggle to show the wrong vs right way.
- Keep code readable: comments at the top of each section say which interview concept it illustrates.

## Frontend
- Vanilla HTML/JS, no build step, no CDN. Use `../shared/style.css` and `../shared/common.js` (read both first; use `api(slug, path, body)`, `kpis`, `table`, `barChart`, `lineChart`, `md`, tabs).
- Page template (copy exactly, fill in):
```html
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TITLE</title><link rel="stylesheet" href="../shared/style.css"></head><body>
<header class="top"><a href="../">&larr; Lab</a><h1>ICON TITLE</h1></header>
<main>
<div class="tabs"><div class="tab" data-pane="demo">Live demo</div><div class="tab" data-pane="guide">Interview guide</div></div>
<div class="pane" id="demo"> ...interactive UI... </div>
<div class="pane" id="guide">Loading...</div>
</main>
<script src="../shared/common.js"></script>
<script>const SLUG='<slug>'; loadGuide(SLUG, $('#guide')); /* demo code or <script src="app.js"> */</script></body></html>
```
- Each demo must have short "What to notice" callouts (`<div class="note">`) tying what's on screen to an interview talking point.

## Guide (`guides/<slug>.md`) - rendered by shared `md()` (supports #/##/###, **bold**, `code`, ``` fences, lists, tables, > quotes, `<details>`)
Write at the level of a senior/staff ML engineer interview at a large tech company. Required sections, in this order (## headings):
1. **Problem statement & how to open** - the exact prompt, then the clarifying questions to ask (grouped: product, scale, constraints, data) with *why each matters* and typical answers.
2. **Objectives & metrics** - 2-3 framing options with trade-offs, chosen one, and offline/online/guardrail metrics.
3. **High-level architecture** - ASCII diagram (in a ``` fence) + latency/QPS budget numbers.
4. **Data & labels** - sources, positives/negatives, sampling, bias, label delay, leakage traps.
5. **Feature engineering** - table: feature | type | encoding | why | freshness. Include 2-3 creative features and at least one with an explicit formula. (The interviewer weights this heavily.)
6. **Modeling** - baseline + 2-3 models with trade-off table (scalability, latency, cold start, interpretability); the decision and circle-back to the objective.
7. **Training** - data splits (time-based!), losses, negative sampling, imbalance, retraining cadence.
8. **Evaluation** - offline AND online (A/B design, guardrails, sample size intuition, novelty effects).
9. **Debugging, monitoring, deployment** - drift, logging skew, feedback loops, slices, shadow/canary/rollback.
10. **Scaling & advanced topics** - what changes at 10x / 100x; multi-objective; extensibility.
11. **Mapping to the demo** - which demo endpoint/UI element demonstrates which concept above.
12. **Follow-up questions** - at least 10, each as `<details><summary>Question</summary>` followed by a concise strong answer (each answer on following lines, then `</details>`). Mix: probing ("why not X?"), twist ("now cold-start / 100x scale / privacy constraint"), debugging ("metric dropped 5% after launch, what do you do?"), and a "common mistakes" list at the end (## Common mistakes).
Keep it dense and concrete: real numbers, formulas, named techniques. No filler.

## Done checklist (verify before finishing)
- `python -c "import backend.demos.<mod>"` from ml-system-design/ works; hit every endpoint with fastapi TestClient and confirm 200 + sensible JSON.
- HTML references only `../shared/*` and local files; JS has no console errors (check by reading carefully; no browser available).
- Guide has all 12 sections + Common mistakes.
Report back in <=8 lines: endpoints, models used, and anything incomplete.
