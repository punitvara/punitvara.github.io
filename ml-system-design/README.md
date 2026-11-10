# ML System Design Interview Lab

Runnable demos and FAANG-level interview guides for seven classic ML system design questions.
Each module has a FastAPI backend with real models on synthetic data, an interactive page, and a guide
with clarifying questions, features, model trade-offs, evaluation, debugging and follow-up questions.

## Run

```
cd ml-system-design
./run.sh            # installs requirements, serves http://localhost:8000
```

The demos need the Python backend, so run them locally. The landing page and all interview guides are static, so they work on GitHub Pages at `/ml-system-design/` (guides load from `guides/*.md`; the demo tab shows a note explaining how to run it).

## Status

| # | Question | Backend | Demo page | Guide |
|---|---|---|---|---|
| 1 | People You May Know | done | done | done |
| 2 | Matching drivers with customers | drafted, untested | - | - |
| 3 | Place / location recommender | drafted, untested | - | - |
| 4 | Ads ranking for the news feed | done | done | done |
| 5 | Ranking experiment (A/B testing) | drafted, untested | - | - |
| 6 | Language classifier | drafted, untested | - | - |
| 7 | Account bot detection | drafted, untested | - | - |

## Layout

- `backend/main.py` discovers every module in `backend/demos/`
- `web/<slug>/index.html` is the demo page, `guides/<slug>.md` the interview guide
- `CONTRACT.md` is the spec each module follows
