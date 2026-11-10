# ML Systems Practice Lab

Runnable demos and in-depth interview guides for seven ML system design problems.
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
| 1 | Friend recommendation | done | done | done |
| 2 | Ride dispatch matching | drafted, untested | - | - |
| 3 | Nearby place recommendations | drafted, untested | - | - |
| 4 | Ad ranking in a feed | done | done | done |
| 5 | Testing a new ranker | drafted, untested | - | - |
| 6 | Text language detection | drafted, untested | - | - |
| 7 | Fake account detection | drafted, untested | - | - |

## Layout

- `backend/main.py` discovers every module in `backend/demos/`
- `web/<slug>/index.html` is the demo page, `guides/<slug>.md` the interview guide
- `CONTRACT.md` is the spec each module follows
