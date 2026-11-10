"""ML System Design Interview Lab - FastAPI shell.

Auto-discovers every module in backend/demos/*.py. Each module must expose:
  META   = {"slug": str, "title": str, "tagline": str, "icon": str}
  router = fastapi.APIRouter()   (mounted at /api/<slug>)
Guides are served from ../guides/<slug>.md ; frontend from ../web.
Run:  uvicorn backend.main:app --reload   (from ml-system-design/)
"""
import importlib
import pkgutil
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="ML System Design Interview Lab")
MODULES: list[dict] = []

from . import demos  # noqa: E402

for info in pkgutil.iter_modules(demos.__path__):
    mod = importlib.import_module(f"{demos.__name__}.{info.name}")
    if hasattr(mod, "router") and hasattr(mod, "META"):
        app.include_router(mod.router, prefix=f"/api/{mod.META['slug']}")
        MODULES.append(mod.META)
MODULES.sort(key=lambda m: m.get("order", 99))


@app.get("/api/modules")
def modules():
    return MODULES


@app.get("/api/guide/{slug}", response_class=PlainTextResponse)
def guide(slug: str):
    p = ROOT / "guides" / f"{Path(slug).name}.md"
    if not p.exists():
        raise HTTPException(404, "guide not found")
    return p.read_text(encoding="utf-8")


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
