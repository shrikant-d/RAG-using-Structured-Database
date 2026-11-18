"""Local HTTP API + static host for the web UI (replaces the Streamlit app.py).

Run:  python -m webui.server         (from the repo root; serves webui/frontend/dist on :8600)

Thin wrapper only: every step below is the call app.py made -- build_runner,
execute_with_correction, answer_from_result, check_grounding -- with the same
MAX_RETRIES and display cap. Nothing in the evaluated pipeline changes. API keys
stay in this process (core.llm reads .env); responses carry no secrets.
Binds to 127.0.0.1 only, accepts only localhost Host headers, and opens the
databases read-only.
"""
import os
import threading
import time
from contextvars import ContextVar
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook
from pydantic import BaseModel, Field

from core.answer import DISPLAY_MAX_ROWS, answer_from_result
from core.correct import execute_with_correction
from core.llm import get_llm
from core.types import PipelineResult
from core.validate import check_grounding
from db.connection import get_engine, get_inspector
from db.schema import render_ddl
from paths.registry import ALL_PATHS, build_runner

ROOT = Path(__file__).resolve().parent.parent
MAX_RETRIES = 2  # same as app.py
TABLE_MAX_ROWS = 1000  # rows sent to the browser; the answer still sees DISPLAY_MAX_ROWS

PATH_LABELS = {
    "A_vector": "A · Vector retrieval",
    "B_stuff": "B · Full schema stuffing",
    "B_select": "B · LLM self-selection",
    "C_agentic": "C · Agentic tool exploration",
    "D_graph": "D · Agentic + join graph",
}
MODELS = {"tui:gemma4:31b": "Gemma 4 31B (TU Ilmenau server, reasoning off)"}
DEFAULT_MODEL = "tui:gemma4:31b"
# Reasoning off, as in the thesis' reasoning ablation; set TUI_REASONING_EFFORT in .env to override.
os.environ.setdefault("TUI_REASONING_EFFORT", "none")


def _databases() -> dict[str, str]:
    dbs = {"chinook": ROOT / "data" / "chinook.sqlite", "tpch_sf001": ROOT / "data" / "tpch_sf001.sqlite"}
    # Spider is not redistributed here; listed only if downloaded (README, "Spider databases")
    spider = ROOT / "data" / "spider" / "spider_data" / "database"
    if spider.is_dir():
        for f in sorted(spider.glob("*/*.sqlite")):
            dbs[f"spider/{f.parent.name}"] = f
    # read-only at the driver: a second guard behind core.answer's SELECT-only check
    return {k: f"sqlite:///file:{v.as_posix()}?mode=ro&uri=true" for k, v in dbs.items() if v.exists()}


DATABASES = _databases()

# ---- agent-step capture: a LangChain callback injected through a context var, so
# the paths' code is untouched and only this request's runs are recorded.
_steps_var: ContextVar = ContextVar("webui_steps", default=None)
register_configure_hook(_steps_var, inheritable=True)


class _StepRecorder(BaseCallbackHandler):
    def __init__(self):
        self.steps: list[dict] = []
        self._by_run: dict = {}

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kw):
        if parent_run_id in self._by_run:  # inner call of guard_repeats' wrapper: same step
            self._by_run[run_id] = self._by_run[parent_run_id]
            return
        step = {"tool": (serialized or {}).get("name") or kw.get("name", "tool"),
                "input": kw.get("inputs") or input_str, "output": None}
        self._by_run[run_id] = step
        self.steps.append(step)

    def on_tool_end(self, output, *, run_id, **kw):
        if run_id in self._by_run:
            self._by_run[run_id]["output"] = str(getattr(output, "content", output))[:4000]

    def on_tool_error(self, error, *, run_id, **kw):
        if run_id in self._by_run:
            self._by_run[run_id]["output"] = f"error: {error}"


# ---- cached heavy objects, as app.py's st.cache_resource did
_engines, _llms, _runners = {}, {}, {}
_lock = threading.Lock()  # ponytail: one question at a time; local Ollama serialises anyway


def _engine(db):
    if db not in _engines:
        _engines[db] = get_engine(DATABASES[db])
    return _engines[db]


def _llm(model):
    if model not in _llms:
        _llms[model] = get_llm(model)
    return _llms[model]


def _runner(path, model, db):
    key = (path, model, db)
    if key not in _runners:
        _runners[key] = build_runner(path, _engine(db), _llm(model), db.split("/")[-1])
        return _runners[key][0], _runners[key][1]
    return _runners[key][0], 0.0


def _jsonable(v):
    return v if v is None or isinstance(v, (bool, int, float, str)) else str(v)


app = FastAPI(title="Structured RAG", docs_url=None, redoc_url=None, openapi_url=None)
# DNS-rebinding guard: a web page cannot reach this API through a hostname of its own
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])


class Ask(BaseModel):
    question: str = Field(max_length=2000)
    path: str = "B_stuff"
    model: str = DEFAULT_MODEL
    db: str = "chinook"


@app.get("/api/options")
def options():
    return {"paths": [{"id": p, "label": PATH_LABELS[p]} for p in ALL_PATHS],
            "models": [{"id": k, "label": v} for k, v in MODELS.items()],
            "databases": list(DATABASES)}


@app.post("/api/ask")
def ask(req: Ask):
    q = req.question.strip()
    if not q:
        raise HTTPException(400, "empty question")
    if req.path not in PATH_LABELS or req.model not in MODELS or req.db not in DATABASES:
        raise HTTPException(400, "unknown path, model or database")

    with _lock:
        rec = _StepRecorder()
        token = _steps_var.set(rec)
        try:
            return _ask(q, req, rec)
        except Exception as e:  # e.g. Ollama down, missing API key: show it, don't crash
            raise HTTPException(502, f"{type(e).__name__}: {e}")
        finally:
            _steps_var.reset(token)


def _ask(q, req, rec):
    engine = _engine(req.db)
    inspector = get_inspector(engine)
    all_tables = inspector.get_table_names()
    full_ddl = render_ddl(inspector, all_tables)

    start = time.perf_counter()
    runner, setup_ms = _runner(req.path, req.model, req.db)
    try:
        result = runner(q)
    except Exception as e:  # same treatment as run_eval: a crash is a result
        result = PipelineResult(question=q, db_id=req.db, path=req.path, error=f"pipeline crash: {e}")
    generated_sql = result.sql
    gen_steps = len(rec.steps)

    rows, grounding, answer, answer_tokens = [], None, "", 0
    if result.error and not result.sql.strip():
        answer = f"Pipeline error: {result.error}"
    else:
        rows, exec_error = execute_with_correction(engine, _llm(req.model), result, full_ddl,
                                                   max_retries=MAX_RETRIES)
        if exec_error:
            answer = f"SQL failed after {result.retries} repair attempt(s): {exec_error}"
        else:
            # same split as app.py: prompt sees a capped sample, grounding the full set
            answer, answer_tokens = answer_from_result(_llm(req.model), q, result.sql,
                                                       rows[:DISPLAY_MAX_ROWS])
            grounding = check_grounding(answer, rows, q)
            if len(rows) > DISPLAY_MAX_ROWS:
                answer += f"\n\n(summarised from the first {DISPLAY_MAX_ROWS} of {len(rows)} rows)"
    total_s = time.perf_counter() - start

    # schema the path put in front of the model
    trace = [t for t in result.tool_trace.split(",") if t]
    if req.path == "B_stuff":
        used = all_tables
    else:
        named = set(trace) | {str(v) for s in rec.steps[:gen_steps] if isinstance(s["input"], dict)
                              for v in s["input"].values()}
        used = [t for t in all_tables if t in named]

    completion = result.completion_tokens + answer_tokens
    return {
        "answer": answer,
        "sql": result.sql,
        "generated_sql": generated_sql,
        "error": result.error,
        "columns": list(rows[0].keys()) if rows else [],
        "rows": [[_jsonable(v) for v in r.values()] for r in rows[:TABLE_MAX_ROWS]],
        "row_count": len(rows),
        "schema": {"tables": used, "ddl": render_ddl(inspector, used) if used else ""},
        "trace": trace,
        "steps": [{**s, "input": _jsonable(s["input"]) if not isinstance(s["input"], dict)
                   else {k: _jsonable(v) for k, v in s["input"].items()}} for s in rec.steps[:gen_steps]],
        "metrics": {
            "total_s": round(total_s, 2),
            "generation_ms": round(result.latency_ms, 1),
            "setup_ms": setup_ms,
            "tool_calls": result.tool_calls,
            "retries": result.retries,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": completion,
            "tok_s": round(completion / total_s, 1) if completion and total_s > 0 else 0,
        },
        "grounding": None if grounding is None else
        {"grounded": grounding.grounded, "flagged": grounding.flagged, "checked": grounding.checked},
    }


DIST = Path(__file__).parent / "frontend" / "dist"
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8600)
