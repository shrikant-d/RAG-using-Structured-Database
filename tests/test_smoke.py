"""Offline checks (no LLM, no API key): `python -m pytest tests` or `python tests/test_smoke.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from core.answer import execute_sql  # noqa: E402
from webui.server import DATABASES, _engine, app  # noqa: E402

client = TestClient(app, base_url="http://127.0.0.1")


def test_databases_present():
    assert {"chinook", "tpch_sf001"} <= set(DATABASES)
    rows, err = execute_sql(_engine("chinook"), "SELECT COUNT(*) AS n FROM Artist")
    assert err is None and rows[0]["n"] == 275
    rows, err = execute_sql(_engine("tpch_sf001"), "SELECT COUNT(*) AS n FROM region")
    assert err is None and rows[0]["n"] == 5


def test_sql_guard():
    eng = _engine("chinook")
    for bad in ("DROP TABLE Artist", "SELECT 1; DROP TABLE Artist",
                "WITH x AS (SELECT 1) DELETE FROM Artist", "PRAGMA table_info(Artist)"):
        rows, err = execute_sql(eng, bad)
        assert err and not rows, bad


def test_database_is_read_only():
    try:
        with _engine("chinook").connect() as conn:
            conn.execute(text("DELETE FROM Artist"))
        raise AssertionError("write was not refused")
    except Exception as e:
        assert "readonly" in str(e).lower(), e


def test_api():
    opts = client.get("/api/options").json()
    assert [m["id"] for m in opts["models"]] == ["tui:gemma4:31b"]
    assert client.post("/api/ask", json={"question": "x" * 2001}).status_code == 422
    assert client.post("/api/ask", json={"question": "hi", "db": "../secret"}).status_code == 400
    evil = TestClient(app, base_url="http://evil.example")
    assert evil.get("/api/options").status_code == 400  # DNS-rebinding guard
    assert client.get("/docs").status_code == 404


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
