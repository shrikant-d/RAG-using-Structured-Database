from langchain_core.tools import tool

from db.connection import get_inspector
from paths.agentic_common import BASE_SYSTEM_PROMPT, run_react_path, sample_limit, track_call

# Path C is the un-augmented arm: the shared base prompt with nothing added.
# Path D is this plus the graph suffix — see paths/agentic_common.py.
SYSTEM_PROMPT = BASE_SYSTEM_PROMPT


def _resolve_table(inspector, name: str) -> str | None:
    """Case-insensitive table lookup — agents pass lowercased or value-derived names."""
    tables = inspector.get_table_names()
    if name in tables:
        return name
    return next((t for t in tables if t.lower() == name.lower()), None)


def _make_tools(engine):
    inspector = get_inspector(engine)
    call_count = {"n": 0, "trace": []}

    @tool
    def list_tables() -> list[str]:
        """Return all table names in the connected database."""
        track_call(call_count, "list_tables")
        return inspector.get_table_names()

    @tool
    def describe_table(table_name: str) -> dict:
        """Return column names, data types, and PK/FK constraints for one table."""
        track_call(call_count, "describe_table")
        resolved = _resolve_table(inspector, table_name)
        if resolved is None:
            # error goes back as an observation so the agent can recover, never raises
            return {"error": f"no table named '{table_name}'. Available: {inspector.get_table_names()}"}
        return {
            "columns": inspector.get_columns(resolved),
            "foreign_keys": inspector.get_foreign_keys(resolved),
            "primary_key": inspector.get_pk_constraint(resolved),
        }

    @tool
    def sample_rows(table_name: str, n: int = 5) -> list[dict]:
        """Return up to 20 representative rows from a table, to see value formats."""
        track_call(call_count, "sample_rows")
        resolved = _resolve_table(inspector, table_name)
        if resolved is None:
            return [{"error": f"no table named '{table_name}'. Available: {inspector.get_table_names()}"}]
        with engine.connect() as conn:
            rows = conn.exec_driver_sql(f'SELECT * FROM "{resolved}" LIMIT {sample_limit(n)}').mappings().all()
        return [dict(r) for r in rows]

    return [list_tables, describe_table, sample_rows], call_count


def run_agentic(question: str, engine, llm, db_id: str, history: list[dict] | None = None):
    tools, call_count = _make_tools(engine)
    return run_react_path(question, engine, llm, db_id, "C_agentic",
                          tools, call_count, SYSTEM_PROMPT, history=history)
