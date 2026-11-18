import networkx as nx
from langchain_core.tools import tool

from db.connection import get_inspector
from paths.agentic_common import (
    BASE_SYSTEM_PROMPT,
    GRAPH_PROMPT_SUFFIX,
    MAX_AGENT_STEPS,  # noqa: F401 — re-exported; experiments import it from here
    run_react_path,
    sample_limit,
    track_call,
)

# Path D is Path C plus the graph. The prompt is the shared base verbatim plus
# the graph suffix, so the two arms differ only in the graph tools and the text
# naming them — which is the manipulated variable.
SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + GRAPH_PROMPT_SUFFIX


def _resolve_table(inspector, name: str) -> str | None:
    """Case-insensitive table lookup — agents pass lowercased or value-derived names."""
    tables = inspector.get_table_names()
    if name in tables:
        return name
    return next((t for t in tables if t.lower() == name.lower()), None)


def build_graph_known(inspector, table_names: list[str]) -> nx.DiGraph:
    """Known-schema build: one pass over FK/PK metadata. For Spider, tables.json already
    has FK pairs, so this can be populated directly from JSON instead of an inspector call."""
    g = nx.DiGraph()
    for t in table_names:
        g.add_node(t)
    for t in table_names:
        for fk in inspector.get_foreign_keys(t):
            g.add_edge(
                t, fk["referred_table"],
                local_cols=fk["constrained_columns"],
                remote_cols=fk["referred_columns"],
            )
    return g


def _make_tools(engine, graph: nx.DiGraph, blind: bool = False):
    """blind=True is the D_null ablation: the two graph tools are present, named
    in the prompt and callable with the same schemas, but answer every valid
    table pair with the empty result they give for an unconnected pair. It
    separates "the graph's content misleads the agent" from "the extra tools and
    the longer transcript cost it" -- D_graph confounds the two, and qwen2.5:7b
    calls the graph tools in every rollout while scoring below C_agentic
    (design_notes §36)."""
    inspector = get_inspector(engine)
    call_count = {"n": 0, "trace": []}

    @tool
    def list_tables() -> list[str]:
        """Return all table names in the connected database."""
        track_call(call_count, "list_tables")
        return list(graph.nodes)

    @tool
    def describe_table(table_name: str) -> dict:
        """Return columns, types, PK/FK constraints. Also registers edges in the join graph
        (unknown-schema mode: the graph grows as this tool is called)."""
        track_call(call_count, "describe_table")
        resolved = _resolve_table(inspector, table_name)
        if resolved is None:
            # error goes back as an observation so the agent can recover, never raises
            return {"error": f"no table named '{table_name}'. Available: {inspector.get_table_names()}"}
        fks = inspector.get_foreign_keys(resolved)
        for fk in fks:
            graph.add_edge(
                resolved, fk["referred_table"],
                local_cols=fk["constrained_columns"],
                remote_cols=fk["referred_columns"],
            )
        return {
            "columns": inspector.get_columns(resolved),
            "foreign_keys": fks,
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

    def resolve_node(name: str) -> str | None:
        # Same case-insensitive lookup describe_table uses. Without it an
        # unknown name -- a column passed as a table ("Country"), a typo
        # ("MediaTypes"), or a lowercased "customer" -- came back as the same
        # empty result as "no path exists", and qwen2.5:7b read that as "a join
        # is needed" and invented one (design_notes §36).
        return next((t for t in graph.nodes if t.lower() == str(name).lower()), None)

    def unknown(name: str) -> dict:
        return {"error": f"no table named '{name}'. Available: {sorted(graph.nodes)}"}

    def join_condition(a: str, b: str) -> str:
        # Every column of a composite FK, not just the first: TPC-H's
        # lineitem -> partsupp is (l_partkey, l_suppkey), and the first column
        # alone joins each line item to all four suppliers of its part.
        if graph.has_edge(a, b):
            d, left, right = graph.edges[a, b], a, b
        else:
            d, left, right = graph.edges[b, a], b, a
        return " AND ".join(f"{left}.{lc} = {right}.{rc}"
                            for lc, rc in zip(d["local_cols"], d["remote_cols"]))

    @tool
    def find_join_path(table_x: str, table_y: str) -> dict:
        """Return the FK join chain connecting two tables: table sequence plus the exact
        JOIN conditions (real column names) to use. Empty if no path exists."""
        track_call(call_count, "find_join_path")
        x, y = resolve_node(table_x), resolve_node(table_y)
        if x is None or y is None:
            return unknown(table_x if x is None else table_y)
        if blind:
            return {"tables": [], "joins": []}
        table_x, table_y = x, y

        # Self-join: nx.shortest_path(G, x, x) always returns the trivial
        # single-node path and never traverses a self-loop edge, even when one
        # exists (e.g. Employee.ReportsTo -> Employee.EmployeeId) — confirmed
        # this left the agent with a useless {'tables': [x], 'joins': []}
        # response on every attempt, which it couldn't tell apart from "no
        # relationship exists," so it retried the same call repeatedly instead
        # of concluding the FK was self-referencing (D_graph q12, clean_baseline:
        # 15 find_join_path calls before giving up with no SQL produced at all).
        if table_x == table_y and graph.has_edge(table_x, table_x):
            return {"tables": [table_x, table_x], "joins": [join_condition(table_x, table_x)],
                    "note": "self-referencing FK - join the table to itself with an alias"}

        undirected = graph.to_undirected()
        try:
            path = nx.shortest_path(undirected, table_x, table_y)
        except nx.NetworkXNoPath:
            return {"tables": [], "joins": []}
        return {"tables": path, "joins": [join_condition(a, b) for a, b in zip(path, path[1:])]}

    @tool
    def get_related_tables(table_name: str, hops: int = 2) -> list[str] | dict:
        """Return tables within N hops of the given table in the join graph."""
        track_call(call_count, "get_related_tables")
        resolved = resolve_node(table_name)
        if resolved is None:
            return unknown(table_name)
        if blind:
            return []
        return list(nx.ego_graph(graph.to_undirected(), resolved, radius=hops).nodes)

    return [list_tables, describe_table, sample_rows, find_join_path, get_related_tables], call_count


def run_graph_agentic(
    question: str, engine, llm, db_id: str, prebuilt_graph: nx.DiGraph | None = None,
    history: list[dict] | None = None, blind: bool = False,
):
    inspector = get_inspector(engine)
    graph = prebuilt_graph if prebuilt_graph is not None else nx.DiGraph()
    if prebuilt_graph is None:
        for t in inspector.get_table_names():
            graph.add_node(t)

    tools, call_count = _make_tools(engine, graph, blind=blind)
    return run_react_path(question, engine, llm, db_id, "D_null" if blind else "D_graph",
                          tools, call_count, SYSTEM_PROMPT, history=history)
