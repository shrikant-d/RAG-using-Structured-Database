"""The one place a schema retrieval path is constructed.

There were two: `build_runner` in experiments/run_eval.py and `run_path` in
app.py. They had already diverged — the app's Path A ignored the `k` the sweep
varies — so the chatbot and the benchmark could retrieve different numbers of
tables for the same question. Every caller now goes through `build_runner`, and
a new path is registered once.

Setup cost is returned rather than discarded. Path A's index build and Path D's
catalogue pass are the preprocessing that RQ3 is *about*, and both used to happen
outside every timer in the harness, so the trade-off the research question turns
on was the one quantity never recorded.
"""
import time

from db.connection import get_inspector


def build_runner(path: str, engine, llm, db_id: str, k: int = 5) -> tuple[callable, float]:
    """(runner, setup_ms) for the named path.

    The runner takes one question and returns a PipelineResult; all expensive
    per-path setup happens once, here, and is timed.
    """
    start = time.perf_counter()

    if path == "A_vector":
        from paths.path_a_vector import build_index, run_vector
        collection = build_index(engine, f"{db_id}_eval")
        runner = lambda q: run_vector(q, engine, llm, db_id, collection, k=k)  # noqa: E731

    elif path == "B_stuff":
        from paths.path_b_stuffing import run_full_stuffing
        runner = lambda q: run_full_stuffing(q, engine, llm, db_id)  # noqa: E731

    elif path == "B_select":
        from paths.path_b_stuffing import run_self_select
        runner = lambda q: run_self_select(q, engine, llm, db_id)  # noqa: E731

    elif path == "C_agentic":
        from paths.path_c_agentic import run_agentic
        runner = lambda q: run_agentic(q, engine, llm, db_id)  # noqa: E731

    elif path in ("D_graph", "D_null"):
        from paths.path_d_graph import build_graph_known, run_graph_agentic
        inspector = get_inspector(engine)
        graph = build_graph_known(inspector, inspector.get_table_names())
        # D_null is the ablation: same prompt, same tools, same catalogue pass,
        # but the graph tools answer nothing. Keeping the build here (rather than
        # skipping it) holds the preprocessing cost identical to D_graph's too.
        blind = path == "D_null"
        # each question gets a copy: the agent mutates the graph it is given, and
        # one question's exploration must not leak into the next question's start
        runner = lambda q: run_graph_agentic(  # noqa: E731
            q, engine, llm, db_id, prebuilt_graph=graph.copy(), blind=blind)

    else:
        raise ValueError(f"unknown path {path}")

    return runner, round((time.perf_counter() - start) * 1000, 1)


# D_null is deliberately absent: it is an ablation of D_graph, asked for by name
# with --paths, not one of the four pathways the thesis compares.
ALL_PATHS = ["A_vector", "B_stuff", "B_select", "C_agentic", "D_graph"]

# The only paths whose prompt carries a full ReAct transcript, i.e. the only
# ones that need core.llm.DEFAULT_NUM_CTX rather than the cheaper SHORT_NUM_CTX.
AGENTIC_PATHS = {"C_agentic", "D_graph", "D_null"}
