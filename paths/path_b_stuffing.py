from core.generate import generate_sql
from core.usage import usage_of
from db.connection import get_inspector
from db.schema import render_ddl


def run_full_stuffing(question: str, engine, llm, db_id: str, history: list[dict] | None = None):
    """Path B (sub-strategy 1): every table's DDL in the prompt verbatim. No retrieval step.

    `history` is conversation-case-study only (design_notes.md §30) -- see generate_sql.
    """
    inspector = get_inspector(engine)
    context = render_ddl(inspector, inspector.get_table_names())
    return generate_sql(question, context, llm, db_id, path="B_stuff",
                        dialect=engine.dialect.name, history=history)


def run_self_select(question: str, engine, llm, db_id: str, history: list[dict] | None = None):
    """Path B (sub-strategy 2): LLM picks relevant tables from a compact index first,
    full DDL fetched only for those. Two LLM calls, no embeddings.

    `history` is conversation-case-study only (design_notes.md §30) -- see generate_sql.
    """
    inspector = get_inspector(engine)
    tables = inspector.get_table_names()
    index = "\n".join(f"- {t}" for t in tables)
    # A bare follow-up ("and for France?") names no table, so the selection step
    # needs the same prior-question context the generation step gets, or it picks
    # nothing meaningful and the path silently degrades to B_stuff's every-table
    # fallback on turn 2 -- defeating the point of testing B_select's own retrieval
    # under a follow-up. Empty when history is empty, so turn-1 and the benchmark's
    # no-history calls are unaffected.
    history_hint = (
        "Prior questions in this conversation: " + "; ".join(t["question"] for t in history) + "\n\n"
        if history else ""
    )
    select_prompt = (
        f"{history_hint}Tables:\n{index}\n\nQuestion: {question}\n\n"
        "List only the table names relevant to answering this question, comma-separated."
    )
    select_response = llm.invoke(select_prompt)
    select_in, select_out = usage_of(select_response)

    # Case-insensitive, punctuation-tolerant match. The model is asked for bare
    # names but returns "album," or "the Track table"; an exact membership test
    # dropped those, `picked` came back empty, and the path silently fell back to
    # every table — i.e. became B_stuff while still being recorded as B_select,
    # collapsing the very comparison this path exists to support.
    by_lower = {t.lower(): t for t in tables}
    picked, seen = [], set()
    for chunk in select_response.text.replace("\n", ",").split(","):
        name = by_lower.get(chunk.strip().strip("`\"'.*-[]() \t").lower())
        if name and name not in seen:
            seen.add(name)
            picked.append(name)
    selected = picked or tables

    context = render_ddl(inspector, selected)
    result = generate_sql(question, context, llm, db_id, path="B_select",
                          dialect=engine.dialect.name, history=history)
    # the selection call is part of this path's cost, not overhead outside it
    result.prompt_tokens += select_in
    result.completion_tokens += select_out
    # Same trace contract as Path A: this path also has a selection step, so its
    # failures also split into "picked the wrong tables" and "wrote the wrong
    # query over the right ones". Logging both paths' selections is what lets the
    # crossover be read as a retrieval result rather than an accuracy coincidence.
    result.tool_trace = ",".join(selected)
    return result
