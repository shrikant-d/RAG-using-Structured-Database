"""Execute generated SQL and produce a grounded NL answer from the result set."""
import re

from core.refusal import REFUSAL_ERROR, REFUSAL_TOKEN

ANSWER_PROMPT = """Given a SQL query result, answer the user's question in one or two plain sentences.
Use ONLY values present in the result — do not add numbers or names that are not there.

Question: {question}
SQL: {sql}
Result rows: {rows}

Answer:"""

# Display cap only. NEVER pass this to grading: execution match compares result
# sets, and truncating both sides to their first N rows compares two arbitrary
# subsets whenever the query has no ORDER BY. A semantically correct rewrite of
# Chinook q25 (111 rows) scores as WRONG under a 50-row cap, because the two
# engines' natural orders differ past row 50. Grading calls execute_sql with the
# default limit=None; only the chatbot caps, and only for what it renders.
DISPLAY_MAX_ROWS = 50

WRITE_VERBS = ("insert", "update", "delete", "drop", "alter", "create", "truncate",
               "replace", "attach", "detach", "pragma", "vacuum", "grant", "revoke")

# A data-modifying CTE is a WITH statement that writes. PostgreSQL supports these
# fully, and the prefix check alone would wave one through.
_CTE_WRITE = re.compile(r"\b(insert\s+into|update\s+\w|delete\s+from|merge\s+into)\b", re.I)


def _reject_reason(sql: str) -> str | None:
    """Why this statement must not be executed, or None if it may be.

    Prefix matching alone is not a read-only guard. It admits multi-statement
    input ("SELECT 1; DROP TABLE t") and data-modifying CTEs
    ("WITH x AS (...) DELETE FROM t"). Under SQLite the driver happens to refuse
    the first of those, so the guard looked sound; psycopg2 does not, and
    PostgreSQL is a target dialect here. Both are rejected explicitly rather than
    left to the driver.
    """
    if REFUSAL_TOKEN in sql:
        return REFUSAL_ERROR  # the model declined; a class of its own, never repaired
    head = sql.lstrip().lower()
    if not head.startswith(("select", "with")):
        if head.startswith(WRITE_VERBS):
            return "blocked: only SELECT queries are allowed"
        return "no SQL produced: model returned empty or non-SQL output"
    # strip string literals before looking for a statement separator, so a
    # semicolon inside a value ("WHERE Name = 'a;b'") is not read as one
    without_literals = re.sub(r"'[^']*'", "''", sql)
    if ";" in without_literals.rstrip().rstrip(";"):
        return "blocked: multiple statements are not allowed"
    if head.startswith("with") and _CTE_WRITE.search(without_literals):
        return "blocked: only SELECT queries are allowed"
    return None


def execute_sql(engine, sql: str, limit: int | None = None) -> tuple[list[dict], str | None]:
    """Run SQL, return (rows, error). Read-only guard: LLM-generated SQL is
    executed here, so anything that is not a pure read is refused outright.

    limit=None returns the full result set and is what grading must use. Pass an
    integer only for display, where showing every row of a large result is
    pointless; see DISPLAY_MAX_ROWS.

    A refusal and a missing query are reported distinctly. An agent that returned
    prose, a leaked tool call, or nothing at all has failed its own loop, which is
    a different error class from a model genuinely attempting a write — lumping
    them together would misattribute agent-loop failures to the safety guard.
    """
    reason = _reject_reason(sql)
    if reason:
        return [], reason
    try:
        with engine.connect() as conn:
            result = conn.exec_driver_sql(sql)
            # Rows are built by position, with repeated column names made unique.
            # dict(mapping) collapses two columns of the same name into one: a query
            # selecting sn.n_name and cn.n_name unaliased came back one column short
            # and was graded wrong although its result equalled gold (gemma4:31b,
            # TPC-H Q7, 9 rows; design_notes §42).
            seen, keys = {}, []
            for k in result.keys():
                seen[k] = seen.get(k, 0) + 1
                keys.append(k if seen[k] == 1 else f"{k}__{seen[k]}")
            fetched = result.fetchmany(limit) if limit else result.fetchall()
            rows = [dict(zip(keys, r)) for r in fetched]
        return rows, None
    except Exception as e:
        return [], str(e)


def answer_from_result(llm, question: str, sql: str, rows: list[dict]) -> tuple[str, int]:
    """Generate NL answer grounded exclusively on the result set.
    Returns (answer, completion_tokens)."""
    if not rows:
        return "The query returned no results.", 0
    response = llm.invoke(ANSWER_PROMPT.format(question=question, sql=sql, rows=rows))
    usage = getattr(response, "usage_metadata", None) or {}
    return response.text.strip(), usage.get("output_tokens", 0)
