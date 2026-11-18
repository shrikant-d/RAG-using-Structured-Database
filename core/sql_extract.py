import re

from core.refusal import REFUSAL_TOKEN

_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_SQL_STATEMENT_RE = re.compile(r"(SELECT|WITH|INSERT|UPDATE|DELETE)\b.*", re.IGNORECASE | re.DOTALL)
# bare column compared to a string literal (col = 'text') OR one already wrapped
# on the value side (col = LOWER('text')). The LOWER( and its ) are matched as a
# single alternative, never independently — an earlier version matched a trailing
# ')' unconditionally whenever one happened to follow the literal, which silently
# swallowed the closing paren of an enclosing subquery (e.g. `= (SELECT ... = 'x')`)
# whenever the filter had no LOWER() wrapper to begin with. That corrupted
# otherwise-correct SQL into a syntax error post-generation, on 9 of 125 pilot
# queries across four different paths — a bug in this rewriter, not the model.
#
# The lookbehind excludes a word character or a dot, so a match can only start at
# the beginning of a (possibly qualified) column name. It used to exclude "(" in
# place of ".", which does nothing useful -- a column inside a function call is
# followed by ")" and never matches -- and broke grouped filters: in
# `(sn.n_name = 'France'` the match could not start at `sn`, so it started at
# `n_name` and produced `sn.LOWER(n_name)`, a syntax error. That corrupted 7 rows
# of the qwen TPC-H baselines (Q7 and Q19, whose OR-grouped filters take this
# shape) and 2 of the first Haiku TPC-H run.
_TEXT_EQ_RE = re.compile(
    r"(?<![\w.])(\w+(?:\.\w+)?)\s*=\s*(?:LOWER\((\'[^\']*\')\)|(\'[^\']*\'))",
    re.IGNORECASE,
)


def _lower_both_sides(match: re.Match) -> str:
    col = match.group(1)
    literal = match.group(2) or match.group(3)
    return f"LOWER({col}) = LOWER({literal})"


def enforce_ci_text_filters(sql: str) -> str:
    """Rewrite text-literal equality filters to compare case-insensitively on both
    sides. Models (observed: qwen2.5:7b, consistently) apply LOWER() to only one
    side despite explicit instructions — prompt-level fixes don't hold, so this is
    enforced mechanically after generation. Numeric comparisons are untouched
    (pattern requires a quoted literal)."""
    return _TEXT_EQ_RE.sub(_lower_both_sides, sql)


def extract_sql(text: str) -> str:
    """Pull a bare SQL statement out of LLM output that may include prose,
    markdown fences, or both — agents don't reliably follow "SQL only"
    instructions (observed on Path D w/ qwen2.5:7b: reasoning prose + fenced
    block instead of bare SQL)."""
    text = text.strip()

    # A refusal (core/refusal.py) is passed through as its bare token, so that
    # execute_sql reports it as a refusal and not as "no SQL produced". The token
    # never occurs in ordinary model output, so runs without refusal are unaffected.
    if REFUSAL_TOKEN in text:
        return REFUSAL_TOKEN

    fence_match = _SQL_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    stmt_match = _SQL_STATEMENT_RE.search(text)
    if stmt_match:
        text = stmt_match.group(0).strip()

    # trailing prose after the statement ("Note: ...") — cut at the closing semicolon
    if ";" in text:
        text = text[: text.index(";") + 1]

    return enforce_ci_text_filters(text.rstrip("`").strip())
