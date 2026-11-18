import time

from core import refusal
from core.sql_extract import extract_sql
from core.types import PipelineResult

SQL_GEN_PROMPT = """You are a SQL expert. Given the schema context and question, write one {dialect}-compatible SQL query.

Schema:
{schema_context}

Question: {question}

Return only the SQL query, no explanation, no markdown fences."""

# Case-study only (design_notes.md §30): a follow-up turn gets prior Q/SQL/answer
# so the model can resolve references like "and for France?". Kept as a separate
# template rather than a conditional insert into SQL_GEN_PROMPT so the no-history
# path is byte-identical to the original prompt -- every RQ1-6 result was produced
# without this template and must not change if this case study is never escalated
# to a formal RQ.
HISTORY_GEN_PROMPT = """You are a SQL expert continuing a conversation about a database. Given the schema, the prior turns, and the new question, write one {dialect}-compatible SQL query. Resolve any reference to the prior turns (e.g. "that", "and for France?") using them.

Schema:
{schema_context}

Conversation so far:
{history_block}

New question: {question}

Return only the SQL query, no explanation, no markdown fences."""


def _render_history(history: list[dict]) -> str:
    return "\n\n".join(
        f"Q: {turn['question']}\nSQL: {turn['sql']}\nA: {turn['answer']}" for turn in history
    )


def _build_prompt(question: str, schema_context: str, dialect: str,
                  history: list[dict] | None = None) -> str:
    # refusal.suffix() is "" unless ALLOW_REFUSAL=1, so the default prompt is unchanged
    if history:
        return HISTORY_GEN_PROMPT.format(schema_context=schema_context, question=question,
                                          dialect=dialect, history_block=_render_history(history)) + refusal.suffix()
    return SQL_GEN_PROMPT.format(schema_context=schema_context, question=question, dialect=dialect) + refusal.suffix()


def generate_sql(
    question: str, schema_context: str, llm, db_id: str, path: str, dialect: str = "SQLite",
    history: list[dict] | None = None,
) -> PipelineResult:
    """Shared generation call for non-agentic paths (A, B). Agentic paths (C, D) call
    the LLM inside their own ReAct loop instead of through here.

    `history` is prior conversation turns ({"question", "sql", "answer"}), used only
    by the conversation case study (experiments/run_conversation_case_study.py) --
    the benchmark harness never passes it, so its runs are unaffected.
    """
    prompt = _build_prompt(question, schema_context, dialect, history)
    start = time.perf_counter()
    response = llm.invoke(prompt)
    latency_ms = (time.perf_counter() - start) * 1000

    # .text, not .content: Anthropic returns a list of content blocks when a reply
    # has more than one, and .strip() on a list raises -- a silent zero on the cloud
    # arm only. .text joins the text blocks, and is the same string for Ollama.
    sql = extract_sql(response.text)

    usage = getattr(response, "usage_metadata", None) or {}
    return PipelineResult(
        question=question,
        db_id=db_id,
        path=path,
        sql=sql,
        latency_ms=latency_ms,
        prompt_tokens=usage.get("input_tokens", 0),
        completion_tokens=usage.get("output_tokens", 0),
    )
