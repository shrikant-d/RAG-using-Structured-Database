"""Self-correction loop: execute generated SQL, feed execution errors back to the
LLM for repair, retry up to a limit.

Follows the diagnosis-prescription-treatment pattern of DIN-SQL [9]: the database
error message is the diagnosis, the model prescribes a fix, the retry is the
treatment. Applied AFTER generation on the PipelineResult, so it works uniformly
for every path — single-shot (A, B) and agentic (C, D) alike — instead of being
reimplemented inside each one.

Kept as an explicit opt-in (max_retries=0 disables it) because the exposé's
experiment matrix needs baselines measured without correction: System F is
defined as System E *plus* this loop, so the ablation requires both settings.
"""
from core.answer import execute_sql
from core.sql_extract import extract_sql
from core.types import PipelineResult

REPAIR_PROMPT = """The SQL query below failed to execute. Write a corrected query.

Dialect: {dialect}
Schema:
{schema_context}

Question: {question}
Failed SQL: {sql}
Database error: {error}

Diagnose what the error means, then return ONLY the corrected SQL query.
Use only column names present in the schema above. Do not repeat the same mistake."""


def repair_sql(llm, question, sql, error, schema_context, dialect):
    """One repair attempt. Returns (repaired_sql, prompt_tokens, completion_tokens)."""
    response = llm.invoke(
        REPAIR_PROMPT.format(
            dialect=dialect,
            schema_context=schema_context,
            question=question,
            sql=sql,
            error=error,
        )
    )
    usage = getattr(response, "usage_metadata", None) or {}
    return (
        extract_sql(response.text),
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )


def execute_with_correction(
    engine,
    llm,
    result: PipelineResult,
    schema_context: str,
    max_retries: int = 2,
) -> tuple[list[dict], str | None]:
    """Execute result.sql, repairing on failure up to max_retries times.

    Mutates `result` in place: result.sql becomes the query that finally ran (or the
    last attempt if all failed), and retries/token counts are updated so the
    experiment CSV records the true cost of a corrected answer. Returns (rows, error).
    """
    # The pipeline already failed without producing SQL -- a crash caught by the
    # caller, or an agentic rollout that hit its step or tool-call budget. Keep
    # that error. Executing the empty SQL and assigning result.error below used to
    # overwrite it with "no SQL produced", so an API failure or a runaway loop was
    # recorded exactly like a model that answered in prose.
    if result.error and not result.sql.strip():
        return [], result.error

    rows, error = execute_sql(engine, result.sql)
    dialect = engine.dialect.name

    while error and result.retries < max_retries:
        # A policy refusal is not a repairable SQL error. Neither is an agent that
        # produced no SQL: repairing that would mean generating a query from the
        # schema alone, silently rescuing an agentic-path failure with a
        # single-shot method and confounding the very comparison being measured.
        if error.startswith(("blocked:", "no SQL produced", "refused:")):
            break
        result.retries += 1
        repaired, in_tok, out_tok = repair_sql(
            llm, result.question, result.sql, error, schema_context, dialect
        )
        result.prompt_tokens += in_tok
        result.completion_tokens += out_tok
        if not repaired or repaired == result.sql:
            break  # model returned nothing new — further retries would loop
        result.sql = repaired
        rows, error = execute_sql(engine, result.sql)

    result.error = error
    return rows, error
