"""Opt-in refusal: let a model decline a question the schema cannot answer.

No pathway had a way to say "this database does not hold that". Asked for the
average age of customers on a schema with no birth date, a model either invents
a column (a database error) or answers a neighbouring question with full
confidence, and every downstream check passes, because each compares the answer
with the query and none compares the query with the question.

Off by default. Every result in the thesis was produced with it off, and with it
off the prompts are byte-identical to the ones those runs used (pinned by
test_refusal_is_off_by_default). It is switched on for the unanswerable-question
experiment only.

ponytail: an environment switch, not a parameter threaded through five paths and
the registry. Promote it to a run_eval flag if refusal ever becomes a default.
"""
import os

REFUSAL_TOKEN = "CANNOT_ANSWER"

# One sentence, appended to the single-shot template and to the agent system
# prompt alike, so all five configurations receive the same instruction.
REFUSAL_INSTRUCTION = (
    " If the schema does not contain the information the question asks for, do not write a "
    f"query: respond with exactly {REFUSAL_TOKEN}."
)

REFUSAL_ERROR = "refused: the model declined, the schema does not hold the data"


def enabled() -> bool:
    return os.environ.get("ALLOW_REFUSAL") == "1"


def suffix() -> str:
    """The text to append to a prompt: the instruction when on, nothing when off."""
    return REFUSAL_INSTRUCTION if enabled() else ""
