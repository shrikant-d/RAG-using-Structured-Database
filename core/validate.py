"""Layer 1 hallucination check: result-grounded entity verification.

Extracts numbers and proper-noun-ish tokens from the NL answer and verifies each
against the SQL result set. Zero LLM cost. Catches fabricated values — the most
common and most dangerous hallucination class.

Layer 2 (LLM-as-judge faithfulness, RAGAS-style) plugs in beside this later:
same inputs (answer, rows), returns a 0-1 score instead of a flag list — keep
that signature in mind when extending.
"""
import re
from dataclasses import dataclass, field


@dataclass
class GroundingResult:
    grounded: bool
    flagged: list[str] = field(default_factory=list)  # entities not found in result set
    checked: int = 0


_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")


def _result_value_set(rows: list[dict]) -> set[str]:
    values = set()
    for row in rows:
        for v in row.values():
            if v is None:
                continue
            s = str(v).strip()
            values.add(s.lower())
            # numeric values also normalised without trailing zeros/commas
            try:
                values.add(f"{float(s):g}")
            except ValueError:
                pass
    return values


def _normalise_number(token: str) -> str:
    token = token.replace(",", "").rstrip(".")
    try:
        return f"{float(token):g}"
    except ValueError:
        return token


def _numeric_values(rows: list[dict]) -> list[float]:
    out = []
    for row in rows:
        for v in row.values():
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, (int, float)):
                out.append(float(v))
            else:
                try:
                    out.append(float(str(v).strip()))
                except (TypeError, ValueError):
                    pass
    return out


def _is_rounding_of_result(token: str, numeric_values: list[float]) -> bool:
    """True if some numeric result value rounds to `token` at its own precision.

    An exact string match is the wrong test for a figure a person would quote.
    AVG over Chinook's Canadian invoices is 5.4278571428571425; an answer saying
    "$5.43" has reported that value correctly, and flagging it as fabricated
    penalises the model for formatting a currency figure the way anyone would.
    Every path in the full sweep was marked ungrounded on that one question for
    exactly this reason. Judged from the answer's side, at the precision the
    answer states, so a correct rounding passes and a different number does not.
    """
    token = token.replace(",", "").rstrip(".")
    try:
        stated = float(token)
    except ValueError:
        return False
    dp = len(token.split(".")[1]) if "." in token else 0
    tol = 0.5 * (10 ** -dp) + 1e-9
    return any(abs(v - stated) <= tol for v in numeric_values)


def check_grounding(nl_answer: str, rows: list[dict],
                    question: str | None = None) -> GroundingResult:
    """Every number in the answer must appear in the result set. Numbers are the
    check's backbone — name matching is done as substring lookup because answers
    inflect names ('Rock genre', 'the artist Queen') while numbers don't lie.

    `question` is optional and is used for one narrow purpose: discounting
    figures the question itself supplied. Asking for "genres with more than 500
    tracks" and answering "the genres with more than 500 tracks are ..." is
    restating the user's own threshold, not claiming a value from the data, and
    flagging it as fabricated is wrong. This does not make the check
    question-aware in any deeper sense — it still cannot tell which value in a
    result set answers what was asked, which is why answer coverage is scored
    separately.
    """
    if not rows:
        return GroundingResult(grounded=True)

    values = _result_value_set(rows)
    answer_lower = nl_answer.lower()
    flagged, checked = [], 0

    string_values = [v for v in values if not v.replace(".", "").replace("-", "").isdigit()]

    # A numeral inside a legitimate result string is part of that string, not a
    # claimed figure. Chinook has playlists called "90's Music", "Classical 101"
    # and "On-The-Go 1", and an artist called "U2"; listing any of them correctly
    # was flagged as fabricating 90, 101, 1 or 2, because the value set holds the
    # whole string and not its digits. Remove the string values the answer
    # legitimately quotes first, so only free-standing figures are checked.
    # Length is not the criterion: an earlier version skipped values of two
    # characters or fewer and therefore missed "U2" entirely. What matters is
    # that the value mixes digits with non-digits, i.e. it is a name that
    # happens to contain a numeral.
    residual = answer_lower
    named_with_digits = [v for v in string_values if any(c.isdigit() for c in v)]
    for v in sorted(named_with_digits, key=len, reverse=True):
        residual = residual.replace(v, " ")

    # Figures the question itself supplied are the user's own, not claims about
    # the data: "genres with more than 500 tracks" answered as "the genres with
    # more than 500 tracks are ..." restates the threshold and was flagged as
    # fabricating 500.
    question_numbers = {_normalise_number(t) for t in _NUMBER_RE.findall(question or "")}

    # numbers: exact match after normalisation, or a correct rounding of one
    numeric_values = _numeric_values(rows)
    for token in _NUMBER_RE.findall(residual):
        checked += 1
        if _normalise_number(token) in values:
            continue
        if _normalise_number(token) in question_numbers:
            continue
        if _is_rounding_of_result(token, numeric_values):
            continue
        flagged.append(token)

    # string values from the result set: at least one must appear in the answer
    # (weak signal the answer talks about the actual rows, not parametric memory)
    if string_values:
        checked += 1
        if not any(v in answer_lower for v in string_values if len(v) > 2):
            flagged.append("(no result-set name mentioned in answer)")

    return GroundingResult(grounded=not flagged, flagged=flagged, checked=checked)
