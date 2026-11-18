"""Shared contract for the two agentic paths (C and D).

Paths C and D exist to be compared: D is C plus a join graph, and the whole
point of the comparison is that the graph is the ONLY thing that differs. That
only holds if the system prompt, the step budget, and the loop's failure
handling are identical on both arms — and they were not. Path D had picked up a
general anti-repetition instruction and an explicit recursion_limit that Path C
never got, so a D-over-C win could equally have been "the arm that was told not
to loop, looped less". Both arms now build their prompt from BASE_SYSTEM_PROMPT
and run through run_react_path, so the difference is the tool set and the
sentences naming the graph tools, which is the manipulation itself.
"""
import json
import time
import uuid

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from core import refusal
from core.sql_extract import extract_sql
from core.types import PipelineResult
from core.usage import sum_usage

# Each ReAct "step" is one AI-message-plus-tool-call round trip; a normal query
# resolves in 3-10. 40 gives headroom for legitimately hard multi-hop questions
# while still bounding a stuck loop to ~40 LLM calls instead of unbounded. Both
# agentic paths use this value: the framework default (25) applied to only one
# arm is itself a difference between the arms.
MAX_AGENT_STEPS = 40

# recursion_limit (above) bounds graph super-steps -- LLM turns -- not raw tool
# calls. A single turn whose tool-call message requests many calls (a small
# local model batching or repeating calls within one generation) can run the
# raw count far past what MAX_AGENT_STEPS was meant to cap without ever
# tripping GraphRecursionError. Observed: D_graph q3 in the equalized_full
# sweep, 390 tool calls in one rollout, prompt grew past num_ctx (16876 >
# 16384), and the model produced SQL for a schema that doesn't exist in the
# database -- the exact silent-truncation failure core/llm.py's num_ctx
# docstring warns about, just reached through the tool-call axis instead of a
# long transcript. 60 is generous headroom over the largest legitimate rollout
# seen so far (20 tool calls) while catching a runaway well before it can
# overflow the context window.
MAX_TOOL_CALLS = 60

# sample_rows exists so the agent can see a few rows and learn value formats
# ('F' vs 'Shipped'), not so it can read tables. Uncapped, claude-haiku-4-5
# requested whole tables on Chinook: two rollouts took 45% of its agentic input
# tokens, and one answered in prose from the dumped rows instead of writing SQL.
# TPC-H's lineitem (60,175 rows at SF 0.01) would not fit its 200K window at all.
# qwen2.5:7b never came near this, but both arms must share one tool, so the cap
# lives here. SQLite reads a negative LIMIT as "no limit", so n is clamped at
# both ends.
MAX_SAMPLE_ROWS = 20


def sample_limit(n) -> int:
    """The LIMIT sample_rows actually applies for a requested n."""
    return max(1, min(int(n), MAX_SAMPLE_ROWS))


class ToolCallBudgetExceeded(Exception):
    """Raised by track_call() when a rollout's raw tool-call count exceeds
    MAX_TOOL_CALLS -- the backstop recursion_limit does not provide."""


def track_call(call_count: dict, name: str) -> None:
    """Record one tool invocation; both agentic paths' tools call this instead
    of touching call_count directly, so the budget is enforced once, for every
    tool, on both arms -- not per tool function, per path.

    Raising (rather than returning an error observation, the pattern every
    tool here otherwise follows) is deliberate: a model already stuck in a
    loop has shown it won't reliably act on a text observation telling it to
    stop -- q3 kept going for 390 calls despite the system prompt's explicit
    "do not repeat a call" instruction. langgraph's default ToolNode error
    handler only catches ToolInvocationError and re-raises everything else, so
    this propagates through agent.invoke() uncaught, straight to
    run_react_path's except block below -- no ToolNode configuration needed.
    """
    call_count["n"] += 1
    call_count["trace"].append(name)
    if call_count["n"] > MAX_TOOL_CALLS:
        raise ToolCallBudgetExceeded(
            f"exceeded {MAX_TOOL_CALLS} raw tool calls (stuck loop — "
            f"see tool_trace: {','.join(call_count['trace'][-6:])})"
        )

REPEAT_NOTE = ("You already made this exact call and the result is unchanged. "
               "Stop exploring and write the final SQL.")


def guard_repeats(tools: list) -> list:
    """Wrap every tool so a repeated call (same tool, same arguments ignoring
    case) still returns its result, but with REPEAT_NOTE attached.

    The prompt already says not to repeat calls; qwen2.5:7b ignores it. Traced
    D_graph Chinook q12 alternated two find_join_path calls fourteen times with
    identical results until the step budget ran out (design_notes §36). An
    observation on the repeated call itself is harder to miss than one sentence
    at the top of a long transcript. Applied in run_react_path, so both arms get
    it; MAX_TOOL_CALLS remains the hard backstop.
    """
    from langchain_core.tools import StructuredTool

    seen = set()

    def wrap(t):
        def run(**kwargs):
            key = (t.name, json.dumps({k: str(v).lower() for k, v in kwargs.items()}, sort_keys=True))
            result = t.invoke(kwargs)
            if key in seen:
                return {"repeated_call": REPEAT_NOTE, "result": result}
            seen.add(key)
            return result
        return StructuredTool.from_function(func=run, name=t.name, description=t.description,
                                            args_schema=t.args_schema)
    return [wrap(t) for t in tools]


# Identical on both arms, verbatim. Path D appends GRAPH_PROMPT_SUFFIX and adds
# nothing else — any instruction that is not about the graph belongs here, where
# both arms receive it.
BASE_SYSTEM_PROMPT = (
    "Explore the database schema using the tools, then respond with ONLY the final SQL query, "
    "written for the {dialect} dialect. NEVER guess column names: call describe_table for "
    "every table you reference and use only column names it returns, exactly as spelled. "
    "Once a tool call returns the same result you already have, stop exploring and write the "
    "final SQL — do not repeat a call you have already made with the same or equivalent arguments."
)

# The manipulated variable, stated to the model. Naming the tools is intrinsic to
# giving the agent tools: it cannot call what it has not been told about, so this
# text is part of the graph condition rather than an incidental prompt edge.
GRAPH_PROMPT_SUFFIX = (
    " You also have find_join_path and get_related_tables, which answer questions about the "
    "schema's foreign key topology. Use the JOIN conditions returned by find_join_path verbatim."
)


def _recover_ollama_tool_call(message: AIMessage, tool_names: set[str]) -> AIMessage:
    """Some Ollama-served models (qwen2.5-coder:7b confirmed by direct
    /api/chat testing; see github.com/ollama/ollama/issues/12174 and #10899)
    write an intended tool call as JSON prose in `content` instead of
    populating the native `tool_calls` field, so LangGraph never recognises it
    as a tool call -- the agent stalls with "no SQL produced" or a SQL error
    from the raw JSON leaking into extract_sql. Sometimes it writes several
    such objects back to back in one turn (one call worth batching, not one
    call spliced into a sentence), so this recovers every consecutive JSON
    object at the front of `content`, not just the first.

    Recovered narrowly: only when there are no real tool_calls already, and
    only the leading run of JSON objects that name a tool this agent actually
    has -- content that is not JSON at all (the model skipping tool use and
    guessing SQL directly) is left untouched, that is a real model behaviour
    to observe, not something to paper over here. qwen2.5:7b's native
    tool_calls always populate correctly (confirmed the same way), so this
    never fires for it -- existing runs are unaffected.
    """
    if message.tool_calls or not isinstance(message.content, str):
        return message
    text = message.content.strip()
    decoder = json.JSONDecoder()
    calls, pos = [], 0
    while pos < len(text):
        try:
            obj, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict) and obj.get("name") in tool_names:
            calls.append({
                "name": obj["name"],
                "args": obj.get("arguments", {}),
                "id": f"recovered_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            })
        pos = end
        while pos < len(text) and text[pos].isspace():
            pos += 1
    if not calls:
        return message
    return message.model_copy(update={"content": "", "tool_calls": calls})


def _build_messages(system: str, history: list[dict] | None, question: str) -> list[tuple[str, str]]:
    """[(system, ...), *prior (user, assistant) pairs, (user, question)].

    Conversation-case-study only (design_notes.md §30): prior turns are replayed as
    real chat turns so the agent can resolve a follow-up ("and for France?") the
    same way a chat model would. `history` is empty for every benchmark call, which
    reproduces the original two-message list exactly -- RQ1-6 results are unaffected.
    """
    messages = [("system", system)]
    for turn in (history or []):
        messages.append(("user", turn["question"]))
        messages.append(("assistant", turn["answer"]))
    messages.append(("user", question))
    return messages


def run_react_path(question: str, engine, llm, db_id: str, path: str, tools, call_count,
                   system_prompt: str, history: list[dict] | None = None) -> PipelineResult:
    """One ReAct rollout, shared by Paths C and D.

    Everything here except `tools` and `system_prompt` is held constant across
    the two arms by construction, which is what makes their comparison an
    ablation of the graph layer rather than of two agent configurations.
    """
    tools = guard_repeats(tools)
    tool_names = {t.name for t in tools}
    model = llm.bind_tools(tools) | RunnableLambda(
        lambda m: _recover_ollama_tool_call(m, tool_names))
    agent = create_react_agent(model, tools)
    # same opt-in sentence the single-shot template gets; "" unless ALLOW_REFUSAL=1
    system = system_prompt.format(dialect=engine.dialect.name) + refusal.suffix()

    start = time.perf_counter()
    error = None
    try:
        result = agent.invoke(
            {"messages": _build_messages(system, history, question)},
            config={"recursion_limit": MAX_AGENT_STEPS},
        )
        sql = extract_sql(result["messages"][-1].text)
    except GraphRecursionError:
        # Note: the prebuilt ReAct graph usually exhausts its own step budget and
        # returns a canned "need more steps" message instead of raising, which
        # extract_sql turns into no SQL. This catch covers the raising case so
        # neither arm can fail in a way the other would survive.
        result = {"messages": []}
        sql = ""
        error = (
            f"agent exceeded {MAX_AGENT_STEPS} steps without producing SQL "
            f"(stuck loop — see tool_trace: {','.join(call_count['trace'][-6:])})"
        )
    except ToolCallBudgetExceeded as e:
        result = {"messages": []}
        sql = ""
        error = str(e)
    latency_ms = (time.perf_counter() - start) * 1000

    prompt_tokens, completion_tokens = sum_usage(result["messages"])
    return PipelineResult(
        question=question,
        db_id=db_id,
        path=path,
        sql=sql,
        tool_calls=call_count["n"],
        tool_trace=",".join(call_count["trace"]),
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        error=error,
    )
