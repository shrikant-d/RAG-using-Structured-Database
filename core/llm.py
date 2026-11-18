import os

DEFAULT_NUM_CTX = 16384

# Ample for a full 11-table Chinook DDL stuffing prompt (B_stuff, the largest
# non-agentic prompt) with headroom, but NOT safe under --pad-tables: growing
# the schema grows B_stuff's (and B_select's every-table fallback) DDL past
# what fits here, and that failure is silent (see num_ctx's docstring below).
# Callers must fall back to DEFAULT_NUM_CTX whenever pad_tables is active.
SHORT_NUM_CTX = 4096

# Ample headroom over the largest legitimate single-call generation measured
# (147 tokens, B_select's SQL + NL-answer pair) -- catches a genuine runaway
# generation long before it can consume the context window. Ollama's own
# default (num_predict unset, i.e. unbounded until num_ctx) has no such
# backstop, and request_timeout below does not reliably substitute for one:
# streaming responses reset the idle-read timer on every token, so a
# slow-but-still-producing-tokens generation never trips a wall-clock timeout.
# Observed directly: re-running equalized_full with the MAX_TOOL_CALLS fix
# (see agentic_common.py), D_graph q2 -- "How many customers are from
# Germany?", trivial -- decoded past 10,000 tokens in one turn, still
# climbing, request_timeout=200s long since exceeded, before being killed by
# hand. Very likely the same shape as design_notes.md §13/§16's unexplained
# 8-hour stall.
MAX_OUTPUT_TOKENS = 512


# Models served by the TU Ilmenau DBIS server are asked for as "tui:<model>",
# e.g. --model tui:gemma4:31b. TUI_LLM_BASE_URL in .env overrides the URL.
SERVER_PREFIX = "tui:"
SERVER_BASE_URL = "https://dbhage.prakinf.tu-ilmenau.de/api/v1"

# gemma4 is a reasoning model and its hidden reasoning counts against the output
# cap: at MAX_OUTPUT_TOKENS (512), 19 of the first 21 Chinook failures were
# truncations, not wrong SQL (one hard question: 432 tokens, ~60 of them SQL).
# Thinking stays on (reasoning_effort="none" would turn it off), so the server
# arm gets this larger cap -- a deliberately different condition from the 512 of
# the local and Haiku arms, to be reported as such. TUI_MAX_OUTPUT_TOKENS in the
# environment overrides it, for re-running only the rows that hit this cap.
SERVER_MAX_OUTPUT_TOKENS = 4096
# Reasoning runs ~37 tok/s on the server (4096 tokens in ~109 s), so a raised cap
# needs a timeout scaled to it, or the timeout becomes the cap.
SERVER_TOKENS_PER_SECOND = 30


ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def _load_env_file(path: str = ENV_FILE) -> None:
    """Copy KEY=VALUE lines from the repository's .env into os.environ.

    The cloud API key lives in that file (gitignored) so it never has to be typed
    into a shell, where it lands in the history file, and so a newly added key is
    picked up on the next run without restarting anything. A variable already set
    in the environment wins, and a blank placeholder (`KEY=`) is ignored.
    """
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def get_llm(model_name: str, temperature: float = 0.0, num_ctx: int = DEFAULT_NUM_CTX,
            num_predict: int = MAX_OUTPUT_TOKENS, request_timeout: float = 200.0):
    """Factory returning a LangChain chat model. One switch point for cloud vs local.

    Temperature defaults to 0 (greedy decoding). The providers' own defaults are
    high — Ollama 0.8, the cloud APIs 1.0 — and sampling at that level makes a
    configuration's score irreproducible: the pilot comparison saw 11 of 125
    queries flip verdict between two runs for reasons unrelated to the change
    being measured, which is the same order as the effect itself. Benchmark runs
    need greedy decoding so a score difference means a pipeline difference.

    num_ctx is set explicitly because Ollama's default context window is 2048
    tokens regardless of what the model supports (qwen2.5 is trained for 32768),
    and it truncates over-long prompts *silently* — no error, no warning, just a
    shorter prompt. That default invalidates any measurement of the agentic
    paths, whose ReAct transcripts average ~11k tokens: the system prompt and the
    early describe_table observations fall out of the window, so the agent is
    asked to write SQL for a schema it can no longer see. Raising it costs KV
    cache (VRAM) and therefore speed, which is the real trade-off on a 4 GB GPU —
    but a truncated agent is not a cheaper agent, it is a broken one.

    That trade-off is measured, not assumed: at num_ctx=16384 on the RTX 3050
    Laptop (4 GB), Ollama offloads only 11-12 of qwen2.5:7b's 29 layers to GPU
    (`ollama ps` shows a 59/41 CPU/GPU split), and decode runs 4-6 tok/s. Only
    the two agentic paths' ReAct transcripts need this window — see
    SHORT_NUM_CTX and design_notes.md for the per-path split this motivated.
    """
    _load_env_file()
    if model_name.startswith(SERVER_PREFIX):
        # The TU Ilmenau DBIS server: Open WebUI in front of Ollama, reached through
        # its OpenAI-compatible API. The prefix keeps a server run distinguishable
        # from a local Ollama model of the same name in every result file
        # (run_eval turns "tui:gemma4:31b" into "tui_gemma4_31b").
        key = os.environ.get("TUI_LLM_API_KEY")
        if not key:
            raise RuntimeError("TUI_LLM_API_KEY is not set: add it to the repository's .env")
        from langchain_openai import ChatOpenAI
        # Same temperature, timeout and retries as the other arms; output cap raised
        # (SERVER_MAX_OUTPUT_TOKENS above). No num_ctx: the OpenAI API has no such
        # field, so the window is the server's -- probed 2026-11-04 with a marker at
        # the start of a 42k-token prompt, returned intact (42,054 prompt tokens).
        cap = max(num_predict, int(os.environ.get("TUI_MAX_OUTPUT_TOKENS", SERVER_MAX_OUTPUT_TOKENS)))
        # TUI_REASONING_EFFORT=none switches thinking off (reasoning ablation); unset
        # keeps the server default, thinking on, as in every primary run.
        effort = os.environ.get("TUI_REASONING_EFFORT")
        return ChatOpenAI(model=model_name[len(SERVER_PREFIX):], temperature=temperature,
                          max_tokens=cap, timeout=max(request_timeout, cap / SERVER_TOKENS_PER_SECOND),
                          max_retries=6, **({"reasoning_effort": effort} if effort else {}),
                          base_url=os.environ.get("TUI_LLM_BASE_URL", SERVER_BASE_URL),
                          api_key=key)
    if model_name.startswith("gpt-"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=temperature)
    if model_name.startswith("claude-"):
        from langchain_anthropic import ChatAnthropic
        # Same output cap and timeout as the local branch below: RQ5 compares the
        # two, and an arm with a larger output budget is a different condition, not
        # just a different model. No context-window argument -- the cloud model's
        # 200K window against the local num_ctx is part of what local-vs-cloud
        # means, so it is reported rather than equalised. max_retries is raised from
        # the client default of 2 so a 429 is retried with backoff instead of
        # landing in the results as a model failure.
        return ChatAnthropic(model=model_name, temperature=temperature, max_tokens=num_predict,
                             timeout=request_timeout, max_retries=6)
    # anything else assumed local via Ollama (llama3.1, qwen2.5, mistral, ...)
    from langchain_ollama import ChatOllama
    # An explicit request timeout below the harness's own external OS-level
    # timeout (experiments/run_resume_loop.sh, 300s): the clean_baseline sweep
    # hit 13 stalls, ALL killed by the external wrapper — with a 600s client
    # timeout it structurally could never fire first (600 > 300), so it was
    # dead code the whole run. Set to 200s, comfortably above the slowest
    # legitimate query observed (D_graph, up to ~150s) but below the external
    # 300s ceiling, so a genuine stall is caught here first: it becomes an
    # ordinary in-process exception the per-query try/except already handles
    # (run_eval.py records it as a failed row and moves on), instead of the
    # external wrapper SIGKILLing the whole process and losing whatever partial
    # work was in flight. NOTE: this mechanism previously failed to fire during
    # an 8-hour stall (see design_notes.md §13) for a cause not yet isolated —
    # treat this as a latency improvement to validate empirically, not a proven
    # fix. The external wrapper remains the real safety net regardless.
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
        client_kwargs={"timeout": request_timeout},
    )
