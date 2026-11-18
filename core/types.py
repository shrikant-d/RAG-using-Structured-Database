from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineResult:
    question: str
    db_id: str
    path: str  # "A_vector" | "B_stuff" | "B_select" | "C_agentic" | "D_graph"
    sql: str = ""
    nl_answer: Optional[str] = None
    tool_calls: int = 0
    # What the path put in front of the LLM, as a comma-separated trace. For the
    # agentic paths that is which tools fired, in order (e.g.
    # "list_tables,describe_table,find_join_path"): counts alone cannot answer
    # whether the knowledge graph is actually consulted or whether the agent
    # guesses joins instead, and that distinction is the whole point of Path D.
    # For the retrieval paths (A_vector, B_select) it is the table names the
    # retrieval step selected, which is what makes schema-linking recall — and
    # therefore the Path A / Path B crossover — computable from the results CSV
    # instead of only end-to-end execution accuracy.
    tool_trace: str = ""
    retries: int = 0  # self-correction attempts used (0 = first query executed cleanly)
    gpu_util_mean: float = 0.0  # mean GPU utilization % sampled over the query's wall-clock window
    gpu_util_peak: float = 0.0
    gpu_mem_mib: float = 0.0  # mean VRAM used, MiB — 0.0 for cloud models or no nvidia-smi
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None
