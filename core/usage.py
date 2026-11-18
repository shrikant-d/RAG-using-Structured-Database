def usage_of(response) -> tuple[int, int]:
    """(prompt_tokens, completion_tokens) from one LLM response.

    Every direct llm.invoke() in a path must run its response through this and
    add the result to the PipelineResult. Path B's self-selection call was
    missed for exactly this reason: it makes two LLM calls but only the second
    one's usage was recorded, so B_select's reported cost was one call short in
    every row — and token cost is an axis RQ3 compares paths on.
    """
    u = getattr(response, "usage_metadata", None) or {}
    return u.get("input_tokens", 0), u.get("output_tokens", 0)


def sum_usage(messages) -> tuple[int, int]:
    """Sum (prompt_tokens, completion_tokens) across all agent messages that carry usage."""
    prompt, completion = 0, 0
    for m in messages:
        u = getattr(m, "usage_metadata", None)
        if u:
            prompt += u.get("input_tokens", 0)
            completion += u.get("output_tokens", 0)
    return prompt, completion
