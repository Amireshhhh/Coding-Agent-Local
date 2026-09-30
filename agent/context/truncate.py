"""Tool-output truncation: keep head 60% + tail 40% of the token budget (build plan 6.3 stage 1)."""

from __future__ import annotations

from agent.context.tokens import CharsDiv4Counter, TokenCounter

HEAD_FRACTION = 0.6
MARKER = "\n...[truncated {n} tokens; use read_file with offset/limit or grep to see more]...\n"


async def truncate_output(text: str, max_tokens: int, counter: TokenCounter | None = None) -> str:
    """Return ``text`` unchanged if it fits ``max_tokens``; otherwise head + marker + tail.

    The head gets 60% and the tail 40% of the budget. The result (head + tail, excluding
    the marker) never exceeds ``max_tokens`` under ``counter``. ``N`` in the marker is the
    number of tokens removed: ``total - tokens(head) - tokens(tail)``.
    """
    counter = counter or CharsDiv4Counter()
    total = await counter.count_text(text)
    if total <= max_tokens:
        return text
    if max_tokens <= 0:
        return MARKER.format(n=total)
    chars_per_tok = len(text) / total
    head_budget = int(max_tokens * HEAD_FRACTION)
    tail_budget = max_tokens - head_budget
    scale = 1.0
    for _ in range(40):
        hc = max(0, int(head_budget * chars_per_tok * scale))
        tc = max(0, int(tail_budget * chars_per_tok * scale))
        head = text[:hc]
        tail = text[len(text) - tc:] if tc else ""
        h, t = await counter.count_text(head), await counter.count_text(tail)
        if h <= head_budget and t <= tail_budget:
            return head + MARKER.format(n=max(0, total - h - t)) + tail
        scale *= 0.9
    return MARKER.format(n=total)
