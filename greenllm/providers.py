"""Provider-specific logic: usage parsing and energy-saving request rewrites."""
from __future__ import annotations

import copy
import json
from typing import Any

from .energy import Usage

EPHEMERAL = {"type": "ephemeral"}


# ---------------------------------------------------------------- usage parsing
def anthropic_usage(u: dict[str, Any] | None) -> Usage:
    u = u or {}
    return Usage(
        input_tokens=int(u.get("input_tokens") or 0),
        output_tokens=int(u.get("output_tokens") or 0),
        cache_read_tokens=int(u.get("cache_read_input_tokens") or 0),
        cache_write_tokens=int(u.get("cache_creation_input_tokens") or 0),
    )


def openai_usage(u: dict[str, Any] | None) -> Usage:
    u = u or {}
    prompt = int(u.get("prompt_tokens") or 0)
    cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    return Usage(
        input_tokens=max(prompt - cached, 0),
        output_tokens=int(u.get("completion_tokens") or 0),
        cache_read_tokens=cached,
    )


class SSEUsageParser:
    """Collects token usage from a streamed (SSE) response without altering it."""

    def __init__(self, provider: str):
        self.provider = provider
        self.usage = Usage()
        self._buf = b""

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if not data or data == b"[DONE]":
                continue
            try:
                event = json.loads(data)
            except ValueError:
                continue
            self._handle(event)

    def _handle(self, ev: dict[str, Any]) -> None:
        if self.provider == "anthropic":
            if ev.get("type") == "message_start":
                u = anthropic_usage((ev.get("message") or {}).get("usage"))
                self.usage.input_tokens = u.input_tokens
                self.usage.cache_read_tokens = u.cache_read_tokens
                self.usage.cache_write_tokens = u.cache_write_tokens
                self.usage.output_tokens = u.output_tokens
            elif ev.get("type") == "message_delta" and ev.get("usage"):
                # output_tokens in message_delta is cumulative
                self.usage.output_tokens = int(ev["usage"].get("output_tokens") or 0)
        elif ev.get("usage"):
            self.usage = openai_usage(ev["usage"])


# ---------------------------------------------------------------- request rewrites
def _has_cache_control(obj: Any) -> bool:
    if isinstance(obj, dict):
        return "cache_control" in obj or any(_has_cache_control(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_cache_control(v) for v in obj)
    return False


def _text_len(obj: Any) -> int:
    return len(json.dumps(obj, ensure_ascii=False)) if obj else 0


def add_anthropic_prompt_cache(body: dict[str, Any], min_chars: int) -> tuple[dict, bool]:
    """Mark the stable prefix (tools + system) as cacheable when it is long and the
    caller has not set cache_control themselves. Repeated prefixes are then read
    from the provider's cache instead of being recomputed."""
    if _has_cache_control(body):
        return body, False
    system, tools = body.get("system"), body.get("tools")
    if _text_len(system) + _text_len(tools) < min_chars:
        return body, False
    body = copy.deepcopy(body)
    if system:
        if isinstance(system, str):
            body["system"] = [{"type": "text", "text": system, "cache_control": EPHEMERAL}]
        elif isinstance(system, list) and isinstance(system[-1], dict):
            system = body["system"]
            system[-1]["cache_control"] = EPHEMERAL
        else:
            return body, False
    elif isinstance(tools, list) and tools and isinstance(tools[-1], dict):
        body["tools"][-1]["cache_control"] = EPHEMERAL
    else:
        return body, False
    return body, True


def cap_output(body: dict[str, Any], provider: str, cap: int) -> dict[str, Any]:
    if cap <= 0:
        return body
    body = dict(body)
    keys = ["max_tokens"] if provider == "anthropic" else ["max_tokens", "max_completion_tokens"]
    present = [k for k in keys if k in body]
    if not present:
        body[keys[-1]] = cap
    for k in present:
        if isinstance(body[k], int):
            body[k] = min(body[k], cap)
    return body


def ensure_openai_stream_usage(body: dict[str, Any]) -> dict[str, Any]:
    if body.get("stream"):
        body = dict(body)
        opts = dict(body.get("stream_options") or {})
        opts.setdefault("include_usage", True)
        body["stream_options"] = opts
    return body


def last_user_text(body: dict[str, Any]) -> str | None:
    """Text of the final user message, or None if it is not plain text."""
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs or not isinstance(msgs[-1], dict):
        return None
    last = msgs[-1]
    if last.get("role") != "user":
        return None
    c = last.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and all(isinstance(b, dict) and b.get("type") == "text" for b in c):
        return "\n".join(str(b.get("text", "")) for b in c)
    return None
