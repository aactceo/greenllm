import json

import httpx
import pytest
from fastapi.testclient import TestClient

from greenllm.app import create_app
from greenllm.config import Settings
from greenllm.energy import model_class


class FakeUpstream:
    def __init__(self):
        self.calls: list[httpx.Request] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls.append(req)
        body = json.loads(req.content or b"{}")
        if req.url.host == "api.anthropic.com":
            if body.get("stream"):
                events = [
                    {"type": "message_start", "message": {"usage": {
                        "input_tokens": 40, "output_tokens": 1,
                        "cache_read_input_tokens": 900}}},
                    {"type": "content_block_delta", "delta": {"text": "hi"}},
                    {"type": "message_delta", "usage": {"output_tokens": 25}},
                ]
                sse = "".join(f"event: x\ndata: {json.dumps(e)}\n\n" for e in events)
                return httpx.Response(200, content=sse.encode(),
                                      headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json={
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 100, "output_tokens": 50}})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 20,
                      "prompt_tokens_details": {"cached_tokens": 800}}})


@pytest.fixture
def env(tmp_path):
    clients = []

    def make(**kw):
        up = FakeUpstream()
        app = create_app(Settings(db_path=str(tmp_path / "t.db"), **kw),
                         transport=httpx.MockTransport(up))
        c = TestClient(app).__enter__()
        clients.append(c)
        return c, up
    yield make
    for c in clients:
        c.__exit__(None, None, None)


def msg(**extra):
    return {"model": "claude-haiku-4-5", "max_tokens": 500,
            "messages": [{"role": "user", "content": "hello"}], **extra}


def test_cache_hit_skips_upstream_for_deterministic_requests(env):
    c, up = env()
    r1 = c.post("/anthropic/v1/messages", json=msg(temperature=0), headers={"x-api-key": "k"})
    r2 = c.post("/anthropic/v1/messages", json=msg(temperature=0), headers={"x-api-key": "k"})
    assert r1.headers["x-greenllm-cache"] == "miss"
    assert r2.headers["x-greenllm-cache"] == "hit"
    assert r2.json() == r1.json()
    assert len(up.calls) == 1
    assert up.calls[0].headers["x-api-key"] == "k"
    s = c.get("/stats").json()
    assert s["requests"] == 2 and s["cache_hits"] == 1
    assert s["saved_percent"] == pytest.approx(50.0, abs=0.1)


def test_no_cache_for_sampled_requests(env):
    c, up = env()
    c.post("/anthropic/v1/messages", json=msg(temperature=1))
    c.post("/anthropic/v1/messages", json=msg(temperature=1))
    assert len(up.calls) == 2


def test_header_can_force_cache(env):
    c, up = env()
    for _ in range(2):
        c.post("/anthropic/v1/messages", json=msg(), headers={"x-greenllm-cache": "on"})
    assert len(up.calls) == 1
    assert "x-greenllm-cache" not in up.calls[0].headers


def test_auto_prompt_cache_marks_long_system(env):
    c, up = env(auto_prompt_cache_min_chars=100)
    r = c.post("/anthropic/v1/messages", json=msg(system="x" * 500))
    sent = json.loads(up.calls[0].content)
    assert sent["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert r.headers["x-greenllm-prompt-cache"] == "added"


def test_auto_prompt_cache_respects_user_markup_and_short_prompts(env):
    c, up = env(auto_prompt_cache_min_chars=100)
    c.post("/anthropic/v1/messages", json=msg(system="short"))
    own = [{"type": "text", "text": "y" * 500, "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
    c.post("/anthropic/v1/messages", json=msg(system=own))
    assert json.loads(up.calls[0].content)["system"] == "short"
    assert json.loads(up.calls[1].content)["system"] == own


def test_output_cap(env):
    c, up = env(max_output_cap=200)
    c.post("/anthropic/v1/messages", json=msg())
    c.post("/openai/v1/chat/completions", json={"model": "gpt-4o-mini", "messages": []})
    assert json.loads(up.calls[0].content)["max_tokens"] == 200
    assert json.loads(up.calls[1].content)["max_completion_tokens"] == 200


def test_openai_usage_counts_cached_prompt_tokens(env):
    c, _ = env()
    c.post("/openai/v1/chat/completions", json={"model": "gpt-4o", "messages": []})
    s = c.get("/stats").json()
    assert (s["input_tokens"], s["cache_read_tokens"], s["output_tokens"]) == (200, 800, 20)
    assert s["saved_percent"] > 0  # prompt caching saved prefill work


def test_streaming_is_relayed_and_metered(env):
    c, up = env()
    with c.stream("POST", "/anthropic/v1/messages", json=msg(stream=True)) as r:
        text = b"".join(r.iter_raw())
    assert b"message_delta" in text
    s = c.get("/stats").json()
    assert (s["input_tokens"], s["output_tokens"], s["cache_read_tokens"]) == (40, 25, 900)


def test_openai_stream_requests_usage(env):
    c, up = env()
    c.post("/openai/v1/chat/completions", json={"model": "gpt-4o", "messages": [], "stream": True})
    assert json.loads(up.calls[0].content)["stream_options"] == {"include_usage": True}


def test_other_paths_pass_through(env):
    c, up = env()
    c.post("/anthropic/v1/messages/count_tokens", json=msg())
    assert up.calls[0].url.path == "/v1/messages/count_tokens"


def test_dashboard_renders(env):
    c, _ = env()
    assert "GreenLLM" in c.get("/").text


@pytest.mark.parametrize("model,cls", [
    ("claude-haiku-4-5", "small"), ("gpt-4o-mini", "small"), ("llama-3.1-8b", "small"),
    ("claude-opus-4-1", "large"), ("llama-3.1-70b", "large"), ("claude-sonnet-4-5", "medium"),
])
def test_model_class(model, cls):
    assert model_class(model) == cls
