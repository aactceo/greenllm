import json

import httpx
import pytest
from fastapi.testclient import TestClient

from greenllm.app import create_app
from greenllm.config import Settings
from greenllm.routing import decide


class Upstream:
    """Fake Anthropic/OpenAI. `small_reply` controls what the small model returns."""

    def __init__(self, small_reply="short answer", small_status=200):
        self.calls = []
        self.small_reply, self.small_status = small_reply, small_status

    def __call__(self, req):
        body = json.loads(req.content)
        self.calls.append(body)
        small = "haiku" in body["model"] or "mini" in body["model"]
        text = self.small_reply if small else "big answer"
        status = self.small_status if small else 200
        if status >= 400:
            return httpx.Response(status, json={"type": "error", "error": {"type": "not_found_error"}})
        if req.url.host == "api.anthropic.com":
            return httpx.Response(200, json={"model": body["model"],
                                             "content": [{"type": "text", "text": text}],
                                             "usage": {"input_tokens": 100, "output_tokens": 200}})
        return httpx.Response(200, json={"model": body["model"],
                                         "choices": [{"message": {"content": text}}],
                                         "usage": {"prompt_tokens": 100, "completion_tokens": 200}})


@pytest.fixture
def env(tmp_path):
    clients = []

    def make(upstream=None, **kw):
        up = upstream or Upstream()
        app = create_app(Settings(db_path=str(tmp_path / "t.db"), **kw),
                         transport=httpx.MockTransport(up))
        c = TestClient(app).__enter__()
        clients.append(c)
        return c, up
    yield make
    for c in clients:
        c.__exit__(None, None, None)


def ask(text, model="claude-opus-5-5", **extra):
    return {"model": model, "max_tokens": 500,
            "messages": [{"role": "user", "content": text}], **extra}


# ---------------------------------------------------------------- classifier
@pytest.mark.parametrize("text", [
    "Translate 'good morning' to French",
    "ترجم هذه الجملة إلى الإنجليزية: صباح الخير",
    "What is the capital of Japan?",
    "لخّص هذا النص في سطر واحد: الطقس اليوم مشمس.",
])
def test_simple_requests_route(text):
    assert decide(ask(text), "claude-haiku-5-5").route


@pytest.mark.parametrize("body", [
    ask("Design a scalable architecture for a payments system and explain the trade-offs"),
    ask("صمم معمارية نظام دفع وحلل المخاطر الأمنية"),
    ask("Fix this:\n```python\nprint(x)\n```"),
    ask("hi", tools=[{"name": "t", "input_schema": {"type": "object"}}]),
    ask("hi", thinking={"type": "enabled", "budget_tokens": 2000}),
    ask("hi", model="claude-haiku-5-5"),
    ask("x" * 3000),
    {"model": "claude-opus-5-5", "messages": [{"role": "user", "content": [
        {"type": "image", "source": {}}, {"type": "text", "text": "what is this"}]}]},
])
def test_complex_or_unsafe_requests_stay(body):
    assert not decide(body, "claude-haiku-5-5").route


# ---------------------------------------------------------------- routing in the gateway
def test_routing_on_sends_to_small_model(env):
    c, up = env(routing="on")
    r = c.post("/anthropic/v1/messages", json=ask("Translate 'cat' to Spanish"))
    assert up.calls[0]["model"] == "claude-haiku-5-5"
    assert r.headers["x-greenllm-routed"] == "claude-opus-5-5 -> claude-haiku-5-5"
    s = c.get("/stats").json()
    assert s["routed"] == 1 and s["saved_percent"] > 50


def test_routing_off_by_default_and_per_request_opt_out(env):
    c, up = env()
    c.post("/anthropic/v1/messages", json=ask("Translate 'cat' to Spanish"))
    c2, up2 = env(routing="on")
    c2.post("/anthropic/v1/messages", json=ask("Translate 'cat'"), headers={"x-greenllm-route": "off"})
    assert up.calls[0]["model"] == up2.calls[0]["model"] == "claude-opus-5-5"


def test_shadow_mode_measures_without_changing(env):
    c, up = env(routing="shadow")
    r = c.post("/openai/v1/chat/completions",
               json={"model": "gpt-5", "messages": [{"role": "user", "content": "Translate 'cat'"}]})
    assert up.calls[0]["model"] == "gpt-5"
    assert "x-greenllm-would-route" in r.headers
    s = c.get("/stats").json()
    assert s["would_route"] == 1 and s["routed"] == 0
    assert s["energy_wh"]["potential_from_routing"][1] > 0


@pytest.mark.parametrize("up", [Upstream(small_reply=""), Upstream(small_status=404)])
def test_escalates_when_small_model_fails(env, up):
    c, _ = env(upstream=up, routing="on")
    r = c.post("/anthropic/v1/messages", json=ask("Translate 'cat'"))
    assert [b["model"] for b in up.calls] == ["claude-haiku-5-5", "claude-opus-5-5"]
    assert r.json()["content"][0]["text"] == "big answer"
    assert r.headers["x-greenllm-escalated"] == "true"
    s = c.get("/stats").json()
    assert s["escalated"] == 1 and s["requests"] == 1


# ---------------------------------------------------------------- near-duplicate cache
def test_similar_question_reuses_answer(env):
    c, up = env(similar_cache=True)
    c.post("/anthropic/v1/messages", json=ask("What is the capital of France?", temperature=0))
    r = c.post("/anthropic/v1/messages", json=ask("what is the capital of france", temperature=0))
    assert len(up.calls) == 1
    assert r.headers["x-greenllm-cache"] == "similar-hit"
    assert c.get("/stats").json()["cache_hits_similar"] == 1


def test_similar_arabic_diacritics(env):
    c, up = env(similar_cache=True)
    c.post("/anthropic/v1/messages", json=ask("ما هي عاصمة فرنسا؟", temperature=0))
    c.post("/anthropic/v1/messages", json=ask("ما هِيَ عاصمةُ فرنسا", temperature=0))
    assert len(up.calls) == 1


@pytest.mark.parametrize("a,b", [
    ("What is 2345 + 3?", "What is 2345 + 4?"),
    ("List fruits that contain vitamin C", "List fruits that do not contain vitamin C"),
    ("Write a poem about the sea", "Write a recipe for bread"),
])
def test_similar_cache_guards(env, a, b):
    c, up = env(similar_cache=True)
    c.post("/anthropic/v1/messages", json=ask(a, temperature=0))
    c.post("/anthropic/v1/messages", json=ask(b, temperature=0))
    assert len(up.calls) == 2


def test_similar_requires_same_context(env):
    c, up = env(similar_cache=True)
    c.post("/anthropic/v1/messages", json=ask("What is the capital of France?", temperature=0,
                                              system="Answer in English"))
    c.post("/anthropic/v1/messages", json=ask("what is the capital of france", temperature=0,
                                              system="Answer in Arabic"))
    assert len(up.calls) == 2


def test_similar_off_by_default(env):
    c, up = env()
    c.post("/anthropic/v1/messages", json=ask("What is the capital of France?", temperature=0))
    c.post("/anthropic/v1/messages", json=ask("what is the capital of france", temperature=0))
    assert len(up.calls) == 2


def test_auth_errors_are_not_escalated(env):
    up = Upstream(small_status=401)
    c, _ = env(upstream=up, routing="on")
    r = c.post("/anthropic/v1/messages", json=ask("Translate 'cat'"))
    assert r.status_code == 401 and len(up.calls) == 1
