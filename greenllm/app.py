"""FastAPI gateway.

Point clients at it instead of the provider:
  Anthropic:  ANTHROPIC_BASE_URL=http://localhost:8787/anthropic
  OpenAI:     OPENAI_BASE_URL=http://localhost:8787/openai/v1
Auth headers are forwarded untouched; GreenLLM never stores API keys.
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from . import providers as P
from . import similar as S
from .config import Settings
from .dashboard import render_dashboard
from .energy import Usage
from .routing import decide, response_text
from .store import HIT_EXACT, HIT_NONE, HIT_SIMILAR, Store, cache_key, context_key

HOP_HEADERS = {"host", "content-length", "connection", "accept-encoding",
               "transfer-encoding", "keep-alive", "x-greenllm-cache", "x-greenllm-route"}
RESP_DROP = {"content-length", "content-encoding", "transfer-encoding", "connection"}
OPTIMIZED_PATHS = {"anthropic": "v1/messages", "openai": "v1/chat/completions"}


def create_app(settings: Settings | None = None,
               transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = Store(settings.db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(timeout=settings.timeout_seconds, transport=transport)
        yield
        await app.state.client.aclose()

    app = FastAPI(title="GreenLLM", lifespan=lifespan)
    app.state.store = store
    bases = {"anthropic": settings.anthropic_base, "openai": settings.openai_base}

    def cache_allowed(body: dict[str, Any], header: str | None) -> bool:
        if header == "off" or settings.cache_mode == "off" or body.get("stream"):
            return False
        if header == "on" or settings.cache_mode == "all":
            return True
        return body.get("temperature") == 0  # deterministic mode

    async def proxy(provider: str, path: str, request: Request) -> Response:
        url = f"{bases[provider]}/{path}"
        if request.url.query:
            url += f"?{request.url.query}"
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
        raw = await request.body()
        client: httpx.AsyncClient = request.app.state.client

        if request.method != "POST" or path != OPTIMIZED_PATHS[provider]:
            r = await client.request(request.method, url, headers=headers, content=raw)
            return _plain(r)

        try:
            body = json.loads(raw)
        except ValueError:
            r = await client.post(url, headers=headers, content=raw)
            return _plain(r)

        model = str(body.get("model", ""))
        started = time.perf_counter()
        gl_headers: dict[str, str] = {}
        cache_header = request.headers.get("x-greenllm-cache")

        # 1) Exact response cache: an identical request costs zero upstream compute.
        use_cache = cache_allowed(body, cache_header)
        key = cache_key(provider, path, body) if use_cache else ""
        if use_cache and (hit := store.cache_get(key, settings.cache_ttl_seconds)):
            resp, usage = hit
            store.record(provider, model, usage, HIT_EXACT, 200, _ms(started))
            return JSONResponse(resp, headers={"x-greenllm-cache": "hit"})

        # 2) Near-duplicate cache: same context, near-identical final question.
        last = P.last_user_text(body)
        use_similar = (use_cache and settings.similar_cache and last is not None
                       and S.eligible(last))
        ctx = context_key(provider, path, body) if use_similar else ""
        if use_similar and (hit := store.similar_get(
                ctx, last, settings.similar_threshold, settings.cache_ttl_seconds)):
            resp, usage, score = hit
            store.record(provider, model, usage, HIT_SIMILAR, 200, _ms(started))
            return JSONResponse(resp, headers={"x-greenllm-cache": "similar-hit",
                                               "x-greenllm-similarity": f"{score:.3f}"})

        # 3) Request rewrites.
        body = P.cap_output(body, provider, settings.max_output_cap)
        if provider == "anthropic" and settings.auto_prompt_cache:
            body, added = P.add_anthropic_prompt_cache(body, settings.auto_prompt_cache_min_chars)
            if added:
                gl_headers["x-greenllm-prompt-cache"] = "added"
        if provider == "openai":
            body = P.ensure_openai_stream_usage(body)

        # 4) Model routing.
        routed = would_route = False
        route_header = request.headers.get("x-greenllm-route")
        if settings.routing in {"on", "shadow"} and route_header != "off":
            target = (settings.route_anthropic_model if provider == "anthropic"
                      else settings.route_openai_model)
            d = decide(body, target, settings.route_max_score)
            gl_headers["x-greenllm-route-reason"] = "; ".join(d.reasons)[:200]
            if d.route and settings.routing == "on":
                body = {**body, "model": d.target}
                routed = True
                gl_headers["x-greenllm-routed"] = f"{model} -> {d.target}"
            elif d.route:
                would_route = True
                gl_headers["x-greenllm-would-route"] = f"{model} -> {d.target}"
        served = str(body.get("model", model))
        meta = dict(requested_model=model, routed=routed, would_route=would_route)

        if body.get("stream"):
            # Streams are relayed as they arrive, so a routed stream cannot be escalated.
            req = client.build_request("POST", url, headers=headers, content=json.dumps(body).encode())
            r = await client.send(req, stream=True)
            parser = P.SSEUsageParser(provider)

            async def relay():
                try:
                    async for chunk in r.aiter_bytes():
                        parser.feed(chunk)
                        yield chunk
                finally:
                    await r.aclose()
                    store.record(provider, served, parser.usage, HIT_NONE, r.status_code,
                                 _ms(started), **meta)

            return StreamingResponse(relay(), status_code=r.status_code,
                                     headers={**_resp_headers(r), **gl_headers})

        parse = P.anthropic_usage if provider == "anthropic" else P.openai_usage
        r, data, usage = await _call(client, url, headers, body, parse)

        # 5) Escalation: if the small model failed or returned nothing, retry as asked.
        # Auth errors are not the small model's fault: retrying would only waste a call.
        if routed and r.status_code not in (401, 403) and (
                r.status_code >= 400 or not response_text(provider, data).strip()):
            status = r.status_code if r.status_code >= 400 else 599
            store.record(provider, served, usage, HIT_NONE, status, _ms(started), **meta)
            body = {**body, "model": model}
            served, meta = model, dict(requested_model=model, escalated=True)
            gl_headers.pop("x-greenllm-routed", None)
            gl_headers["x-greenllm-escalated"] = "true"
            r, data, usage = await _call(client, url, headers, body, parse)

        if data is None:
            store.record(provider, served, Usage(), HIT_NONE, r.status_code, _ms(started), **meta)
            return _plain(r)
        store.record(provider, served, usage, HIT_NONE, r.status_code, _ms(started), **meta)
        if use_cache and r.status_code == 200:
            store.cache_put(key, data, usage)
            if use_similar:
                store.similar_put(ctx, last, data, usage)
            gl_headers["x-greenllm-cache"] = "miss"
        return JSONResponse(data, status_code=r.status_code,
                            headers={**_resp_headers(r), **gl_headers})

    @app.api_route("/anthropic/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def anthropic(path: str, request: Request):
        return await proxy("anthropic", path, request)

    @app.api_route("/openai/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def openai(path: str, request: Request):
        return await proxy("openai", path, request)

    @app.get("/stats")
    async def stats(since: float = 0.0):
        return store.stats(settings.grid_gco2_per_kwh, since)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return render_dashboard(store.stats(settings.grid_gco2_per_kwh))

    return app


async def _call(client: httpx.AsyncClient, url: str, headers: dict[str, str],
                body: dict[str, Any], parse) -> tuple[httpx.Response, Any, Usage]:
    r = await client.post(url, headers=headers, content=json.dumps(body).encode())
    try:
        data = r.json()
    except ValueError:
        return r, None, Usage()
    return r, data, parse(data.get("usage") if isinstance(data, dict) else None)


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _resp_headers(r: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in r.headers.items() if k.lower() not in RESP_DROP}


def _plain(r: httpx.Response) -> Response:
    return Response(r.content, status_code=r.status_code, headers=_resp_headers(r))
