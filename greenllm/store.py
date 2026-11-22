"""SQLite storage: exact cache, near-duplicate cache and per-request meter log."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from typing import Any

from . import __author__, __version__
from . import similar as S
from .energy import Usage, gco2, model_class, wh_range

# Fields that do not change the model's answer and must not split the cache key.
_IGNORED_KEYS = {"stream", "stream_options", "metadata", "user"}

HIT_NONE, HIT_EXACT, HIT_SIMILAR = 0, 1, 2
SIMILAR_SCAN_LIMIT = 500


def _digest(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def cache_key(provider: str, path: str, body: dict[str, Any]) -> str:
    clean = {k: v for k, v in body.items() if k not in _IGNORED_KEYS}
    return _digest([provider, path, clean])


def context_key(provider: str, path: str, body: dict[str, Any]) -> str:
    """Key over everything except the final user message."""
    clean = {k: v for k, v in body.items() if k not in _IGNORED_KEYS}
    clean["messages"] = (body.get("messages") or [])[:-1]
    return _digest(["ctx", provider, path, clean])


class Store:
    def __init__(self, path: str):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY, created REAL, response TEXT,
                    input_tokens INT, output_tokens INT,
                    cache_read_tokens INT, cache_write_tokens INT);
                CREATE TABLE IF NOT EXISTS similar (
                    id INTEGER PRIMARY KEY, ctx TEXT, created REAL, guard TEXT,
                    vec TEXT, response TEXT, input_tokens INT, output_tokens INT,
                    cache_read_tokens INT, cache_write_tokens INT);
                CREATE INDEX IF NOT EXISTS similar_ctx ON similar(ctx, id);
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY, ts REAL, provider TEXT, model TEXT,
                    model_class TEXT, input_tokens INT, output_tokens INT,
                    cache_read_tokens INT, cache_write_tokens INT,
                    cache_hit INT, status INT, latency_ms REAL);
                """
            )
            cols = {r[1] for r in self._db.execute("PRAGMA table_info(requests)")}
            for col, typ in (("requested_model", "TEXT"), ("routed", "INT DEFAULT 0"),
                             ("would_route", "INT DEFAULT 0"), ("escalated", "INT DEFAULT 0")):
                if col not in cols:  # migrate databases from v0.1
                    self._db.execute(f"ALTER TABLE requests ADD COLUMN {col} {typ}")

    # ---- exact cache ---------------------------------------------------
    def cache_get(self, key: str, ttl: int) -> tuple[dict, Usage] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT created, response, input_tokens, output_tokens, cache_read_tokens,"
                " cache_write_tokens FROM cache WHERE key=?",
                (key,),
            ).fetchone()
        if not row or time.time() - row[0] > ttl:
            return None
        return json.loads(row[1]), Usage(*row[2:])

    def cache_put(self, key: str, response: dict, usage: Usage) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?,?)",
                (key, time.time(), json.dumps(response), usage.input_tokens,
                 usage.output_tokens, usage.cache_read_tokens, usage.cache_write_tokens),
            )

    # ---- near-duplicate cache -------------------------------------------
    def similar_get(self, ctx: str, text: str, threshold: float,
                    ttl: int) -> tuple[dict, Usage, float] | None:
        g, vec = S.guard(text), S.embed(text)
        with self._lock:
            rows = self._db.execute(
                "SELECT vec, response, input_tokens, output_tokens, cache_read_tokens,"
                " cache_write_tokens FROM similar WHERE ctx=? AND guard=? AND created>=?"
                " ORDER BY id DESC LIMIT ?",
                (ctx, g, time.time() - ttl, SIMILAR_SCAN_LIMIT),
            ).fetchall()
        best = None
        for row in rows:
            other = {int(k): v for k, v in json.loads(row[0]).items()}
            score = S.cosine(vec, other)
            if score >= threshold and (best is None or score > best[2]):
                best = (json.loads(row[1]), Usage(*row[2:]), score)
        return best

    def similar_put(self, ctx: str, text: str, response: dict, usage: Usage) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO similar (ctx, created, guard, vec, response, input_tokens,"
                " output_tokens, cache_read_tokens, cache_write_tokens) VALUES (?,?,?,?,?,?,?,?,?)",
                (ctx, time.time(), S.guard(text), json.dumps(S.embed(text)), json.dumps(response),
                 usage.input_tokens, usage.output_tokens, usage.cache_read_tokens,
                 usage.cache_write_tokens),
            )

    # ---- meter ---------------------------------------------------------
    def record(self, provider: str, model: str, usage: Usage, cache_hit: int,
               status: int, latency_ms: float, requested_model: str | None = None,
               routed: bool = False, would_route: bool = False, escalated: bool = False) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO requests (ts, provider, model, model_class, input_tokens,"
                " output_tokens, cache_read_tokens, cache_write_tokens, cache_hit, status,"
                " latency_ms, requested_model, routed, would_route, escalated)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), provider, model, model_class(model), usage.input_tokens,
                 usage.output_tokens, usage.cache_read_tokens, usage.cache_write_tokens,
                 int(cache_hit), status, latency_ms, requested_model or model,
                 int(routed), int(would_route), int(escalated)),
            )

    def stats(self, grid_gco2_per_kwh: float, since: float = 0.0) -> dict[str, Any]:
        with self._lock:
            rows = self._db.execute(
                "SELECT provider, model, model_class, input_tokens, output_tokens,"
                " cache_read_tokens, cache_write_tokens, cache_hit, status,"
                " COALESCE(requested_model, model), routed, would_route, escalated"
                " FROM requests WHERE ts>=?",
                (since,),
            ).fetchall()
        actual, baseline, shadow_saving = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
        tot = {"requests": 0, "cache_hits": 0, "cache_hits_exact": 0, "cache_hits_similar": 0,
               "routed": 0, "escalated": 0, "would_route": 0, "failed_attempts": 0,
               "input_tokens": 0, "output_tokens": 0,
               "cache_read_tokens": 0, "cache_write_tokens": 0}
        by_model: dict[str, int] = {}

        def add(acc, rng):
            acc[0] += rng[0]
            acc[1] += rng[1]

        for (provider, model, cls, i, o, cr, cw, hit, status, req_model,
             routed, would, esc) in rows:
            u = Usage(i, o, cr, cw)
            if status >= 400:
                # A failed call that still processed tokens (e.g. a routed attempt that was
                # escalated) is real energy spent: count it, never as savings.
                if i or o:
                    tot["failed_attempts"] += 1
                    add(actual, wh_range(u.equivalent_tokens(), cls))
                continue
            tot["requests"] += 1
            tot["cache_hits"] += hit > 0
            tot["cache_hits_exact"] += hit == HIT_EXACT
            tot["cache_hits_similar"] += hit == HIT_SIMILAR
            tot["routed"] += routed
            tot["escalated"] += esc
            tot["would_route"] += would
            by_model[f"{provider}:{model}"] = by_model.get(f"{provider}:{model}", 0) + 1
            # Baseline: the model the client asked for, no response cache, no prompt cache.
            add(baseline, wh_range(u.without_cache().equivalent_tokens(), model_class(req_model)))
            if hit:
                continue  # served locally: no tokens processed upstream
            for k, v in (("input_tokens", i), ("output_tokens", o),
                         ("cache_read_tokens", cr), ("cache_write_tokens", cw)):
                tot[k] += v
            add(actual, wh_range(u.equivalent_tokens(), cls))
            if would and not routed:
                full = wh_range(u.equivalent_tokens(), cls)
                small = wh_range(u.equivalent_tokens(), "small")
                add(shadow_saving, (full[0] - small[0], full[1] - small[1]))
        saved = [baseline[0] - actual[0], baseline[1] - actual[1]]
        pct = 100.0 * saved[1] / baseline[1] if baseline[1] else 0.0
        return {
            "tool": f"GreenLLM {__version__} by {__author__}",
            **tot,
            "by_model": by_model,
            "energy_wh": {"actual": actual, "baseline": baseline, "saved": saved,
                          "potential_from_routing": shadow_saving},
            "co2_g": {
                "actual": [gco2(x, grid_gco2_per_kwh) for x in actual],
                "saved": [gco2(x, grid_gco2_per_kwh) for x in saved],
            },
            "saved_percent": round(pct, 1),
            "note": "Energy values are (low, high) estimates from assumed factors; "
                    "saved_percent is relative to the same factors.",
        }
