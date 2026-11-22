"""Settings, read from environment variables (all prefixed GREENLLM_)."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    anthropic_base: str = "https://api.anthropic.com"
    openai_base: str = "https://api.openai.com"
    db_path: str = "greenllm.db"
    # off | deterministic (only temperature==0 requests) | all
    cache_mode: str = "deterministic"
    cache_ttl_seconds: int = 7 * 24 * 3600
    # Add Anthropic cache_control to long system prompts / tools automatically.
    auto_prompt_cache: bool = True
    auto_prompt_cache_min_chars: int = 4000  # ~1000 tokens; shorter prompts can't be cached anyway
    # Hard ceiling on requested output tokens (0 = disabled).
    max_output_cap: int = 0
    # Grid carbon intensity used for estimates, gCO2e per kWh.
    grid_gco2_per_kwh: float = 450.0
    timeout_seconds: float = 600.0
    # Near-duplicate cache (same gating as the exact cache). off | on
    similar_cache: bool = False
    similar_threshold: float = 0.9
    # Model routing: off | shadow (measure only) | on
    routing: str = "off"
    route_anthropic_model: str = "claude-haiku-5-5"
    route_openai_model: str = "gpt-4o-mini"
    route_max_score: int = 0

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        return cls(
            anthropic_base=e("GREENLLM_ANTHROPIC_BASE", cls.anthropic_base).rstrip("/"),
            openai_base=e("GREENLLM_OPENAI_BASE", cls.openai_base).rstrip("/"),
            db_path=e("GREENLLM_DB", cls.db_path),
            cache_mode=e("GREENLLM_CACHE_MODE", cls.cache_mode).lower(),
            cache_ttl_seconds=int(e("GREENLLM_CACHE_TTL", cls.cache_ttl_seconds)),
            auto_prompt_cache=_bool("GREENLLM_AUTO_PROMPT_CACHE", cls.auto_prompt_cache),
            auto_prompt_cache_min_chars=int(
                e("GREENLLM_AUTO_PROMPT_CACHE_MIN_CHARS", cls.auto_prompt_cache_min_chars)
            ),
            max_output_cap=int(e("GREENLLM_MAX_OUTPUT_CAP", cls.max_output_cap)),
            grid_gco2_per_kwh=float(e("GREENLLM_GRID_GCO2_PER_KWH", cls.grid_gco2_per_kwh)),
            timeout_seconds=float(e("GREENLLM_TIMEOUT", cls.timeout_seconds)),
            similar_cache=_bool("GREENLLM_SIMILAR_CACHE", cls.similar_cache),
            similar_threshold=float(e("GREENLLM_SIMILAR_THRESHOLD", cls.similar_threshold)),
            routing=e("GREENLLM_ROUTING", cls.routing).lower(),
            route_anthropic_model=e("GREENLLM_ROUTE_ANTHROPIC_MODEL", cls.route_anthropic_model),
            route_openai_model=e("GREENLLM_ROUTE_OPENAI_MODEL", cls.route_openai_model),
            route_max_score=int(e("GREENLLM_ROUTE_MAX_SCORE", cls.route_max_score)),
        )
