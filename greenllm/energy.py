"""Energy and carbon *estimates*.

Closed-model providers do not publish energy per token, so every number here is
an assumption with a wide range. GreenLLM always reports a (low, high) range and
the savings it computes are relative (same factors applied to baseline and actual),
which is what makes them meaningful. Override the factors for your own setup;
for self-hosted models, measure GPU power directly instead (e.g. NVML / CodeCarbon).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Wh per 1,000 *output-equivalent* tokens, (low, high). ASSUMPTIONS, not measurements.
WH_PER_1K: dict[str, tuple[float, float]] = {
    "small": (0.02, 0.15),
    "medium": (0.1, 0.6),
    "large": (0.3, 2.0),
}
# Prefill (input) is far cheaper per token than decoding (output); cache reads cheaper still.
INPUT_WEIGHT = 0.15
CACHE_READ_WEIGHT = 0.02

_SMALL = re.compile(r"haiku|mini|nano|small|flash|lite|\b[1-9]b\b|[-_:][1-9]b", re.I)
_LARGE = re.compile(r"opus|gpt-4\.5|gpt-5|\bo1\b|\bo3\b|large|405b|70b|72b|ultra|pro\b", re.I)


def model_class(model: str) -> str:
    m = model or ""
    if _SMALL.search(m):
        return "small"
    if _LARGE.search(m):
        return "large"
    return "medium"


@dataclass
class Usage:
    input_tokens: int = 0          # uncached input
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def equivalent_tokens(self) -> float:
        return (
            self.output_tokens
            + INPUT_WEIGHT * (self.input_tokens + self.cache_write_tokens)
            + CACHE_READ_WEIGHT * self.cache_read_tokens
        )

    def without_cache(self) -> "Usage":
        """What this call would have cost if prompt caching were not used."""
        return Usage(
            input_tokens=self.input_tokens + self.cache_read_tokens + self.cache_write_tokens,
            output_tokens=self.output_tokens,
        )


def wh_range(equiv_tokens: float, cls: str) -> tuple[float, float]:
    lo, hi = WH_PER_1K.get(cls, WH_PER_1K["medium"])
    k = equiv_tokens / 1000.0
    return lo * k, hi * k


def gco2(wh: float, grid_gco2_per_kwh: float) -> float:
    return wh / 1000.0 * grid_gco2_per_kwh
