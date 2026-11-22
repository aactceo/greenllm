#!/usr/bin/env python3
"""Stop hook: sum the session's token usage from the transcript and show a short
energy/carbon estimate. Standalone (stdlib only). Silence with GREEN_AI_QUIET=1.

The factors mirror greenllm/energy.py and are ASSUMPTIONS, shown as a range.
"""
import json
import os
import re
import sys

WH_PER_1K = {"small": (0.02, 0.15), "medium": (0.1, 0.6), "large": (0.3, 2.0)}
INPUT_WEIGHT, CACHE_READ_WEIGHT = 0.15, 0.02
GRID = float(os.environ.get("GREENLLM_GRID_GCO2_PER_KWH", "450"))


def model_class(m: str) -> str:
    if re.search(r"haiku|mini|nano|flash", m, re.I):
        return "small"
    if re.search(r"opus", m, re.I):
        return "large"
    return "medium"


def main() -> None:
    if os.environ.get("GREEN_AI_QUIET") == "1":
        return
    try:
        event = json.load(sys.stdin)
        path = event["transcript_path"]
    except (ValueError, KeyError):
        return
    messages = {}  # message id -> (model, usage); streamed entries repeat, keep the last
    try:
        with open(os.path.expanduser(path), encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                msg = entry.get("message") or {}
                if entry.get("type") == "assistant" and msg.get("usage"):
                    messages[msg.get("id") or len(messages)] = (msg.get("model", ""), msg["usage"])
    except OSError:
        return
    if not messages:
        return

    tot = {"in": 0, "out": 0, "read": 0, "write": 0}
    wh = [0.0, 0.0]
    wh_nocache = [0.0, 0.0]
    for model, u in messages.values():
        i, o = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
        cr = int(u.get("cache_read_input_tokens") or 0)
        cw = int(u.get("cache_creation_input_tokens") or 0)
        tot["in"] += i; tot["out"] += o; tot["read"] += cr; tot["write"] += cw
        lo, hi = WH_PER_1K[model_class(model)]
        eq = o + INPUT_WEIGHT * (i + cw) + CACHE_READ_WEIGHT * cr
        eq0 = o + INPUT_WEIGHT * (i + cw + cr)
        wh[0] += lo * eq / 1000; wh[1] += hi * eq / 1000
        wh_nocache[0] += lo * eq0 / 1000; wh_nocache[1] += hi * eq0 / 1000

    co2 = [x / 1000 * GRID for x in wh]
    saved = 100 * (1 - wh[1] / wh_nocache[1]) if wh_nocache[1] else 0
    msg = (
        f"🌱 Session so far: {len(messages)} model calls, "
        f"{tot['in'] + tot['write']:,} new input + {tot['read']:,} cached input + {tot['out']:,} output tokens. "
        f"Estimated {wh[0]:.2g}–{wh[1]:.2g} Wh, {co2[0]:.2g}–{co2[1]:.2g} g CO₂e "
        f"(prompt caching saved ~{saved:.0f}%; estimates, not measurements)."
    )
    print(json.dumps({"systemMessage": msg, "suppressOutput": True}))


if __name__ == "__main__":
    main()
