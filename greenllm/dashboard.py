"""Minimal Arabic (RTL) dashboard served at GET /."""
from __future__ import annotations

from html import escape
from typing import Any


def _rng(pair: list[float], unit: str) -> str:
    return f"{pair[0]:.3g} – {pair[1]:.3g} {unit}"


def render_dashboard(s: dict[str, Any]) -> str:
    e, c = s["energy_wh"], s["co2_g"]
    hit_rate = 100.0 * s["cache_hits"] / s["requests"] if s["requests"] else 0.0
    rows = "".join(
        f"<tr><td>{escape(m)}</td><td>{n}</td></tr>" for m, n in sorted(s["by_model"].items())
    )
    cards = [
        ("الطلبات", f"{s['requests']}"),
        ("نسبة الإجابات من الذاكرة المؤقتة", f"{hit_rate:.1f}%"),
        ("التوفير التقديري", f"{s['saved_percent']}%"),
        ("الطاقة المستهلكة (تقدير)", _rng(e["actual"], "Wh")),
        ("الطاقة الموفَّرة (تقدير)", _rng(e["saved"], "Wh")),
        ("CO₂e الموفَّر (تقدير)", _rng(c["saved"], "g")),
        ("رموز الإدخال / الإخراج", f"{s['input_tokens']} / {s['output_tokens']}"),
        ("رموز مقروءة من Prompt Cache", f"{s['cache_read_tokens']}"),
        ("إصابات مطابقة / متشابهة", f"{s['cache_hits_exact']} / {s['cache_hits_similar']}"),
        ("طلبات وُجّهت لنموذج أصغر", f"{s['routed']} (أُعيد {s['escalated']} للنموذج الأصلي)"),
        ("قابلة للتوجيه (وضع المراقبة)", f"{s['would_route']}"),
        ("توفير محتمل من التوجيه (تقدير)", _rng(e["potential_from_routing"], "Wh")),
    ]
    card_html = "".join(f"<div class=card><span>{k}</span><b>{v}</b></div>" for k, v in cards)
    return f"""<!doctype html><html lang=ar dir=rtl><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>GreenLLM</title>
<meta http-equiv=refresh content=15>
<style>
:root{{--bg:#f6f8f5;--fg:#1d2a1f;--card:#fff;--muted:#5d6b60;--accent:#2f7d4a}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111613;--fg:#e4ece5;--card:#1a211c;--muted:#9aa89d;--accent:#6fcf8d}}}}
body{{margin:0;padding:24px 16px;font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg)}}
h1{{margin:0 0 4px;color:var(--accent)}} p{{color:var(--muted);margin:0 0 20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}}
.card{{background:var(--card);border-radius:10px;padding:14px;display:flex;flex-direction:column;gap:6px}}
.card span{{color:var(--muted);font-size:.9em}} .card b{{font-size:1.25em}}
table{{margin-top:20px;border-collapse:collapse;width:100%;max-width:600px}}
td{{padding:6px 8px;border-bottom:1px solid var(--muted)}}
</style></head><body>
<h1>GreenLLM</h1><p>{escape(s['note'])}</p>
<div class=grid>{card_html}</div>
<table><tr><td><b>النموذج</b></td><td><b>الطلبات</b></td></tr>{rows}</table>
</body></html>"""
