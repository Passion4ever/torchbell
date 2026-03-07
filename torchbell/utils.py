"""Formatting utilities."""

from typing import Dict

_PRE_PAD = 20


def fmt_time(seconds: float) -> str:
    d, r = divmod(int(seconds), 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    if d:
        return f"{d}d {h}h {m}m"
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _fmt_value(v) -> str:
    """Auto-format numeric value. Compatible with numpy types."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    abs_v = abs(v)
    if abs_v == 0:
        return "0"
    if abs_v < 1e-4:
        return f"{v:.2e}"
    return f"{v:.4f}"


def fmt_metrics(metrics: Dict[str, float], use_pre: bool = True) -> str:
    if not metrics:
        return ""
    max_key = max(len(k) for k in metrics)
    lines = []
    for k, v in metrics.items():
        lines.append(f"{k.ljust(max_key)}   {_fmt_value(v)}")
    content = "\n".join(
        line + " " * max(0, _PRE_PAD - len(line)) for line in lines
    )
    if use_pre:
        return f"<pre>{content}</pre>"
    return content
