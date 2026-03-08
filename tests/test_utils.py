"""Tests for torchbell.utils."""

from torchbell.utils import fmt_time, _fmt_value, fmt_metrics, mask_credential


# ── fmt_time ───────────────────────────────────────

def test_fmt_time_seconds():
    assert fmt_time(45) == "45s"

def test_fmt_time_minutes():
    assert fmt_time(125) == "2m 5s"

def test_fmt_time_hours():
    assert fmt_time(3661) == "1h 1m 1s"

def test_fmt_time_days():
    assert fmt_time(90061) == "1d 1h 1m"

def test_fmt_time_zero():
    assert fmt_time(0) == "0s"


# ── _fmt_value ─────────────────────────────────────

def test_fmt_value_zero():
    assert _fmt_value(0) == "0"

def test_fmt_value_normal():
    assert _fmt_value(3.14159) == "3.1416"

def test_fmt_value_small():
    result = _fmt_value(1e-6)
    assert "e" in result  # scientific notation

def test_fmt_value_negative():
    assert _fmt_value(-2.5) == "-2.5000"

def test_fmt_value_non_numeric():
    assert _fmt_value("hello") == "hello"

def test_fmt_value_integer():
    assert _fmt_value(42) == "42.0000"


# ── fmt_metrics ────────────────────────────────────

def test_fmt_metrics_empty():
    assert fmt_metrics({}) == ""

def test_fmt_metrics_single():
    result = fmt_metrics({"loss": 0.5})
    assert "<pre>" in result
    assert "loss" in result
    assert "0.5000" in result

def test_fmt_metrics_multiple():
    result = fmt_metrics({"loss": 0.5, "accuracy": 0.95})
    assert "loss" in result
    assert "accuracy" in result

def test_fmt_metrics_no_pre():
    result = fmt_metrics({"loss": 0.5}, use_pre=False)
    assert "<pre>" not in result
    assert "loss" in result

# ── mask_credential ────────────────────────────────

def test_mask_credential_normal():
    assert mask_credential("abcdefgh") == "****efgh"


def test_mask_credential_short():
    assert mask_credential("abc") == "***"


def test_mask_credential_exact_show():
    assert mask_credential("abcd") == "****"


def test_mask_credential_custom_show():
    assert mask_credential("abcdefgh", show=2) == "******gh"


def test_mask_credential_long_token():
    token = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
    masked = mask_credential(token)
    assert masked.endswith("ew11")
    assert token not in masked
    assert "*" in masked


# ── fmt_metrics (continued) ───────────────────────

def test_fmt_metrics_padding():
    result = fmt_metrics({"loss": 0.5})
    # Each line inside <pre> should have trailing padding
    inner = result.replace("<pre>", "").replace("</pre>", "")
    for line in inner.split("\n"):
        assert line.endswith(" ")
