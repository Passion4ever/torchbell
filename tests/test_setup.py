"""Tests for torchbell.setup."""

from unittest.mock import patch, MagicMock

from torchbell.setup import setup


# ── setup: happy path ───────────────────────────

def _mock_get_happy(url, **kwargs):
    """Simulate getMe and getUpdates success."""
    resp = MagicMock()
    resp.status_code = 200
    if "getMe" in url:
        resp.json.return_value = {
            "ok": True,
            "result": {"first_name": "MyTrainBot"},
        }
    elif "getUpdates" in url:
        resp.json.return_value = {
            "ok": True,
            "result": [
                {
                    "update_id": 1,
                    "message": {"chat": {"id": 123456789}, "text": "/start"},
                }
            ],
        }
    return resp


@patch("requests.get", side_effect=_mock_get_happy)
def test_setup_happy_path(mock_get, capsys):
    token = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
    setup(token)
    out = capsys.readouterr().out
    assert "MyTrainBot" in out
    assert "123456789" in out
    # Token must NOT appear in output
    assert token not in out
    # Masked token should appear
    assert "ew11" in out
    assert "****" in out


@patch("requests.get", side_effect=_mock_get_happy)
def test_setup_does_not_leak_token(mock_get, capsys):
    token = "SECRET_TOKEN_VALUE_12345678"
    setup(token)
    out = capsys.readouterr().out
    assert token not in out


# ── setup: invalid token ────────────────────────

@patch("requests.get")
def test_setup_invalid_token(mock_get, capsys):
    resp = MagicMock()
    resp.status_code = 401
    resp.json.return_value = {"ok": False, "description": "Unauthorized"}
    mock_get.return_value = resp

    setup("bad-token")
    out = capsys.readouterr().out
    assert "invalid token" in out


# ── setup: no messages ──────────────────────────

def _mock_get_no_messages(url, **kwargs):
    resp = MagicMock()
    resp.status_code = 200
    if "getMe" in url:
        resp.json.return_value = {
            "ok": True,
            "result": {"first_name": "TestBot"},
        }
    elif "getUpdates" in url:
        resp.json.return_value = {"ok": True, "result": []}
    return resp


@patch("requests.get", side_effect=_mock_get_no_messages)
def test_setup_no_messages(mock_get, capsys):
    setup("some-token-1234")
    out = capsys.readouterr().out
    assert "No messages found" in out
    assert "/start" in out


# ── setup: network error ────────────────────────

@patch("requests.get")
def test_setup_network_error(mock_get, capsys):
    import requests
    mock_get.side_effect = requests.ConnectionError("connection refused")
    setup("some-token-1234")
    out = capsys.readouterr().out
    assert "network error" in out


# ── setup: getMe ok=False ───────────────────────

@patch("requests.get")
def test_setup_getme_not_ok(mock_get, capsys):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"ok": False, "description": "bot was blocked"}
    mock_get.return_value = resp

    setup("some-token-1234")
    out = capsys.readouterr().out
    assert "bot was blocked" in out


# ── setup: updates without message key ──────────

def _mock_get_updates_no_message(url, **kwargs):
    resp = MagicMock()
    resp.status_code = 200
    if "getMe" in url:
        resp.json.return_value = {
            "ok": True,
            "result": {"first_name": "TestBot"},
        }
    elif "getUpdates" in url:
        resp.json.return_value = {
            "ok": True,
            "result": [{"update_id": 1, "callback_query": {}}],
        }
    return resp


@patch("requests.get", side_effect=_mock_get_updates_no_message)
def test_setup_updates_without_message(mock_get, capsys):
    setup("some-token-1234")
    out = capsys.readouterr().out
    assert "No messages found" in out
