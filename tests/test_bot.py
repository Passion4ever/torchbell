"""Tests for torchbell.bot (TelegramBot)."""

from unittest.mock import patch, MagicMock

import requests

from torchbell.bot import TelegramBot, _MAX_RETRIES


def _make_bot():
    return TelegramBot("fake-token", 12345)


# ── _do_send ───────────────────────────────────────

@patch("torchbell.bot.requests.post")
def test_do_send_success(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {"ok": True, "result": {"message_id": 42}},
    )
    bot = _make_bot()
    mid = bot._do_send("hello")
    assert mid == 42
    mock_post.assert_called_once()


@patch("torchbell.bot.requests.post")
def test_do_send_truncates_long_text(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {"ok": True, "result": {"message_id": 1}},
    )
    bot = _make_bot()
    bot._do_send("x" * 5000)
    call_json = mock_post.call_args[1]["json"]
    assert len(call_json["text"]) == 4096


@patch("torchbell.bot.requests.post")
def test_do_send_api_error(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {"ok": False, "description": "bad request"},
    )
    bot = _make_bot()
    mid = bot._do_send("hello")
    assert mid is None


@patch("torchbell.bot.requests.post")
def test_do_send_http_error(mock_post):
    mock_post.return_value = MagicMock(status_code=500)
    mock_post.return_value.raise_for_status.side_effect = requests.HTTPError("500")
    bot = _make_bot()
    try:
        bot._do_send("hello")
        assert False, "Should have raised"
    except requests.HTTPError:
        pass


# ── _do_edit ───────────────────────────────────────

@patch("torchbell.bot.requests.post")
def test_do_edit_success(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {"ok": True},
    )
    bot = _make_bot()
    bot._do_edit(1, "updated")
    mock_post.assert_called_once()


@patch("torchbell.bot.requests.post")
def test_do_edit_not_modified_silent(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {"ok": False, "description": "message is not modified"},
    )
    bot = _make_bot()
    # Should not raise — "message is not modified" is silently ignored
    bot._do_edit(1, "same text")


@patch("torchbell.bot.requests.post")
def test_do_edit_other_error(mock_post, capsys):
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {"ok": False, "description": "something else"},
    )
    bot = _make_bot()
    bot._do_edit(1, "text")
    captured = capsys.readouterr()
    assert "something else" in captured.out


# ── send_sync retries ─────────────────────────────

@patch("torchbell.bot.time.sleep")
@patch("torchbell.bot.requests.post")
def test_send_sync_success(mock_post, mock_sleep):
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {"ok": True, "result": {"message_id": 7}},
    )
    bot = _make_bot()
    assert bot.send_sync("hi") == 7
    mock_sleep.assert_not_called()


@patch("torchbell.bot.time.sleep")
@patch("torchbell.bot.requests.post")
def test_send_sync_retry_then_success(mock_post, mock_sleep):
    fail = MagicMock()
    fail.raise_for_status.side_effect = requests.HTTPError("err")
    success = MagicMock(
        status_code=200,
        json=lambda: {"ok": True, "result": {"message_id": 8}},
    )
    mock_post.side_effect = [fail, success]
    bot = _make_bot()
    assert bot.send_sync("hi") == 8
    assert mock_sleep.call_count == 1


@patch("torchbell.bot.time.sleep")
@patch("torchbell.bot.requests.post")
def test_send_sync_all_fail(mock_post, mock_sleep):
    fail = MagicMock()
    fail.raise_for_status.side_effect = requests.HTTPError("err")
    mock_post.return_value = fail
    bot = _make_bot()
    assert bot.send_sync("hi") is None
    assert mock_sleep.call_count == _MAX_RETRIES


# ── edit_sync retries ─────────────────────────────

@patch("torchbell.bot.time.sleep")
@patch("torchbell.bot.requests.post")
def test_edit_sync_success(mock_post, mock_sleep):
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {"ok": True},
    )
    bot = _make_bot()
    bot.edit_sync(1, "text")
    mock_sleep.assert_not_called()


@patch("torchbell.bot.time.sleep")
@patch("torchbell.bot.requests.post")
def test_edit_sync_retry_then_success(mock_post, mock_sleep):
    fail = MagicMock()
    fail.raise_for_status.side_effect = requests.HTTPError("err")
    success = MagicMock(
        status_code=200,
        json=lambda: {"ok": True},
    )
    mock_post.side_effect = [fail, success]
    bot = _make_bot()
    bot.edit_sync(1, "text")
    assert mock_sleep.call_count == 1


@patch("torchbell.bot.time.sleep")
@patch("torchbell.bot.requests.post")
def test_edit_sync_all_fail(mock_post, mock_sleep, capsys):
    fail = MagicMock()
    fail.raise_for_status.side_effect = requests.HTTPError("err")
    mock_post.return_value = fail
    bot = _make_bot()
    bot.edit_sync(1, "text")
    assert mock_sleep.call_count == _MAX_RETRIES
    captured = capsys.readouterr()
    assert "edit failed" in captured.out
