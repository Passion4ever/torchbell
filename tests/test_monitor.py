"""Tests for torchbell.monitor (TorchBell)."""

import os
from unittest.mock import patch, MagicMock

import pytest

from torchbell.monitor import TorchBell


def _make_bell(**kwargs):
    """Create a TorchBell with a mocked bot."""
    defaults = dict(run_name="test-run", token="tok", chat_id=123)
    defaults.update(kwargs)
    with patch("torchbell.monitor.TelegramBot") as MockBot:
        mock_bot = MagicMock()
        mock_bot.send_sync.return_value = 1  # message_id
        MockBot.return_value = mock_bot
        bell = TorchBell(**defaults)
    return bell, mock_bot


# ── _detect_is_main ───────────────────────────────

def test_detect_is_main_rank_zero():
    assert TorchBell._detect_is_main(None, 0) is True

def test_detect_is_main_rank_nonzero():
    assert TorchBell._detect_is_main(None, 1) is False

def test_detect_is_main_accelerator():
    acc = MagicMock()
    acc.is_main_process = False
    assert TorchBell._detect_is_main(acc, None) is False

def test_detect_is_main_accelerator_true():
    acc = MagicMock()
    acc.is_main_process = True
    assert TorchBell._detect_is_main(acc, None) is True

def test_detect_is_main_env_rank(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    assert TorchBell._detect_is_main(None, None) is True

def test_detect_is_main_env_rank_nonzero(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    assert TorchBell._detect_is_main(None, None) is False

def test_detect_is_main_env_local_rank(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert TorchBell._detect_is_main(None, None) is True

def test_detect_is_main_env_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_PROCID", "0")
    assert TorchBell._detect_is_main(None, None) is True

def test_detect_is_main_default():
    # No rank, no accelerator, no env vars → single GPU → True
    assert TorchBell._detect_is_main(None, None) is True


# ── Constructor ────────────────────────────────────

def test_missing_token_raises():
    with pytest.raises(ValueError, match="token and chat_id required"):
        TorchBell(run_name="test")

def test_env_var_loading(monkeypatch):
    monkeypatch.setenv("TG_BOT_TOKEN", "env-tok")
    monkeypatch.setenv("TG_CHAT_ID", "999")
    with patch("torchbell.monitor.TelegramBot"):
        bell = TorchBell(run_name="test")
    assert bell._is_main is True

def test_non_main_process_no_bot():
    with patch("torchbell.monitor.TelegramBot"):
        bell = TorchBell(run_name="test", token="t", chat_id=1, rank=1)
    assert bell._bot is None


# ── Lifecycle: start → log → finish ───────────────

def test_start_log_finish():
    bell, mock_bot = _make_bell()
    bell.start(total=100)

    assert bell._state == "running"
    assert bell._total == 100
    mock_bot.send_sync.assert_called_once()  # initial status message

    bell.log(50, {"loss": 0.5})
    assert bell._step == 50
    assert bell._latest_metrics == {"loss": 0.5}

    bell.finish(final_metrics={"loss": 0.1})
    assert bell._state == "finished"
    # edit_sync for status update + send_sync for completion notification
    assert mock_bot.edit_sync.call_count == 1
    assert mock_bot.send_sync.call_count == 2  # start + finish


# ── Lifecycle: start → error ──────────────────────

def test_start_error():
    bell, mock_bot = _make_bell()
    bell.start()
    assert bell._state == "running"

    exc = ValueError("boom")
    try:
        raise exc
    except ValueError:
        bell.error(exc)

    assert bell._state == "crashed"
    assert mock_bot.edit_sync.call_count == 1
    assert mock_bot.send_sync.call_count == 2  # start + error


# ── watch decorator ───────────────────────────────

def test_watch_normal():
    bell, mock_bot = _make_bell()

    @bell.watch(total=10)
    def train():
        bell.log(10, {"acc": 0.9})
        return "done"

    result = train()
    assert result == "done"
    assert bell._state == "finished"


def test_watch_exception():
    bell, mock_bot = _make_bell()

    @bell.watch()
    def train():
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError, match="fail"):
        train()
    assert bell._state == "crashed"


def test_watch_keyboard_interrupt():
    bell, mock_bot = _make_bell()

    @bell.watch()
    def train():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        train()
    assert bell._state == "stopped"


# ── _build_status ─────────────────────────────────

def test_build_status_running_with_total():
    bell, _ = _make_bell(unit="epoch")
    bell.start(total=100)
    bell._step = 50
    status = bell._build_status()
    assert "50 / 100" in status
    assert "epoch" in status
    assert "ETA" in status

def test_build_status_running_no_total():
    bell, _ = _make_bell()
    bell.start()
    bell._step = 25
    status = bell._build_status()
    assert "25" in status
    assert "ETA" not in status

def test_build_status_idle():
    bell, _ = _make_bell()
    status = bell._build_status()
    assert "Idle" in status


# ── Non-main process no-ops ───────────────────────

def test_non_main_noop():
    with patch("torchbell.monitor.TelegramBot"):
        bell = TorchBell(run_name="test", token="t", chat_id=1, rank=1)
    bell.start()
    bell.log(1, {"x": 1})
    bell.finish()
    bell.error()
    bell.notify("hi")
    # No exceptions, all silently no-op
