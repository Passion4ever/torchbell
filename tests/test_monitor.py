"""Tests for torchbell.monitor (TorchBell)."""

import os
import signal
import threading
from unittest.mock import patch, MagicMock

import pytest

from torchbell.monitor import TorchBell
from torchbell.notifier import Notifier


def _make_bell(**kwargs):
    """Create a TorchBell with a mocked bot."""
    defaults = dict(run_name="test-run", token="tok", chat_id=123)
    defaults.update(kwargs)
    with patch("torchbell.monitor.TelegramBot") as MockBot:
        mock_bot = MagicMock()
        mock_bot.send_sync.return_value = 1  # message_id
        mock_bot.supports_edit = True
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

def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv("TG_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TG_CHAT_ID", raising=False)
    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.delenv("SMTP_PORT", raising=False)
    monkeypatch.delenv("SMTP_USER", raising=False)
    monkeypatch.delenv("SMTP_PASS", raising=False)
    with pytest.raises(ValueError, match="No notifier configured"):
        TorchBell(run_name="test")

def test_env_var_loading(monkeypatch):
    monkeypatch.setenv("TG_BOT_TOKEN", "env-tok")
    monkeypatch.setenv("TG_CHAT_ID", "999")
    with patch("torchbell.monitor.TelegramBot"):
        bell = TorchBell(run_name="test")
    assert bell._is_main is True

def test_non_main_process_no_notifiers():
    with patch("torchbell.monitor.TelegramBot"):
        bell = TorchBell(run_name="test", token="t", chat_id=1, rank=1)
    assert bell._notifiers == []


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


# ── _on_stop ─────────────────────────────────────

def test_on_stop_sets_stopped_state():
    bell, mock_bot = _make_bell()
    bell.start()
    bell._on_stop()
    assert bell._state == "stopped"


def test_on_stop_sets_stop_refresh_event():
    bell, mock_bot = _make_bell()
    bell.start()
    bell._on_stop()
    assert bell._stop_refresh.is_set()


def test_on_stop_sends_notification():
    bell, mock_bot = _make_bell()
    bell.start()
    bell._on_stop()
    # edit_sync for status update + send_sync for stop notification
    assert mock_bot.edit_sync.call_count == 1
    assert mock_bot.send_sync.call_count == 2  # start + stop
    stop_msg = mock_bot.send_sync.call_args[0][0]
    assert "Manually stopped" in stop_msg


def test_on_stop_skips_edit_without_status_msg_id():
    bell, mock_bot = _make_bell()
    bell.start()
    bell._status_msg_ids = {}
    bell._on_stop()
    assert bell._state == "stopped"
    assert mock_bot.edit_sync.call_count == 0
    assert mock_bot.send_sync.call_count == 2  # start + stop


def test_on_stop_noop_when_not_running():
    bell, mock_bot = _make_bell()
    # state is "idle", _on_stop should early return
    bell._on_stop()
    assert bell._state == "idle"
    mock_bot.edit_sync.assert_not_called()


def test_on_stop_not_called_twice():
    bell, mock_bot = _make_bell()
    bell.start()
    bell._on_stop()
    assert bell._state == "stopped"
    # Second call should early return (state != "running")
    bell._on_stop()
    assert mock_bot.edit_sync.call_count == 1
    assert mock_bot.send_sync.call_count == 2  # start + first stop only


def test_on_stop_bot_exception_still_sets_stopped(capsys):
    bell, mock_bot = _make_bell()
    bell.start()
    mock_bot.edit_sync.side_effect = RuntimeError("network error")
    bell._on_stop()
    assert bell._state == "stopped"
    captured = capsys.readouterr()
    assert "stop notification failed" in captured.out


# ── _on_exit ──────────────────────────────────────

def test_on_exit_calls_on_stop_when_running():
    bell, mock_bot = _make_bell()
    bell.start()
    bell._on_exit()
    assert bell._state == "stopped"
    assert mock_bot.send_sync.call_count == 2  # start + stop


def test_on_exit_noop_when_idle():
    bell, mock_bot = _make_bell()
    bell._on_exit()
    assert bell._state == "idle"
    mock_bot.send_sync.assert_not_called()


def test_on_exit_noop_when_finished():
    bell, mock_bot = _make_bell()
    bell.start()
    bell.finish()
    initial_send_count = mock_bot.send_sync.call_count
    bell._on_exit()
    assert bell._state == "finished"
    assert mock_bot.send_sync.call_count == initial_send_count


# ── _refresh_loop ─────────────────────────────────

def test_refresh_loop_calls_edit(capsys):
    bell, mock_bot = _make_bell(refresh_interval=0.05)
    bell.start()
    # Let the refresh thread run a few cycles
    import time
    time.sleep(0.2)
    bell._stop_refresh.set()
    bell._refresh_thread.join(timeout=2)
    # edit (async) should have been called at least once via _refresh_loop
    assert mock_bot.edit.call_count >= 1


def test_refresh_loop_stops_when_event_set():
    bell, mock_bot = _make_bell(refresh_interval=0.05)
    bell.start()
    bell._stop_refresh.set()
    bell._refresh_thread.join(timeout=2)
    assert not bell._refresh_thread.is_alive()


def test_refresh_loop_handles_exception(capsys):
    bell, mock_bot = _make_bell(refresh_interval=0.05)
    mock_bot.edit.side_effect = RuntimeError("api down")
    bell.start()
    import time
    time.sleep(0.15)
    bell._stop_refresh.set()
    bell._refresh_thread.join(timeout=2)
    captured = capsys.readouterr()
    assert "refresh failed" in captured.out


# ── _handle_signal ────────────────────────────────

def test_handle_signal_calls_on_stop_and_raises():
    bell, mock_bot = _make_bell()
    bell.start()
    with pytest.raises(SystemExit) as exc_info:
        bell._handle_signal(signal.SIGTERM, None)
    assert exc_info.value.code == 1
    assert bell._state == "stopped"


def test_handle_signal_raises_even_when_not_running():
    bell, mock_bot = _make_bell()
    # state is "idle", _on_stop will be a no-op, but SystemExit still raised
    with pytest.raises(SystemExit) as exc_info:
        bell._handle_signal(signal.SIGTERM, None)
    assert exc_info.value.code == 1


# ── notifier= parameter (single) ─────────────────

def test_notifier_parameter():
    mock_notifier = MagicMock()
    mock_notifier.send_sync.return_value = None
    mock_notifier.supports_edit = False
    bell = TorchBell(run_name="test", notifier=mock_notifier)
    assert bell._notifiers == [mock_notifier]


def test_notifier_no_edit_on_finish():
    """When notifier.supports_edit=False, finish should skip edit_sync."""
    mock_notifier = MagicMock()
    mock_notifier.send_sync.return_value = None  # no message_id
    mock_notifier.supports_edit = False
    bell = TorchBell(run_name="test", notifier=mock_notifier)
    bell.start()
    # send_sync returned None, so _status_msg_ids is empty
    assert bell._status_msg_ids == {}
    bell.finish()
    mock_notifier.edit_sync.assert_not_called()


def test_notifier_refresh_skips_non_edit():
    """When supports_edit=False, _refresh_loop skips the notifier entirely."""
    mock_notifier = MagicMock()
    mock_notifier.send_sync.return_value = None
    mock_notifier.supports_edit = False
    bell = TorchBell(run_name="test", notifier=mock_notifier,
                     refresh_interval=0.05)
    bell.start()
    import time
    time.sleep(0.2)
    bell._stop_refresh.set()
    bell._refresh_thread.join(timeout=2)
    # Non-edit notifier should be completely skipped during refresh
    mock_notifier.edit.assert_not_called()
    mock_notifier.send.assert_not_called()


def test_smtp_env_fallback(monkeypatch):
    """When only SMTP env vars are set, EmailNotifier is created automatically."""
    monkeypatch.delenv("TG_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TG_CHAT_ID", raising=False)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "465")
    monkeypatch.setenv("SMTP_USER", "user@example.com")
    monkeypatch.setenv("SMTP_PASS", "secret")
    bell = TorchBell(run_name="test")
    from torchbell.email_notifier import EmailNotifier
    assert len(bell._notifiers) == 1
    assert isinstance(bell._notifiers[0], EmailNotifier)


# ── notifier= parameter (list / multi-notifier) ──

def test_notifier_list():
    """Passing a list of notifiers stores them all."""
    n1 = MagicMock(spec=Notifier)
    n1.send_sync.return_value = 10
    n1.supports_edit = True
    n2 = MagicMock(spec=Notifier)
    n2.send_sync.return_value = None
    n2.supports_edit = False
    bell = TorchBell(run_name="test", notifier=[n1, n2])
    assert bell._notifiers == [n1, n2]


def test_notifier_empty_list_raises():
    """Passing an empty list raises ValueError."""
    with pytest.raises(ValueError, match="must not be empty"):
        TorchBell(run_name="test", notifier=[])


def test_notifier_list_bad_type_raises():
    """Passing non-Notifier in list raises TypeError."""
    with pytest.raises(TypeError, match="Notifier instance"):
        TorchBell(run_name="test", notifier=["not-a-notifier"])


def test_multi_notifier_start_collects_msg_ids():
    """start() collects message_ids from all notifiers that return one."""
    n1 = MagicMock(spec=Notifier)
    n1.send_sync.return_value = 42
    n1.supports_edit = True
    n2 = MagicMock(spec=Notifier)
    n2.send_sync.return_value = None
    n2.supports_edit = False
    bell = TorchBell(run_name="test", notifier=[n1, n2])
    bell.start()
    assert bell._status_msg_ids == {0: 42}
    # Both should have received send_sync
    n1.send_sync.assert_called_once()
    n2.send_sync.assert_called_once()


def test_multi_notifier_finish_sends_to_all():
    """finish() edits status for notifiers with msg_id, sends to all."""
    n1 = MagicMock(spec=Notifier)
    n1.send_sync.return_value = 42
    n1.supports_edit = True
    n2 = MagicMock(spec=Notifier)
    n2.send_sync.return_value = None
    n2.supports_edit = False
    bell = TorchBell(run_name="test", notifier=[n1, n2])
    bell.start()
    bell.finish()
    # n1 has msg_id → edit_sync called
    assert n1.edit_sync.call_count == 1
    # n2 has no msg_id → edit_sync NOT called
    n2.edit_sync.assert_not_called()
    # Both receive the final notification via send_sync (start + finish = 2 each)
    assert n1.send_sync.call_count == 2
    assert n2.send_sync.call_count == 2


def test_multi_notifier_error_sends_to_all():
    """error() edits status for notifiers with msg_id, sends to all."""
    n1 = MagicMock(spec=Notifier)
    n1.send_sync.return_value = 42
    n1.supports_edit = True
    n2 = MagicMock(spec=Notifier)
    n2.send_sync.return_value = None
    n2.supports_edit = False
    bell = TorchBell(run_name="test", notifier=[n1, n2])
    bell.start()
    exc = ValueError("boom")
    try:
        raise exc
    except ValueError:
        bell.error(exc)
    assert n1.edit_sync.call_count == 1
    n2.edit_sync.assert_not_called()
    assert n1.send_sync.call_count == 2
    assert n2.send_sync.call_count == 2


def test_multi_notifier_notify_sends_to_all():
    """notify() sends to all notifiers."""
    n1 = MagicMock(spec=Notifier)
    n1.send_sync.return_value = 10
    n1.supports_edit = True
    n2 = MagicMock(spec=Notifier)
    n2.send_sync.return_value = None
    n2.supports_edit = False
    bell = TorchBell(run_name="test", notifier=[n1, n2])
    bell.notify("hello")
    n1.send.assert_called_once()
    n2.send.assert_called_once()


def test_multi_notifier_refresh_edits_only():
    """_refresh_loop edits for supports_edit=True, skips supports_edit=False."""
    n1 = MagicMock(spec=Notifier)
    n1.send_sync.return_value = 42
    n1.supports_edit = True
    n2 = MagicMock(spec=Notifier)
    n2.send_sync.return_value = None
    n2.supports_edit = False
    bell = TorchBell(run_name="test", notifier=[n1, n2], refresh_interval=0.05)
    bell.start()
    import time
    time.sleep(0.2)
    bell._stop_refresh.set()
    bell._refresh_thread.join(timeout=2)
    # n1 supports edit and has msg_id → edit called
    assert n1.edit.call_count >= 1
    n1.send.assert_not_called()
    # n2 does not support edit → completely skipped
    n2.edit.assert_not_called()
    n2.send.assert_not_called()


def test_env_both_tg_and_smtp(monkeypatch):
    """When both TG and SMTP env vars are set, both notifiers are created."""
    monkeypatch.setenv("TG_BOT_TOKEN", "tok")
    monkeypatch.setenv("TG_CHAT_ID", "123")
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "465")
    monkeypatch.setenv("SMTP_USER", "user@example.com")
    monkeypatch.setenv("SMTP_PASS", "secret")
    with patch("torchbell.monitor.TelegramBot") as MockBot:
        mock_tg = MagicMock()
        MockBot.return_value = mock_tg
        bell = TorchBell(run_name="test")
    from torchbell.email_notifier import EmailNotifier
    assert len(bell._notifiers) == 2
    assert bell._notifiers[0] is mock_tg
    assert isinstance(bell._notifiers[1], EmailNotifier)


def test_start_sends_start_notification_to_non_edit_notifier():
    """supports_edit=False notifier receives concise start notification."""
    mock_notifier = MagicMock(spec=Notifier)
    mock_notifier.send_sync.return_value = None
    mock_notifier.supports_edit = False
    bell = TorchBell(run_name="test", notifier=mock_notifier)
    bell.start(total=100)
    start_msg = mock_notifier.send_sync.call_args[0][0]
    assert "Monitoring started" in start_msg
    assert "100" in start_msg
    # Should NOT contain Progress/Elapsed (that's the status message)
    assert "Progress" not in start_msg
    assert "Elapsed" not in start_msg


def test_start_sends_status_to_edit_notifier():
    """supports_edit=True notifier receives full status message."""
    bell, mock_bot = _make_bell()
    bell.start(total=100)
    start_msg = mock_bot.send_sync.call_args[0][0]
    assert "Progress" in start_msg
    assert "Monitoring started" not in start_msg


def test_start_notification_without_total():
    """Start notification omits total line when total is None."""
    mock_notifier = MagicMock(spec=Notifier)
    mock_notifier.send_sync.return_value = None
    mock_notifier.supports_edit = False
    bell = TorchBell(run_name="test", notifier=mock_notifier)
    bell.start()
    start_msg = mock_notifier.send_sync.call_args[0][0]
    assert "Monitoring started" in start_msg
    assert "\U0001f522" not in start_msg  # 🔢 not present without total


def test_multi_notifier_start_dispatches_correctly():
    """In multi-notifier setup, each notifier receives appropriate start msg."""
    n1 = MagicMock(spec=Notifier)
    n1.send_sync.return_value = 42
    n1.supports_edit = True
    n2 = MagicMock(spec=Notifier)
    n2.send_sync.return_value = None
    n2.supports_edit = False
    bell = TorchBell(run_name="test", notifier=[n1, n2])
    bell.start(total=50)
    # n1 (edit-capable) gets status with Progress
    msg1 = n1.send_sync.call_args[0][0]
    assert "Progress" in msg1
    # n2 (non-edit) gets concise start notification
    msg2 = n2.send_sync.call_args[0][0]
    assert "Monitoring started" in msg2
    assert "50" in msg2


def test_multi_notifier_on_stop():
    """_on_stop sends to all notifiers."""
    n1 = MagicMock(spec=Notifier)
    n1.send_sync.return_value = 42
    n1.supports_edit = True
    n2 = MagicMock(spec=Notifier)
    n2.send_sync.return_value = None
    n2.supports_edit = False
    bell = TorchBell(run_name="test", notifier=[n1, n2])
    bell.start()
    bell._on_stop()
    assert bell._state == "stopped"
    assert n1.edit_sync.call_count == 1
    n2.edit_sync.assert_not_called()
    assert n1.send_sync.call_count == 2  # start + stop
    assert n2.send_sync.call_count == 2
