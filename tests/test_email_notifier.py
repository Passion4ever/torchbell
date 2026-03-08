"""Tests for torchbell.email_notifier (EmailNotifier)."""

from unittest.mock import patch, MagicMock, call

from torchbell.email_notifier import (
    EmailNotifier,
    _html_to_plain,
    _extract_subject,
    _reformat_for_email,
    _to_email_html,
    _MAX_RETRIES,
)
from torchbell.notifier import Notifier


def _make_notifier(**kwargs):
    defaults = dict(
        smtp_host="smtp.example.com",
        smtp_port=465,
        smtp_user="user@example.com",
        smtp_pass="secret",
    )
    defaults.update(kwargs)
    return EmailNotifier(**defaults)


# ── isinstance check ──────────────────────────────

def test_is_notifier_instance():
    n = _make_notifier()
    assert isinstance(n, Notifier)


def test_repr_does_not_leak_password():
    n = _make_notifier(smtp_pass="super_secret_password")
    r = repr(n)
    assert "super_secret_password" not in r
    assert "EmailNotifier" in r
    assert "smtp.example.com" in r
    assert "****" in r


def test_supports_edit_is_false():
    n = _make_notifier()
    assert n.supports_edit is False


# ── _html_to_plain ────────────────────────────────

def test_html_to_plain_strips_tags():
    html = "<b>Hello</b> <i>world</i>"
    assert _html_to_plain(html) == "Hello world"


def test_html_to_plain_br_to_newline():
    html = "line1<br>line2<br/>line3"
    assert _html_to_plain(html) == "line1\nline2\nline3"


def test_html_to_plain_pre_code():
    html = '<pre><code class="language-python">x = 1</code></pre>'
    assert _html_to_plain(html) == "x = 1"


def test_html_to_plain_complex():
    html = (
        '<b>Training</b>\n'
        '━━━\n'
        '<pre>loss   0.1234            </pre>'
    )
    result = _html_to_plain(html)
    assert "Training" in result
    assert "loss" in result
    assert "<" not in result


# ── _extract_subject ──────────────────────────────

def test_extract_subject_plain_text():
    text = "Hello world\nMore details"
    assert _extract_subject(text) == "\U0001f514 [TorchBell] Hello world"


def test_extract_subject_strips_bell_emoji():
    text = "\U0001f514 MyRun\ndetails"
    assert _extract_subject(text) == "\U0001f514 [TorchBell] MyRun"


def test_extract_subject_strips_clipboard_emoji():
    text = "\U0001f4cb MyRun\ndetails"
    assert _extract_subject(text) == "\U0001f514 [TorchBell] MyRun"


def test_extract_subject_started_tag():
    text = "\U0001f514 MyRun\n\nMonitoring started\nmore"
    assert _extract_subject(text) == "\U0001f514 [TorchBell] MyRun - Started"


def test_extract_subject_complete_tag():
    text = "\U0001f514 MyRun\n\nTraining complete!\nmore"
    assert _extract_subject(text) == "\U0001f514 [TorchBell] MyRun - Complete"


def test_extract_subject_crashed_tag():
    text = "\U0001f514 MyRun\n\nTraining crashed!\nmore"
    assert _extract_subject(text) == "\U0001f514 [TorchBell] MyRun - Crashed"


def test_extract_subject_stopped_tag():
    text = "\U0001f514 MyRun\n\nManually stopped\nmore"
    assert _extract_subject(text) == "\U0001f514 [TorchBell] MyRun - Stopped"


def test_extract_subject_skips_empty_lines():
    text = "\n\n  Hello  \nworld"
    assert _extract_subject(text) == "\U0001f514 [TorchBell] Hello"


def test_extract_subject_truncates_long_line():
    text = "A" * 100
    subj = _extract_subject(text)
    # emoji + space + "[TorchBell] " + up to 66 chars
    assert "[TorchBell]" in subj
    # subject part (after emoji prefix) should be capped
    after_prefix = subj.split("[TorchBell] ", 1)[1]
    assert len(after_prefix) <= 66


def test_extract_subject_empty():
    assert _extract_subject("") == "\U0001f514 [TorchBell] Notification"


# ── _reformat_for_email ──────────────────────────

def test_reformat_promotes_status_line():
    text = "\U0001f514 <b>MyRun</b>\n\u2501\u2501\u2501\n\n\u2705 Training complete!\n\U0001f4c5 2026-03-08"
    result = _reformat_for_email(text)
    assert result.startswith("\u2705 Training complete!")
    assert "\u2501" * 18 in result
    assert "\U0001f4c5 2026-03-08" in result
    assert "<b>MyRun</b>" not in result


def test_reformat_started():
    text = "\U0001f514 <b>Run</b>\n\u2501\u2501\u2501\n\n\U0001f680 Monitoring started\n\U0001f4c5 now"
    result = _reformat_for_email(text)
    assert result.startswith("\U0001f680 Monitoring started")


def test_reformat_crashed():
    text = "\U0001f514 <b>Run</b>\n\u2501\u2501\u2501\n\n\U0001f525 Training crashed!\n\U0001f4c5 now"
    result = _reformat_for_email(text)
    assert result.startswith("\U0001f525 Training crashed!")


def test_reformat_stopped():
    text = "\U0001f514 <b>Run</b>\n\u2501\u2501\u2501\n\n\u23f8 Manually stopped\n\U0001f4c5 now"
    result = _reformat_for_email(text)
    assert result.startswith("\u23f8 Manually stopped")


def test_reformat_leaves_custom_notify_untouched():
    text = "\U0001f514 <b>Run</b>\n\u2501\u2501\u2501\n\nHello custom message"
    result = _reformat_for_email(text)
    assert result == text  # no status emoji → unchanged


def test_reformat_leaves_no_header_untouched():
    text = "Just some plain text"
    assert _reformat_for_email(text) == text


# ── _to_email_html ───────────────────────────────

def test_to_email_html_wraps_in_template():
    html = _to_email_html("<b>Hello</b>")
    assert "<!DOCTYPE html>" in html
    assert "Sent by TorchBell" in html
    assert "<b>Hello</b>" in html


def test_to_email_html_converts_newlines_to_br():
    html = _to_email_html("line1\nline2")
    assert "<br>" in html


def test_to_email_html_styles_pre_blocks():
    html = _to_email_html('<pre>code</pre>')
    assert "background:#f5f5f5" in html
    assert "pre-wrap" in html


def test_to_email_html_preserves_pre_content():
    html = _to_email_html('<pre>line1\nline2</pre>')
    # Newlines inside <pre> should NOT be converted to <br>
    assert "<br>" not in html.split("<pre")[1].split("</pre>")[0]


# ── smtp_to default ──────────────────────────────

def test_smtp_to_defaults_to_user():
    n = _make_notifier()
    assert n._to == "user@example.com"


def test_smtp_to_custom():
    n = _make_notifier(smtp_to="other@example.com")
    assert n._to == "other@example.com"


# ── _do_send SSL (port 465) ──────────────────────

@patch("torchbell.email_notifier.smtplib.SMTP_SSL")
def test_do_send_ssl(mock_ssl):
    server = MagicMock()
    mock_ssl.return_value = server

    n = _make_notifier(smtp_port=465)
    n._do_send("<b>Hello</b>")

    mock_ssl.assert_called_once_with("smtp.example.com", 465, timeout=15)
    server.login.assert_called_once_with("user@example.com", "secret")
    server.sendmail.assert_called_once()
    server.quit.assert_called_once()


@patch("torchbell.email_notifier.smtplib.SMTP_SSL")
def test_do_send_sends_html_email(mock_ssl):
    """Emails are sent as HTML, not plain text."""
    server = MagicMock()
    mock_ssl.return_value = server

    n = _make_notifier(smtp_port=465)
    n._do_send("<b>Hello</b>")

    raw = server.sendmail.call_args[0][2]
    assert "text/html" in raw


# ── _do_send STARTTLS (port 587) ─────────────────

@patch("torchbell.email_notifier.smtplib.SMTP")
def test_do_send_starttls(mock_smtp):
    server = MagicMock()
    mock_smtp.return_value = server

    n = _make_notifier(smtp_port=587)
    n._do_send("<b>Hello</b>")

    mock_smtp.assert_called_once_with("smtp.example.com", 587, timeout=15)
    server.starttls.assert_called_once()
    server.login.assert_called_once_with("user@example.com", "secret")
    server.sendmail.assert_called_once()
    server.quit.assert_called_once()


# ── send_sync retries ────────────────────────────

@patch("torchbell.email_notifier.time.sleep")
@patch("torchbell.email_notifier.smtplib.SMTP_SSL")
def test_send_sync_success(mock_ssl, mock_sleep):
    mock_ssl.return_value = MagicMock()
    n = _make_notifier()
    result = n.send_sync("hello")
    assert result is None  # email has no message_id
    mock_sleep.assert_not_called()


@patch("torchbell.email_notifier.time.sleep")
@patch("torchbell.email_notifier.smtplib.SMTP_SSL")
def test_send_sync_retry_then_success(mock_ssl, mock_sleep):
    server_fail = MagicMock()
    server_fail.login.side_effect = Exception("auth error")
    server_ok = MagicMock()
    mock_ssl.side_effect = [server_fail, server_ok]

    n = _make_notifier()
    n.send_sync("hello")
    assert mock_sleep.call_count == 1


@patch("torchbell.email_notifier.time.sleep")
@patch("torchbell.email_notifier.smtplib.SMTP_SSL")
def test_send_sync_all_fail(mock_ssl, mock_sleep, capsys):
    server = MagicMock()
    server.login.side_effect = Exception("auth error")
    mock_ssl.return_value = server

    n = _make_notifier()
    n.send_sync("hello")
    assert mock_sleep.call_count == _MAX_RETRIES
    captured = capsys.readouterr()
    assert "email send failed" in captured.out
