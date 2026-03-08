"""Email notification backend using smtplib (stdlib, zero extra dependencies)."""

import queue
import re
import smtplib
import threading
import time
from email.mime.text import MIMEText
from typing import Optional, Union

from .notifier import Notifier

_SEND_TIMEOUT = 15
_MAX_RETRIES = 2


def _html_to_plain(html_text: str) -> str:
    """Convert simple HTML (as used by TorchBell messages) to plain text."""
    text = re.sub(r"<br\s*/?>", "\n", html_text)
    text = re.sub(r"</?(?:b|strong)>", "", text)
    text = re.sub(r"</?(?:i|em)>", "", text)
    text = re.sub(r"<pre[^>]*>", "", text)
    text = re.sub(r"</pre>", "", text)
    text = re.sub(r'<code[^>]*>', "", text)
    text = re.sub(r"</code>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


_SUBJECT_TAGS = {
    "Monitoring started": "Started",
    "Training complete":  "Complete",
    "Training crashed":   "Crashed",
    "Manually stopped":   "Stopped",
}


def _extract_subject(text: str) -> str:
    """Extract a descriptive subject line from plain text content.

    Detects the event type and appends a tag so the user can tell
    emails apart at a glance in the inbox.  The 🔔 prefix is a
    consistent brand marker; status emojis live in the email body.
    """
    # Get run name from first non-empty line
    run_name = ""
    for line in text.splitlines():
        line = line.strip()
        if line:
            run_name = line
            break

    if not run_name:
        return "\U0001f514 [TorchBell] Notification"

    # Strip leading emoji prefix (🔔, 📋, 🚀)
    for prefix in ["\U0001f514 ", "\U0001f4cb ", "\U0001f680 "]:
        if run_name.startswith(prefix):
            run_name = run_name[len(prefix):]
            break

    # Detect event type via keyword mapping
    tag = None
    for keyword, t in _SUBJECT_TAGS.items():
        if keyword in text:
            tag = t
            break

    if tag:
        subject = "{} - {}".format(run_name, tag)
    else:
        subject = run_name

    if len(subject) > 66:
        subject = subject[:63] + "..."

    return "\U0001f514 [TorchBell] {}".format(subject)


# Matches the TG-style header: 🔔/📋 <b>Name</b>\n━━━\n\n
_EMAIL_HEADER_RE = re.compile(
    r'^[\U0001f514\U0001f4cb] <b>[^<]*</b>\n'
    r'\u2501+\n'
    r'\n'
)
_STATUS_EMOJIS = ('\U0001f680', '\u2705', '\U0001f525', '\u23f8')
_SEP = '\u2501' * 18


def _reformat_for_email(text: str) -> str:
    """Remove the 🔔 Name header and promote the status line to the top.

    Email subjects already carry the project name, so the body starts
    directly with the status emoji line + separator + details.
    Only reformats when the body has a recognisable status emoji line;
    custom ``notify()`` messages are left untouched.
    """
    m = _EMAIL_HEADER_RE.match(text)
    if not m:
        return text
    rest = text[m.end():]
    if not any(rest.startswith(e) for e in _STATUS_EMOJIS):
        return text
    first_nl = rest.find('\n')
    if first_nl == -1:
        return rest
    status_line = rest[:first_nl]
    remaining = rest[first_nl + 1:]
    return status_line + '\n' + _SEP + '\n\n' + remaining


def _to_email_html(text: str) -> str:
    """Convert Telegram-flavored HTML to a styled email HTML document."""
    body = text

    # Style <pre> blocks
    body = re.sub(
        r'<pre[^>]*>',
        '<pre style="background:#f5f5f5; padding:12px; border-radius:6px; '
        'font-size:13px; line-height:1.4; white-space:pre-wrap; '
        'font-family:Menlo,Monaco,Consolas,monospace;">',
        body,
    )
    body = re.sub(
        r'<code[^>]*>',
        '<code style="font-family:Menlo,Monaco,Consolas,monospace; '
        'font-size:13px;">',
        body,
    )

    # Convert \n to <br> outside of <pre> blocks
    parts = re.split(r'(<pre[^>]*>.*?</pre>)', body, flags=re.DOTALL)
    converted = []
    for part in parts:
        if part.startswith('<pre'):
            converted.append(part)
        else:
            converted.append(part.replace('\n', '<br>\n'))
    body = ''.join(converted)

    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8"></head>'
        '<body style="margin:0; padding:20px; background:#f4f4f4;">'
        '<div style="max-width:480px; margin:0 auto; background:#ffffff; '
        'border-radius:8px; padding:24px; '
        'font-family:-apple-system,Arial,sans-serif; '
        'font-size:15px; line-height:1.6; color:#333; '
        'box-shadow:0 1px 3px rgba(0,0,0,0.1);">'
        + body +
        '<hr style="border:none; border-top:1px solid #eee; '
        'margin:20px 0 12px;">'
        '<div style="font-size:12px; color:#999;">Sent by TorchBell</div>'
        '</div></body></html>'
    )


class EmailNotifier(Notifier):
    """Send notifications via SMTP email.

    Parameters
    ----------
    smtp_host : SMTP server hostname
    smtp_port : SMTP server port (465 = SSL, others use STARTTLS)
    smtp_user : SMTP username (also used as sender address)
    smtp_pass : SMTP password / app password
    smtp_to   : recipient address (default: same as smtp_user)
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        smtp_user: str,
        smtp_pass: str,
        smtp_to: Optional[str] = None,
    ):
        self._host = smtp_host
        self._port = smtp_port
        self._user = smtp_user
        self._pass = smtp_pass
        self._to = smtp_to or smtp_user
        self._queue: queue.Queue = queue.Queue()
        self._sender_thread: Optional[threading.Thread] = None

    def __repr__(self) -> str:
        from .utils import mask_credential
        return "EmailNotifier(host={}, port={}, user={}, pass={})".format(
            self._host, self._port, self._user, mask_credential(self._pass)
        )

    def send(self, text: str, block: bool = False) -> Optional[int]:
        """Queue an email. Returns None (email has no message_id concept)."""
        done_event = threading.Event() if block else None
        self._queue.put((text, done_event))
        self._ensure_sender()
        if done_event:
            done_event.wait(timeout=_SEND_TIMEOUT)
        return None

    def send_sync(self, text: str) -> Optional[int]:
        """Synchronous send with retries. Returns None."""
        for attempt in range(_MAX_RETRIES + 1):
            try:
                self._do_send(text)
                return None
            except BaseException as e:
                if attempt < _MAX_RETRIES:
                    time.sleep(1)
                else:
                    print("[TorchBell] email send failed "
                          "({} retries): {}".format(_MAX_RETRIES, e))
        return None

    def _ensure_sender(self):
        if self._sender_thread and self._sender_thread.is_alive():
            return
        self._sender_thread = threading.Thread(
            target=self._send_loop, daemon=True, name="TorchBell-Email"
        )
        self._sender_thread.start()

    def _send_loop(self):
        while True:
            try:
                text, done_event = self._queue.get(timeout=5)
            except queue.Empty:
                return
            try:
                self._do_send(text)
            except Exception as e:
                print("[TorchBell] email send failed: {}".format(e))
            finally:
                if done_event:
                    done_event.set()

    def _do_send(self, text: str) -> None:
        plain = _html_to_plain(text)
        subject = _extract_subject(plain)
        body = _reformat_for_email(text)
        html = _to_email_html(body)

        msg = MIMEText(html, "html", "utf-8")
        msg["Subject"] = subject
        msg["From"] = self._user
        msg["To"] = self._to

        server: Union[smtplib.SMTP_SSL, smtplib.SMTP]
        if self._port == 465:
            server = smtplib.SMTP_SSL(self._host, self._port,
                                      timeout=_SEND_TIMEOUT)
        else:
            server = smtplib.SMTP(self._host, self._port,
                                  timeout=_SEND_TIMEOUT)
            server.starttls()

        try:
            server.login(self._user, self._pass)
            server.sendmail(self._user, [self._to], msg.as_string())
        finally:
            server.quit()
