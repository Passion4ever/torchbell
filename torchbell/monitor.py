"""Core monitor class: TorchBell.

- Self-updating status message throughout training lifecycle
- Notification on finish / crash / manual stop (triggers phone alert)
- Auto-detects multi-GPU (Accelerate / DDP / DeepSpeed / SLURM)
- Multiple projects & machines can share one bot
"""

import atexit
import html
import os
import signal
import threading
import time
import traceback
from datetime import datetime
from functools import wraps
from typing import Dict, List, Optional, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .bot import TelegramBot
from .notifier import Notifier
from .utils import fmt_time, fmt_metrics

_SEP = "━" * 21


class TorchBell:
    """
    Parameters
    ----------
    run_name         : experiment name shown in all messages
    token            : Bot Token, defaults to env TG_BOT_TOKEN
    chat_id          : Chat ID, defaults to env TG_CHAT_ID
    notifier         : single Notifier, list of Notifiers, or None
    accelerator      : Accelerate instance (optional)
    rank             : process rank (optional), 0 = main process
    unit             : display unit for progress (e.g. "epoch", "step"), default none
    silent           : mute notifications (no vibration/ring)
    refresh_interval : status message refresh interval in seconds (default 30)

    Notifier resolution:
      1. explicit notifier parameter (single or list)
      2. auto-detect from params / env vars — all configured channels activate
      3. raise ValueError if nothing configured

    Multi-GPU dedup priority:
      1. explicit rank parameter
      2. accelerator.is_main_process
      3. env vars RANK / LOCAL_RANK / SLURM_PROCID
      4. none found -> single GPU
    """

    def __init__(
        self,
        run_name: str = "Training",
        token: Optional[str] = None,
        chat_id: Optional[int] = None,
        notifier: Union[Notifier, List[Notifier], None] = None,
        accelerator=None,
        rank: Optional[int] = None,
        unit: Optional[str] = None,
        silent: bool = False,
        refresh_interval: int = 30,
    ):
        self._is_main = self._detect_is_main(accelerator, rank)
        self.run_name = run_name
        self._safe_name = html.escape(run_name)
        self._unit = f" {unit}" if unit else ""

        if not self._is_main:
            self._notifiers: List[Notifier] = []
            return

        self._notifiers = self._resolve_notifiers(notifier, token, chat_id, silent)
        self._refresh_interval = refresh_interval

        self._start_time: Optional[float] = None
        self._step: int = 0
        self._total: Optional[int] = None
        self._latest_metrics: Dict[str, float] = {}

        self._status_msg_ids: Dict[int, int] = {}
        self._state: str = "idle"  # idle / running / finished / crashed / stopped
        self._refresh_thread: Optional[threading.Thread] = None
        self._stop_refresh = threading.Event()

        atexit.register(self._on_exit)
        try:
            signal.signal(signal.SIGTERM, self._handle_signal)
        except (OSError, ValueError):
            pass

    # ── Public API ──────────────────────────────────

    def start(self, total: Optional[int] = None):
        """Begin monitoring. total: total steps/epochs (optional)."""
        if not self._is_main:
            return
        if self._state == "running":
            self._stop_refresh.set()
            if self._refresh_thread:
                self._refresh_thread.join(timeout=2)

        self._start_time = time.time()
        self._state = "running"
        self._total = total
        self._step = 0
        self._latest_metrics = {}

        status = self._build_status()
        start_notif = self._build_start_notification()
        self._status_msg_ids = {}
        for i, n in enumerate(self._notifiers):
            msg = status if n.supports_edit else start_notif
            mid = n.send_sync(msg)
            if mid is not None:
                self._status_msg_ids[i] = mid

        self._stop_refresh.clear()
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop, daemon=True, name="TorchBell-Refresh"
        )
        self._refresh_thread.start()
        print(f"[TorchBell] Started · {self.run_name}")

    def log(self, step: int, metrics: Dict[str, float]):
        """Report step and metrics."""
        if not self._is_main:
            return
        self._step = step
        self._latest_metrics = metrics

    def finish(self, final_metrics: Optional[Dict[str, float]] = None):
        """Training completed normally."""
        if not self._is_main:
            return
        self._state = "finished"
        self._stop_refresh.set()

        if final_metrics:
            self._latest_metrics = final_metrics

        status = self._build_status()
        for i, n in enumerate(self._notifiers):
            if i in self._status_msg_ids:
                n.edit_sync(self._status_msg_ids[i], status)

        elapsed = time.time() - self._start_time if self._start_time else 0
        metrics_str = f"\n\n{fmt_metrics(self._latest_metrics)}" if self._latest_metrics else ""

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_str = f"\n🔢 {self._total}{self._unit}" if self._total else ""
        text = (
            f"🔔 <b>{self._safe_name}</b>\n"
            f"{_SEP}\n"
            f"\n"
            f"✅ Training complete!\n"
            f"📅 {now}\n"
            f"⏱ {fmt_time(elapsed)}"
            f"{total_str}"
            f"{metrics_str}"
        )
        for n in self._notifiers:
            n.send_sync(text)

    def error(self, exception: Optional[Exception] = None):
        """Training crashed."""
        if not self._is_main:
            return
        self._state = "crashed"
        self._stop_refresh.set()

        if exception and exception.__traceback__:
            tb = "".join(traceback.format_exception(
                type(exception), exception, exception.__traceback__
            ))[-1200:]
        else:
            tb = traceback.format_exc()[-1200:]
        tb = html.escape(tb)
        err_type = type(exception).__name__ if exception else "Exception"
        err_msg = html.escape(str(exception)) if exception else "Unknown"

        status = self._build_status()
        for i, n in enumerate(self._notifiers):
            if i in self._status_msg_ids:
                n.edit_sync(self._status_msg_ids[i], status)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = (
            f"🔔 <b>{self._safe_name}</b>\n"
            f"{_SEP}\n"
            f"\n"
            f"🔥 Training crashed!\n"
            f"📅 {now}\n"
            f"❌ <b>{err_type}</b>: {err_msg}\n\n"
            f'<pre><code class="language-python">{tb}</code></pre>'
        )
        for n in self._notifiers:
            n.send_sync(text)

    def notify(self, message: str):
        """Send a custom notification."""
        if not self._is_main:
            return
        text = f"🔔 <b>{self._safe_name}</b>\n{_SEP}\n\n{message}"
        for n in self._notifiers:
            n.send(text)

    # ── Decorator ───────────────────────────────────

    def watch(self, total: Optional[int] = None):
        """
        Decorator: auto start / finish / error.

            @bell.watch(total=1000)
            def train():
                for step in range(1000):
                    bell.log(step, {"loss": loss})
        """
        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                self.start(total=total)
                try:
                    result = fn(*args, **kwargs)
                    self.finish()
                    return result
                except KeyboardInterrupt:
                    self._on_stop()
                    raise
                except Exception as e:
                    self.error(e)
                    raise
            return wrapper
        return decorator

    # ── Internal ────────────────────────────────────

    @staticmethod
    def _resolve_notifiers(
        notifier: Union[Notifier, List[Notifier], None],
        token: Optional[str],
        chat_id: Optional[int],
        silent: bool,
    ) -> List[Notifier]:
        # 1. Explicit notifier(s)
        if notifier is not None:
            if isinstance(notifier, list):
                if not notifier:
                    raise ValueError("notifier list must not be empty.")
                for n in notifier:
                    if not isinstance(n, Notifier):
                        raise TypeError(
                            "Every item in notifier list must be a "
                            "Notifier instance, got {}.".format(type(n).__name__)
                        )
                return list(notifier)
            return [notifier]

        # 2. Auto-detect all available channels
        result = []  # type: List[Notifier]

        _token = token if token is not None else os.environ.get("TG_BOT_TOKEN")
        _chat_id_raw = chat_id if chat_id is not None else os.environ.get("TG_CHAT_ID")
        if _token and _chat_id_raw:
            result.append(TelegramBot(_token, int(_chat_id_raw), silent))

        smtp_host = os.environ.get("SMTP_HOST")
        smtp_port = os.environ.get("SMTP_PORT")
        smtp_user = os.environ.get("SMTP_USER")
        smtp_pass = os.environ.get("SMTP_PASS")
        if smtp_host and smtp_port and smtp_user and smtp_pass:
            from .email_notifier import EmailNotifier
            result.append(EmailNotifier(
                smtp_host, int(smtp_port), smtp_user, smtp_pass,
                smtp_to=os.environ.get("SMTP_TO"),
            ))

        if not result:
            raise ValueError(
                "No notifier configured. "
                "Pass notifier=, token/chat_id, "
                "or set TG_BOT_TOKEN/TG_CHAT_ID or SMTP_* env vars."
            )

        return result

    @staticmethod
    def _detect_is_main(accelerator, rank) -> bool:
        if rank is not None:
            return bool(rank == 0)
        if accelerator is not None:
            return bool(getattr(accelerator, "is_main_process", True))
        for var in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
            val = os.environ.get(var)
            if val is not None:
                return int(val) == 0
        return True

    def _handle_signal(self, signum, frame):
        self._on_stop()
        raise SystemExit(1)

    def _on_stop(self):
        if self._state != "running":
            return
        self._state = "stopped"
        self._stop_refresh.set()

        # Block SIGINT during cleanup to prevent interrupted HTTP requests
        is_main_thread = threading.current_thread() is threading.main_thread()
        if is_main_thread:
            prev_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            status = self._build_status()
            for i, n in enumerate(self._notifiers):
                if i in self._status_msg_ids:
                    n.edit_sync(self._status_msg_ids[i], status)

            elapsed = time.time() - self._start_time if self._start_time else 0
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            text = (
                f"🔔 <b>{self._safe_name}</b>\n"
                f"{_SEP}\n"
                f"\n"
                f"⏸ Manually stopped\n"
                f"📅 {now}\n"
                f"⏱ {fmt_time(elapsed)}\n"
                f"🔢 {self._step}"
                + (f" / {self._total}" if self._total else "")
                + self._unit
            )
            for n in self._notifiers:
                n.send_sync(text)
        except BaseException as e:
            print(f"[TorchBell] stop notification failed: {e}")
        finally:
            if is_main_thread:
                signal.signal(signal.SIGINT, prev_handler)

    def _on_exit(self):
        if self._state == "running":
            self._on_stop()

    def _build_start_notification(self) -> str:
        """Build a concise start notification for non-edit notifiers (Email)."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_str = f"\n\U0001f522 {self._total}{self._unit}" if self._total else ""
        return (
            f"\U0001f514 <b>{self._safe_name}</b>\n"
            f"{_SEP}\n"
            f"\n"
            f"\U0001f680 Monitoring started\n"
            f"\U0001f4c5 {now}"
            f"{total_str}"
        )

    def _build_status(self) -> str:
        elapsed = time.time() - self._start_time if self._start_time else 0
        now = datetime.now().strftime("%H:%M:%S")

        lines = [f"📋 <b>{self._safe_name}</b>", _SEP, ""]

        u = self._unit

        if self._state == "running":
            if self._total:
                lines.append(f"▸ Progress    {self._step} / {self._total}{u}")
            else:
                lines.append(f"▸ Progress    {self._step}{u}")
            lines.append(f"▸ Elapsed     {fmt_time(elapsed)}")

            if self._total and self._step > 0:
                speed = elapsed / self._step
                remaining = speed * (self._total - self._step)
                lines.append(f"▸ ETA         {fmt_time(remaining)}")

            if self._latest_metrics:
                lines.append("")
                lines.append(fmt_metrics(self._latest_metrics))
            lines.append(f"\n🔄 {now}")

        elif self._state == "finished":
            lines.append("✅ Complete")
            if self._total:
                lines.append(f"▸ Progress    {self._step} / {self._total}{u}")
            else:
                lines.append(f"▸ Progress    {self._step}{u}")
            lines.append(f"▸ Total       {fmt_time(elapsed)}")

            if self._latest_metrics:
                lines.append("")
                lines.append(fmt_metrics(self._latest_metrics))

        elif self._state == "crashed":
            lines.append("🔥 Crashed")
            if self._total:
                lines.append(f"▸ At          {self._step} / {self._total}{u}")
            else:
                lines.append(f"▸ At          {self._step}{u}")
            lines.append(f"▸ Elapsed     {fmt_time(elapsed)}")

            if self._latest_metrics:
                lines.append("")
                lines.append(fmt_metrics(self._latest_metrics))

        elif self._state == "stopped":
            lines.append("⏸ Stopped")
            if self._total:
                lines.append(f"▸ At          {self._step} / {self._total}{u}")
            else:
                lines.append(f"▸ At          {self._step}{u}")
            lines.append(f"▸ Elapsed     {fmt_time(elapsed)}")

            if self._latest_metrics:
                lines.append("")
                lines.append(fmt_metrics(self._latest_metrics))

        else:
            return f"📋 <b>{self._safe_name}</b>\n{_SEP}\n\n💤 Idle"

        return "\n".join(lines)

    def _refresh_loop(self):
        while not self._stop_refresh.wait(timeout=self._refresh_interval):
            try:
                status = self._build_status()
                for i, n in enumerate(self._notifiers):
                    if n.supports_edit and i in self._status_msg_ids:
                        n.edit(self._status_msg_ids[i], status)
            except Exception as e:
                print(f"[TorchBell] refresh failed: {e}")
