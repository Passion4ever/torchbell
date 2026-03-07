"""Telegram send/edit module. Single background thread consumes a message queue."""

import queue
import threading
import time
from typing import Optional

import requests

_SEND_TIMEOUT = 15
_MAX_RETRIES = 2


class TelegramBot:
    def __init__(self, token: str, chat_id: int, silent: bool = False):
        self._base = f"https://api.telegram.org/bot{token}"
        self._chat_id = chat_id
        self._silent = silent
        self._queue: queue.Queue = queue.Queue()
        self._sender_thread: Optional[threading.Thread] = None

    def send(self, text: str, block: bool = False) -> Optional[int]:
        """Queue a message. block=True waits for completion."""
        result = {}
        done_event = threading.Event() if block else None
        self._queue.put(("send", text, None, done_event, result))
        self._ensure_sender()
        if done_event:
            done_event.wait(timeout=_SEND_TIMEOUT)
        return result.get("message_id")

    def send_sync(self, text: str) -> Optional[int]:
        """Synchronous send with retries. Returns message_id."""
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self._do_send(text)
            except BaseException as e:
                if attempt < _MAX_RETRIES:
                    time.sleep(1)
                else:
                    print(f"[TorchBell] send failed ({_MAX_RETRIES} retries): {e}")
        return None

    def edit(self, message_id: int, text: str):
        """Queue an edit."""
        self._queue.put(("edit", text, message_id, None, None))
        self._ensure_sender()

    def edit_sync(self, message_id: int, text: str):
        """Synchronous edit with retries."""
        for attempt in range(_MAX_RETRIES + 1):
            try:
                self._do_edit(message_id, text)
                return
            except BaseException as e:
                if attempt < _MAX_RETRIES:
                    time.sleep(1)
                else:
                    print(f"[TorchBell] edit failed ({_MAX_RETRIES} retries): {e}")

    def _ensure_sender(self):
        if self._sender_thread and self._sender_thread.is_alive():
            return
        self._sender_thread = threading.Thread(
            target=self._send_loop, daemon=True, name="TGBot-Sender"
        )
        self._sender_thread.start()

    def _send_loop(self):
        while True:
            try:
                action, text, msg_id, done_event, result = self._queue.get(timeout=5)
            except queue.Empty:
                return
            try:
                if action == "send":
                    mid = self._do_send(text)
                    if result is not None and mid:
                        result["message_id"] = mid
                elif action == "edit":
                    self._do_edit(msg_id, text)
            except Exception as e:
                print(f"[TorchBell] send failed: {e}")
            finally:
                if done_event:
                    done_event.set()

    def _do_send(self, text: str) -> Optional[int]:
        resp = requests.post(
            f"{self._base}/sendMessage",
            json={
                "chat_id": self._chat_id,
                "text": text[:4096],
                "disable_notification": self._silent,
                "parse_mode": "HTML",
            },
            timeout=_SEND_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("ok"):
            return data["result"]["message_id"]
        print(f"[TorchBell] send error: {data.get('description', 'unknown')}")
        return None

    def _do_edit(self, message_id: int, text: str):
        resp = requests.post(
            f"{self._base}/editMessageText",
            json={
                "chat_id": self._chat_id,
                "message_id": message_id,
                "text": text[:4096],
                "parse_mode": "HTML",
            },
            timeout=_SEND_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            desc = data.get("description", "")
            if "message is not modified" not in desc:
                print(f"[TorchBell] edit error: {desc}")
