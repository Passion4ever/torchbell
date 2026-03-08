"""torchbell — Lightweight training monitor with Telegram notifications."""

from .bot import TelegramBot
from .monitor import TorchBell
from .notifier import Notifier
from .email_notifier import EmailNotifier
from .setup import setup

__all__ = ["TorchBell", "TelegramBot", "Notifier", "EmailNotifier", "setup"]
__version__ = "0.2.0"
