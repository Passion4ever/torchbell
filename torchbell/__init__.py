"""torchbell — Lightweight training monitor with Telegram notifications."""

from .monitor import TorchBell
from .setup import setup

__all__ = ["TorchBell", "setup"]
__version__ = "0.1.1"
