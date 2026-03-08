"""Abstract base class for notification backends."""

from abc import ABC, abstractmethod
from typing import Optional


class Notifier(ABC):
    """Unified interface for all notification channels."""

    @abstractmethod
    def send(self, text: str, block: bool = False) -> Optional[int]:
        """Send a message. Returns message_id if available."""

    @abstractmethod
    def send_sync(self, text: str) -> Optional[int]:
        """Synchronous send with retries. Returns message_id if available."""

    def edit(self, message_id: int, text: str) -> None:
        """Edit an existing message. Default no-op for channels that don't support editing."""

    def edit_sync(self, message_id: int, text: str) -> None:
        """Synchronous edit with retries. Default no-op."""

    def __repr__(self) -> str:
        return "<{}>".format(self.__class__.__name__)

    @property
    def supports_edit(self) -> bool:
        """Whether this notifier supports editing messages in-place."""
        return False
