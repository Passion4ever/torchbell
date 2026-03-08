"""Tests for torchbell.notifier (Notifier ABC)."""

import pytest

from torchbell.notifier import Notifier


def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        Notifier()


class _MinimalNotifier(Notifier):
    def send(self, text, block=False):
        return None

    def send_sync(self, text):
        return None


def test_subclass_with_required_methods():
    n = _MinimalNotifier()
    assert n.send("hi") is None
    assert n.send_sync("hi") is None


def test_default_edit_is_noop():
    n = _MinimalNotifier()
    # Should not raise
    n.edit(1, "text")
    n.edit_sync(1, "text")


def test_default_supports_edit_is_false():
    n = _MinimalNotifier()
    assert n.supports_edit is False


def test_repr_shows_class_name():
    n = _MinimalNotifier()
    assert repr(n) == "<_MinimalNotifier>"
