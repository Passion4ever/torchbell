"""Live test — verify all notification scenarios.

Usage:
    cp .env.example .env   # fill in your credentials
    pip install -e .
    python examples/test_live.py
"""

import time
import sys

from torchbell import TorchBell


def test_normal_finish():
    """Normal lifecycle: start → log → finish."""
    print("\n=== Test 1: Normal Finish ===")
    bell = TorchBell(run_name="Test Normal", refresh_interval=5)
    bell.start(total=5)
    for step in range(1, 6):
        time.sleep(2)
        bell.log(step, {"loss": 1.0 / step, "acc": step * 0.18})
        print(f"  step {step}/5")
    bell.finish(final_metrics={"loss": 0.2, "acc": 0.90})
    print("  -> should receive: status update + completion notification")


def test_error():
    """Crash lifecycle: start → error."""
    print("\n=== Test 2: Error / Crash ===")
    bell = TorchBell(run_name="Test Crash", refresh_interval=5)
    bell.start(total=100)
    for step in range(1, 4):
        time.sleep(1)
        bell.log(step, {"loss": 0.5})
    try:
        raise RuntimeError("CUDA out of memory (simulated)")
    except RuntimeError as e:
        bell.error(e)
    print("  -> should receive: status update + crash notification with traceback")


def test_custom_notify():
    """Custom notification via notify()."""
    print("\n=== Test 3: Custom Notify ===")
    bell = TorchBell(run_name="Test Notify", refresh_interval=5)
    bell.notify("This is a <b>custom</b> notification message.")
    print("  -> should receive: custom message")


def test_watch_decorator():
    """Watch decorator: auto start/finish."""
    print("\n=== Test 4: Watch Decorator ===")
    bell = TorchBell(run_name="Test Watch", refresh_interval=5)

    @bell.watch(total=3)
    def train():
        for step in range(1, 4):
            time.sleep(2)
            bell.log(step, {"loss": 0.3, "lr": 1e-4})
            print(f"  step {step}/3")

    train()
    print("  -> should receive: status update + completion notification")


def test_watch_decorator_crash():
    """Watch decorator: auto error on exception."""
    print("\n=== Test 5: Watch Decorator Crash ===")
    bell = TorchBell(run_name="Test Watch Crash", refresh_interval=5)

    @bell.watch(total=10)
    def train():
        for step in range(1, 4):
            time.sleep(1)
            bell.log(step, {"loss": 0.5})
        raise ValueError("NaN loss detected (simulated)")

    try:
        train()
    except ValueError:
        pass
    print("  -> should receive: status update + crash notification")


def test_no_total():
    """No total provided — no ETA, no progress bar."""
    print("\n=== Test 6: No Total (open-ended) ===")
    bell = TorchBell(run_name="Test No Total", refresh_interval=5)
    bell.start()
    for step in range(1, 4):
        time.sleep(2)
        bell.log(step, {"loss": 0.8 - step * 0.1})
        print(f"  step {step}")
    bell.finish()
    print("  -> should receive: status (no ETA) + completion notification")


def test_refresh_update():
    """Longer run to verify real-time status refresh."""
    print("\n=== Test 7: Refresh Update (15s) ===")
    bell = TorchBell(run_name="Test Refresh", refresh_interval=5)
    bell.start(total=5)
    for step in range(1, 6):
        time.sleep(3)
        bell.log(step, {"loss": 1.0 / step})
        print(f"  step {step}/5")
    bell.finish()
    print("  -> Telegram status message should have updated in-place 2-3 times")


def main():
    tests = [
        test_normal_finish,
        test_error,
        test_custom_notify,
        test_watch_decorator,
        test_watch_decorator_crash,
        test_no_total,
        test_refresh_update,
    ]

    print("Running {} live tests...".format(len(tests)))
    print("Check your notification channels after each test.")

    for i, test_fn in enumerate(tests):
        test_fn()
        if i < len(tests) - 1:
            print("\n  -- pausing 3s before next test --")
            time.sleep(3)

    print("\n=== All {} tests done! ===".format(len(tests)))
    print("Go check your Telegram / Email for {} groups of messages.".format(len(tests)))


if __name__ == "__main__":
    main()
