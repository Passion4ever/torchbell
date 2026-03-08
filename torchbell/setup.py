"""Interactive setup helper: validate token and retrieve chat_id."""

from typing import Any, Dict

import requests

_API_TIMEOUT = 15


def _mask_token(token: str) -> str:
    """Return masked token showing only last 4 characters."""
    if len(token) <= 4:
        return "****"
    return "****:****" + token[-4:]


def setup(token: str) -> None:
    """Validate a Telegram bot token and print the chat_id.

    Usage::

        python -c "import torchbell; torchbell.setup('YOUR_TOKEN')"
    """
    base = "https://api.telegram.org/bot{}".format(token)
    masked = _mask_token(token)

    # 1. Validate token via getMe
    try:
        resp = requests.get(
            "{}/getMe".format(base), timeout=_API_TIMEOUT
        )
    except requests.RequestException as e:
        print("[TorchBell] Setup failed: network error — {}".format(e))
        return

    if resp.status_code == 401:
        print("[TorchBell] Setup failed: invalid token.")
        return

    data = resp.json()  # type: Dict[str, Any]
    if not data.get("ok"):
        print(
            "[TorchBell] Setup failed: {}".format(
                data.get("description", "unknown error")
            )
        )
        return

    bot_name = data["result"].get("first_name", "Unknown")

    # 2. Fetch chat_id via getUpdates
    try:
        resp = requests.get(
            "{}/getUpdates".format(base), timeout=_API_TIMEOUT
        )
    except requests.RequestException as e:
        print("[TorchBell] Setup failed: network error — {}".format(e))
        return

    data = resp.json()
    results = data.get("result", [])

    if not results:
        print("[TorchBell] No messages found.")
        print("  Send a message (e.g. /start) to your bot first, then re-run setup.")
        return

    # Use the most recent message's chat id
    chat_id = None
    for update in reversed(results):
        msg = update.get("message")
        if msg and msg.get("chat"):
            chat_id = msg["chat"]["id"]
            break

    if chat_id is None:
        print("[TorchBell] No messages found.")
        print("  Send a message (e.g. /start) to your bot first, then re-run setup.")
        return

    # 3. Print results
    print("[TorchBell] Setup")
    print("  Bot:     {}".format(bot_name))
    print("  Chat ID: {}".format(chat_id))
    print()
    print("Set environment variables:")
    print('  export TG_BOT_TOKEN="{}"'.format(masked))
    print('  export TG_CHAT_ID="{}"'.format(chat_id))
    print()
    print("Never hardcode tokens in your code. Use environment variables.")
