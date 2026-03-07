# TorchBell

Lightweight training monitor with Telegram notifications.

One self-updating status message tracks your training in real time. When training finishes, crashes, or is manually stopped, you get a phone notification.

## Install

```bash
pip install torchbell
```

## Setup

1. Create a Telegram bot via [@BotFather](https://t.me/BotFather)
2. Set environment variables:

```bash
export TG_BOT_TOKEN="your-token"
export TG_CHAT_ID="your-chat-id"
```

## Quick Start

```python
from torchbell import TorchBell

bell = TorchBell(run_name="my-experiment")

@bell.watch(total=100)
def train():
    for epoch in range(100):
        loss = train_one_epoch(...)
        bell.log(epoch + 1, {"loss": loss})

train()
```

That's it. You'll get a live status message on Telegram and a notification when it's done (or if it crashes).

## License

Apache-2.0
