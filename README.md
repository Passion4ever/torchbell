# 🔔 TorchBell

[![PyPI](https://img.shields.io/pypi/v/torchbell)](https://pypi.org/project/torchbell/)
[![Python](https://img.shields.io/pypi/pyversions/torchbell)](https://pypi.org/project/torchbell/)
[![License](https://img.shields.io/github/license/Passion4ever/torchbell)](https://github.com/Passion4ever/torchbell/blob/main/LICENSE)

Monitor your training remotely.
Get notified when it completes, crashes, or is stopped — no need to watch the terminal.

## 📦 Install

```bash
pip install torchbell
```

## ⚙️ Setup

1. Create a Telegram bot via [@BotFather](https://t.me/BotFather) and get your **token**
2. Send a message to your bot, then get your **chat_id** from `https://api.telegram.org/bot<token>/getUpdates`
3. Set environment variables or pass them directly:

```bash
export TG_BOT_TOKEN="your-token"
export TG_CHAT_ID="your-chat-id"
```

```python
bell = TorchBell(token="your-token", chat_id="your-chat-id")
```

## 🚀 Quick Start

```python
from torchbell import TorchBell

bell = TorchBell(run_name="my-experiment")

@bell.watch()
def train():
    # your training code, unchanged
    ...

train()
```

That's it. You'll get a status message on Telegram and a notification when it's done (or if it crashes). 🎉

**Want progress and metrics?** Add `bell.log()` inside your loop:

```python
@bell.watch(total=100)
def train():
    for epoch in range(100):
        loss = train_one_epoch(...)
        bell.log(epoch + 1, {"loss": loss})

train()
```
