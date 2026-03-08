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
2. Send any message to your bot (e.g. `/start`)
3. Run the setup helper — it will validate your token and print the `export` commands you need:

```bash
python -c "import torchbell; torchbell.setup('YOUR_TOKEN')"
```

Output:
```
[TorchBell] Setup
  Bot:     MyTrainBot
  Chat ID: 123456789

Set environment variables:
  export TG_BOT_TOKEN="****...ABcD"
  export TG_CHAT_ID="123456789"
```

4. Set the environment variables `TG_BOT_TOKEN` and `TG_CHAT_ID`.

> **Do not hardcode tokens in your code.** Always use environment variables.

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
