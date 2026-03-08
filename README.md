# 🔔 TorchBell

[![PyPI](https://img.shields.io/pypi/v/torchbell)](https://pypi.org/project/torchbell/)
[![Python](https://img.shields.io/pypi/pyversions/torchbell)](https://pypi.org/project/torchbell/)
[![License](https://img.shields.io/github/license/Passion4ever/torchbell)](https://github.com/Passion4ever/torchbell/blob/main/LICENSE)

Monitor your training remotely.
Get notified when it completes, crashes, or is stopped — via **Telegram** and/or **Email**.

## 📦 Install

```bash
pip install torchbell
```

## ⚙️ Setup

### Telegram

1. Create a bot via [@BotFather](https://t.me/BotFather) and get your **token**
2. Send any message to your bot (e.g. `/start`)
3. Run the setup helper:

```bash
python -c "import torchbell; torchbell.setup('YOUR_TOKEN')"
```

4. Set the environment variables:

```bash
export TG_BOT_TOKEN="your-token"
export TG_CHAT_ID="123456789"
```

### Email

Set the SMTP environment variables:

```bash
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="465"          # 465 = SSL, 587 = STARTTLS
export SMTP_USER="you@gmail.com"
export SMTP_PASS="app-password"
export SMTP_TO="recipient@example.com"  # optional, defaults to SMTP_USER
```

> **Tip:** Set both Telegram and Email variables to receive notifications on both channels simultaneously.

> **Do not hardcode credentials in your code.** Always use environment variables or a `.env` file.

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

That's it. You'll get a notification when it's done (or if it crashes).

**Want progress and metrics?** Add `bell.log()` inside your loop:

```python
@bell.watch(total=100)
def train():
    for epoch in range(100):
        loss = train_one_epoch(...)
        bell.log(epoch + 1, {"loss": loss})

train()
```

## 📖 API

### Constructor

```python
TorchBell(
    run_name="Training",     # experiment name
    token=None,              # Telegram token (or env TG_BOT_TOKEN)
    chat_id=None,            # Telegram chat ID (or env TG_CHAT_ID)
    notifier=None,           # custom Notifier or list of Notifiers
    unit=None,               # display unit ("epoch", "step")
    refresh_interval=30,     # status refresh interval in seconds
)
```

### Methods

| Method | Description |
|--------|-------------|
| `start(total=None)` | Begin monitoring |
| `log(step, metrics)` | Report progress and metrics |
| `finish(final_metrics=None)` | Training completed |
| `error(exception=None)` | Training crashed |
| `notify(message)` | Send a custom notification |
| `watch(total=None)` | Decorator: auto start/finish/error |

### Custom Notifiers

Use the `notifier=` parameter for advanced setups:

```python
from torchbell import TorchBell, EmailNotifier

# Email only
bell = TorchBell(
    run_name="my-run",
    notifier=EmailNotifier("smtp.gmail.com", 465, "user", "pass"),
)

# Multiple channels
from torchbell.bot import TelegramBot
bell = TorchBell(
    run_name="my-run",
    notifier=[
        TelegramBot("token", 123456),
        EmailNotifier("smtp.gmail.com", 465, "user", "pass"),
    ],
)
```

## 🔧 Multi-GPU

TorchBell automatically detects multi-GPU setups (Accelerate / DDP / SLURM) and only sends notifications from the main process.

## 📝 License

[Apache-2.0](LICENSE)
