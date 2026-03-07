"""
全自动综合测试：依次测试三种场景，全部自动完成。
1. 正常完成（有 total，8 steps，每步 2s）
2. 崩溃（有 total，第 5 步崩溃）
3. 无 total + 自动 kill（跑 10s 后自动发 SIGINT）
"""

import math
import os
import random
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_env(path=os.path.join(os.path.dirname(__file__), "..", ".env")):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

_load_env()

from torchbell import TorchBell


def fake_metrics(step):
    loss = 2.0 * math.exp(-0.04 * step) + random.uniform(-0.05, 0.05)
    acc = min(0.95, 0.3 + 0.02 * step + random.uniform(-0.02, 0.02))
    lr = 1e-3 * (0.95 ** step)
    return {"loss": round(loss, 4), "acc": round(acc, 4), "lr": round(lr, 6)}


# ── 测试 1：正常完成 ──────────────────────────

print("=" * 40)
print("测试 1：正常完成（8 steps）")
print("=" * 40)

bell1 = TorchBell(run_name="Test-Normal")

@bell1.watch(total=8)
def train_normal():
    for step in range(8):
        time.sleep(2)
        bell1.log(step + 1, fake_metrics(step))
        print(f"  step {step + 1}/8")

train_normal()
print("测试 1 完成 ✓\n")
time.sleep(3)


# ── 测试 2：崩溃 ─────────────────────────────

print("=" * 40)
print("测试 2：中途崩溃（第 5 步）")
print("=" * 40)

bell2 = TorchBell(run_name="Test-Crash")

@bell2.watch(total=15)
def train_crash():
    for step in range(15):
        time.sleep(2)
        bell2.log(step + 1, fake_metrics(step))
        print(f"  step {step + 1}/15")
        if step == 4:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

try:
    train_crash()
except RuntimeError:
    pass

print("测试 2 完成 ✓\n")
time.sleep(3)


# ── 测试 3：无 total + 自动 kill ──────────────

print("=" * 40)
print("测试 3：无 total 模式（10s 后自动中止）")
print("=" * 40)

# 用子进程跑，主进程 10s 后发 SIGINT
script = '''
import math, os, random, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def _load_env(path=os.path.join(os.path.dirname(__file__), "..", ".env")):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())
_load_env()

from torchbell import TorchBell
bell = TorchBell(run_name="Test-CtrlC")

@bell.watch()
def train():
    step = 0
    while True:
        time.sleep(2)
        step += 1
        loss = 2.0 * math.exp(-0.04 * step) + random.uniform(-0.05, 0.05)
        acc = min(0.95, 0.3 + 0.02 * step + random.uniform(-0.02, 0.02))
        bell.log(step, {"loss": round(loss, 4), "acc": round(acc, 4)})
        print(f"  step {step} ...")

try:
    train()
except KeyboardInterrupt:
    print("\\n手动中止 ✓")
'''

# 写临时脚本
tmp_script = os.path.join(os.path.dirname(__file__), "_tmp_ctrlc.py")
with open(tmp_script, "w") as f:
    f.write(script)

proc = subprocess.Popen([sys.executable, tmp_script])
time.sleep(12)  # 让它跑几个 step
print("  → 发送 SIGINT...")
proc.send_signal(signal.SIGINT)
proc.wait(timeout=15)
print("测试 3 完成 ✓\n")

# 清理临时文件
os.remove(tmp_script)

print("=" * 40)
print("全部测试完成！去 TG 看效果吧")
print("=" * 40)
