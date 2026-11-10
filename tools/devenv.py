"""Окружение разработки: загрузка .env (значения не выводятся) и учёт токенов по полям usage.

Только для инструментов разработки (tools/, eval). Ядро .env не читает: на платформе переменные задаёт она сама.

Журнал токенов — eval/token_ledger.csv, одна строка на группу вызовов:
    ts, stage, purpose, model, calls, prompt_tokens, completion_tokens, total_tokens, cached_tokens
"""

from __future__ import annotations

import csv
import os
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "eval" / "token_ledger.csv"
LEDGER_FIELDS = ["ts", "stage", "purpose", "model", "calls", "prompt_tokens", "completion_tokens", "total_tokens",
                 "cached_tokens"]
KEY_BUDGET = 100_000_000
_LOCK = threading.Lock()


def read_dotenv(path: Path | None = None) -> dict[str, str]:
    """KEY=VALUE из .env, не трогая os.environ."""
    path = path or ROOT / ".env"
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k.startswith("export "):
            k = k[len("export "):].strip()
        out[k] = v.strip().strip('"').strip("'")
    return out


def load_dotenv(path: Path | None = None, override: bool = False) -> list[str]:
    """Загружает KEY=VALUE из .env в os.environ. Возвращает имена загруженных переменных (без значений)."""
    values = read_dotenv(path)
    for k, v in values.items():
        if override or not os.environ.get(k):
            os.environ[k] = v
    return list(values)


def cached_tokens(usage: dict) -> int:
    """Закэшированные токены запроса: DeepSeek — prompt_cache_hit_tokens, OpenAI — prompt_tokens_details.cached_tokens."""
    if not isinstance(usage, dict):
        return 0
    if usage.get("prompt_cache_hit_tokens") is not None:
        return int(usage.get("prompt_cache_hit_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0


class Usage:
    """Накопитель usage по моделям; flush() пишет строки в журнал."""

    def __init__(self, stage: str, purpose: str) -> None:
        self.stage, self.purpose = stage, purpose
        self.by_model: dict[str, dict[str, int]] = {}
        self.lock = threading.Lock()
        self.cumulative = 0   # всего токенов с создания (flush не сбрасывает)

    def add(self, model: str, usage: dict | None) -> None:
        usage = usage or {}
        with self.lock:
            m = self.by_model.setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                                 "total_tokens": 0, "cached_tokens": 0})
            m["calls"] += 1
            pt = int(usage.get("prompt_tokens") or 0)
            ct = int(usage.get("completion_tokens") or 0)
            m["prompt_tokens"] += pt
            m["completion_tokens"] += ct
            m["total_tokens"] += int(usage.get("total_tokens") or (pt + ct))
            self.cumulative += int(usage.get("total_tokens") or (pt + ct))
            m["cached_tokens"] += cached_tokens(usage)

    def total(self) -> int:
        return self.cumulative

    def flush(self) -> None:
        with self.lock:
            rows = [{"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": self.stage, "purpose": self.purpose,
                     "model": model, **m} for model, m in self.by_model.items() if m["calls"]]
            self.by_model = {}
        if not rows:
            return
        with _LOCK:
            LEDGER.parent.mkdir(exist_ok=True)
            new = not LEDGER.is_file()
            with open(LEDGER, "a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
                if new:
                    w.writeheader()
                w.writerows(rows)


def ledger_summary() -> dict:
    """Итоги журнала: по этапам и всего; остаток от бюджета ключа."""
    by_stage: dict[str, int] = {}
    total = 0
    if LEDGER.is_file():
        with open(LEDGER, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                t = int(r["total_tokens"] or 0)
                by_stage[r["stage"]] = by_stage.get(r["stage"], 0) + t
                total += t
    return {"by_stage": by_stage, "total": total, "key_left": KEY_BUDGET - total}


if __name__ == "__main__":
    s = ledger_summary()
    for stage, t in s["by_stage"].items():
        print(f"{stage:30s} {t:>12,}".replace(",", " "))
    print(f"{'всего':30s} {s['total']:>12,}".replace(",", " "))
    print(f"{'остаток ключа (по журналу)':30s} {s['key_left']:>12,}".replace(",", " "))
