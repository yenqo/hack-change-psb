"""Настройки ядра из окружения, константы, общий дедлайн, журнал и отладочный дамп."""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import date
from pathlib import Path


T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    return default if not value or value == name else value


def env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(float(env(name)))))
    except ValueError:
        return default


def env_float(name: str, default: float, low: float = 0.0, high: float = 2.0) -> float:
    try:
        return max(low, min(high, float(env(name))))
    except ValueError:
        return default


DATA_DIR = Path(env("DATA_DIR", "data"))
API_BASE = (env("LLM_BASE_URL") or env("BASE_URL") or env("OPENAI_BASE_URL")).rstrip("/")
LLM_KEY = env("LLM_API_KEY") or env("OPENAI_API_KEY")
EMBED_KEY = env("EMBEDDER_API_KEY") or LLM_KEY
RERANK_KEY = env("RERANKER_API_KEY") or LLM_KEY
LLM_MODEL = env("LLM_MODEL") or env("GEN_MODEL") or env("OPENAI_MODEL", "deepseek/deepseek-v4.1-flash")
EMB_MODEL = env("EMB_MODEL", "baai/bge-m3")
RERANKER_MODEL = env("RERANKER_MODEL", "cohere/rerank-5-fast")

# Значения по умолчанию и границы — как в стартовом примере организаторов: прокси может не принимать
# запросы с параметрами сверх заданных платформой.
MAX_WORKERS = env_int("MAX_WORKERS", 2, 1, 16)
TEMPERATURE = env_float("TEMPERATURE", 0.0)
MAX_TOKENS = env_int("MAX_TOKENS", 1024, 128, 8192)
TOP_K_ENV = env_int("TOP_K", 0, 0, 200)
RERANK_N_ENV = env_int("RERANKER_TOP_N", 0, 0, 100)

# Платформа даёт 20 минут; оставляем запас на запуск контейнера и запись файла.
TIME_BUDGET = env_int("SOLUTION_TIME_BUDGET", 16 * 60 + 30, 30, 3600)
DEADLINE = T0 + TIME_BUDGET

CANDIDATES = min(40, max(30, TOP_K_ENV))          # кандидатов из гибридного поиска в реранкер
LAW_TOP = min(14, max(8, RERANK_N_ENV))           # фрагментов законов в контексте
LAW_BUDGET_CHARS = 12000                          # объём текста законов в контексте (полный режим)
SMALL_LAW_CHARS = 7000                            # короткая статья закона идёт в контекст целиком
COMPANY_FULL_LIMIT = 45000                        # до этого объёма документы компании идут целиком
CHUNK_MAX = 1600                                  # максимальная длина фрагмента закона
ANSWER_MAX_CHARS = 3000

OUTPUT_FILE = "submission.csv"
DEBUG = env("SOLUTION_DEBUG", "") == "1"   # локальная отладка: пишет debug.jsonl с контекстом и черновиками
DEBUG_LOCK = threading.Lock()

# Текущая дата для промпта. SOLUTION_TODAY (ГГГГ-ММ-ДД) фиксирует её в тестах; на платформе не задаётся.
SOLUTION_TODAY = env("SOLUTION_TODAY")


def today() -> date:
    if SOLUTION_TODAY:
        try:
            return date.fromisoformat(SOLUTION_TODAY)
        except ValueError:
            pass
    return date.today()


def debug_dump(record: dict) -> None:
    if not DEBUG:
        return
    import json
    with DEBUG_LOCK, open("debug.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def remaining() -> float:
    return DEADLINE - time.monotonic()
