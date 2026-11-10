"""Состояние прогона: флаги деградации, скользящие средние времени, прогресс, выбор уровня обработки."""

from __future__ import annotations

import threading
import time

from config import MAX_WORKERS, remaining


class Flags:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.chat_fail_streak = 0      # подряд неудачных вызовов модели (после всех повторов)
        self.last_probe = 0.0
        self.rerank_fail = 0           # неустранимые ошибки реранкера (404, формат)
        self.rerank_streak = 0         # подряд любых неудач реранкера
        self.rerank_off = False
        self.rerank_logged = False
        self.dense = False
        self.compact = False           # прокси отклонил полный запрос — все вопросы в компактном режиме
        self.minimal = False           # прокси отклонил и компактный — минимальный режим
        self.tiny = False              # прокси отклонил и минимальный — запрос размером со стартовый пример
        self.stats = {"llm_ok": 0, "fallback": 0, "repaired": 0, "trimmed": 0, "rerank_ok": 0,
                      "level4": 0, "level3": 0, "level2": 0, "level1": 0, "skipped": 0}

    def inc(self, k: str) -> None:
        with self.lock:
            self.stats[k] = self.stats.get(k, 0) + 1


FLAGS = Flags()


class Timing:
    """Скользящее среднее длительности вызовов: по нему выбирается уровень обработки."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ema: dict[str, float] = {}

    def add(self, key: str, value: float) -> None:
        with self.lock:
            old = self.ema.get(key)
            self.ema[key] = value if old is None else 0.7 * old + 0.3 * value

    def get(self, key: str, default: float) -> float:
        with self.lock:
            return self.ema.get(key, default)


TIMING = Timing()


def llm_seems_down() -> bool:
    """Предохранитель: после 6 подряд неудачных вызовов не ждём модель, а раз в 20 с пробуем снова."""
    with FLAGS.lock:
        if FLAGS.chat_fail_streak < 6:
            return False
        now = time.monotonic()
        if now - FLAGS.last_probe > 20:
            FLAGS.last_probe = now
            return False
        return True


class Progress:
    def __init__(self) -> None:
        self.n = 1
        self.finished = 0
        self.lock = threading.Lock()


PROG = Progress()


def per_question_budget() -> float:
    """Сколько секунд в среднем можно потратить на каждый оставшийся вопрос."""
    with PROG.lock:
        left = max(1, PROG.n - PROG.finished)
    return max(0.0, remaining() - 15) * MAX_WORKERS / left


def choose_level() -> int:
    """4 — реранкер + полный контекст + исправление чисел; 3 — полный контекст без реранкера;
    2 — компактный контекст; 1 — минимальный контекст; 0 — времени не осталось (запасной ответ)."""
    full = TIMING.get("chat_full", 8.0)
    comp = TIMING.get("chat_compact", full * 0.6)
    mini = TIMING.get("chat_min", comp * 0.7)
    if remaining() - 15 < min(mini, 15.0):
        return 0
    per_q = per_question_budget()
    if FLAGS.minimal:
        return 1
    if FLAGS.compact:
        return 2 if per_q >= comp * 1.15 else 1
    rr = 0.0 if FLAGS.rerank_off else TIMING.get("rerank", 2.0)
    if per_q >= (full + rr) * 1.6:
        return 4
    if per_q >= full * 1.2:
        return 3
    if per_q >= comp * 1.15:
        return 2
    return 1
