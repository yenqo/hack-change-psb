"""Виртуальные часы для симуляции долгих прогонов (медленный прокси, сотни вопросов) за секунды реального времени.

time.monotonic / time.sleep подменяются. У каждого рабочего потока — своя «полоса» времени (как у параллельного
обращения к прокси); главный поток видит максимум по полосам. Заглушка прокси вызывает advance(задержка). Реальная
пауза — задержка × scale, чтобы потоки сменяли друг друга в том же порядке, что и в виртуальном времени.
"""

from __future__ import annotations

import threading
import time


class SimClock:
    def __init__(self, scale: float = 0.002, start: float = 1000.0) -> None:
        self.scale = scale
        self.lock = threading.Lock()
        self.local = threading.local()
        self.maxt = start
        self.start = start
        self.real_sleep = time.sleep

    def _lane(self) -> float | None:
        if threading.current_thread() is threading.main_thread():
            return None
        t = getattr(self.local, "t", None)
        if t is None:
            with self.lock:
                t = self.maxt
            self.local.t = t
        return t

    def monotonic(self) -> float:
        t = self._lane()
        if t is None:
            with self.lock:
                return self.maxt
        return t

    def advance(self, seconds: float) -> None:
        if seconds is None or seconds <= 0:
            return
        t = self._lane()
        if t is None:
            with self.lock:
                self.maxt += seconds
        else:
            self.local.t = t + seconds
            with self.lock:
                self.maxt = max(self.maxt, self.local.t)
        self.real_sleep(seconds * self.scale)

    def elapsed(self) -> float:
        with self.lock:
            return self.maxt - self.start

    def install(self) -> None:
        time.monotonic = self.monotonic
        time.sleep = self.advance
