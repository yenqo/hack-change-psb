"""Проверки платформы на старте: какие переменные окружения заданы, готовность модели."""

from __future__ import annotations

import os
import time

from config import (DATA_DIR, EMB_MODEL, LLM_MODEL, MAX_TOKENS, MAX_WORKERS, RERANKER_MODEL, TEMPERATURE, env, log,
    remaining)
from llm import APIError, _chat


def log_env() -> None:
    present = {k: bool(os.getenv(k)) for k in ("LLM_BASE_URL", "OPENAI_BASE_URL", "LLM_API_KEY", "OPENAI_API_KEY",
                                                "LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL")}
    log(f"Переменные: {present}")
    log(f"Модель={LLM_MODEL} EMB={EMB_MODEL} RERANK={RERANKER_MODEL} MAX_WORKERS={MAX_WORKERS} "
        f"TEMPERATURE={TEMPERATURE} MAX_TOKENS(env)={env('MAX_TOKENS')} TOP_K={env('TOP_K')} "
        f"RERANKER_TOP_N={env('RERANKER_TOP_N')} CHUNKER={env('CHUNKER')} DATA_DIR={DATA_DIR}")


def wait_for_llm() -> bool:
    """Пробный короткий вызов модели. Пока прокси не отвечает (сеть, 5xx, 429), ждём с паузами:
    на платформе модель или прокси могут подняться не сразу. Отказ 4xx не ждём — его обработает
    уменьшение запроса."""
    delay, attempt = 2.0, 0
    probe = [{"role": "user", "content": "Ответь одним словом: готов."}]
    while remaining() > 150:
        attempt += 1
        t0 = time.monotonic()
        try:
            _chat(probe, MAX_TOKENS, 90.0)
            log(f"Модель отвечает (проба {attempt}, {time.monotonic() - t0:.1f} c)")
            return True
        except APIError as exc:
            if not exc.transient:
                log(f"Проба модели: отказ без повтора ({exc}) — продолжаем с уменьшением запроса")
                return False
            log(f"Проба модели {attempt}: нет ответа ({exc}), ждём {delay:.0f} c")
        time.sleep(min(delay, max(0.0, remaining() - 150)))
        delay = min(delay * 2, 20.0)
    log("Модель так и не ответила на пробу — продолжаем, вопросы будут ждать её сами")
    return False
