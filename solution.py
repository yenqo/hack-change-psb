"""ПСБ Hack&Change 2026 — юридический ассистент, версия v1.3 (сабмит 09.11.2026, 24,652).

Точка входа платформы: читает DATA_DIR/questions.csv и базу знаний, пишет submission.csv.
Зависимости: стандартная библиотека Python + httpx.

Конвейер на вопрос:
  разбор диалога -> гибридный поиск по законам (BM25 + bge-m3 -> RRF -> реранкер)
  -> контекст: все документы компании целиком + найденные пункты законов + справка о редакциях
  -> генерация (DeepSeek через прокси) -> проверка чисел кодом -> при необходимости одно исправление
  -> запасной экстрактивный ответ при любом сбое.

Модули ядра лежат рядом (плоско в корне): config, textproc, dates, kb, runtime, llm, index,
prompts, checks, pipeline, probe. Здесь — только пакетный прогон: вопросы → submission.csv.
"""

from __future__ import annotations

import csv
import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import runtime
from config import API_BASE, LLM_KEY, MAX_WORKERS, OUTPUT_FILE, log, remaining
from kb import KB, build_kb, decode_bytes, find_questions
from runtime import choose_level
from index import Index, embed_questions
from pipeline import Prepared, answer_one, fallback_answer, parse_dialog
from probe import log_env, platform_probe, run_stats, wait_for_llm


def load_questions(path: Path) -> list[tuple[str, str]]:
    raw = path.read_bytes()
    text = decode_bytes(raw)
    reader = csv.DictReader(io.StringIO(text))
    fields = [f.strip().lower() for f in (reader.fieldnames or [])]
    if "id" not in fields or "question" not in fields:
        raise ValueError("В questions.csv нужны колонки id и question")
    reader.fieldnames = fields
    rows, seen = [], set()
    for r in reader:
        qid = str(r.get("id") or "").strip()
        if not qid or qid in seen:
            continue
        seen.add(qid)
        rows.append((qid, str(r.get("question") or "").strip()))
    if not rows:
        raise ValueError("В questions.csv нет вопросов")
    return rows


WRITE_LOCK = threading.Lock()


def write_submission(questions: list[tuple[str, str]], answers: list[str]) -> None:
    with WRITE_LOCK:
        tmp = OUTPUT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
            w.writerow(["id", "answer"])
            for (qid, _), a in zip(questions, answers):
                a = (a or "").strip() or "Не удалось сформировать ответ по базе знаний."
                w.writerow([qid, a[:9500]])
        os.replace(tmp, OUTPUT_FILE)


def main() -> None:
    log_env()
    questions = load_questions(find_questions())
    n = len(questions)
    log(f"Вопросов: {n}")
    try:
        kb = build_kb()
        index = Index(kb)
        answers = [fallback_answer(kb, index, q) for _, q in questions]
    except Exception as exc:  # noqa: BLE001 — без базы всё равно отдаём корректный файл
        log(f"База знаний не загружена: {type(exc).__name__}: {exc}")
        kb = KB([], [])
        index = Index(kb)
        answers = ["В базе знаний не найдено документов по этому вопросу, поэтому ответить по документам нельзя."] * n
    write_submission(questions, answers)  # страховка: полный файл есть с первых секунд
    log("Записан предварительный submission.csv с запасными ответами")

    if not API_BASE or not LLM_KEY:
        log("Нет доступа к API — остаются запасные ответы")
        platform_probe(kb, questions, index, None, None)
        run_stats(n, n, 0)
        return

    preps = [Prepared(*parse_dialog(q)) for _, q in questions]
    llm_ready = wait_for_llm()  # эмбеддинги строим после того, как прокси ответил
    dense_sec = None
    if kb.chunks:
        t_dense = time.monotonic()
        index.build_dense(cap=60.0)
        dense_sec = round(time.monotonic() - t_dense, 1)
        if index.vecs is not None:
            embed_questions(preps)
    platform_probe(kb, questions, index, llm_ready, dense_sec)

    runtime.PROG.n = n
    failed: list[int] = []
    lock = threading.Lock()

    def work(i: int) -> tuple[int, str | None]:
        try:
            level = choose_level()
            if level == 0:
                runtime.FLAGS.inc("skipped")
                return i, None
            ans = answer_one(kb, index, questions[i][1], preps[i], level)
            runtime.FLAGS.inc("llm_ok")
            return i, ans
        except Exception as exc:  # noqa: BLE001
            log(f"Вопрос {questions[i][0]}: запасной ответ ({type(exc).__name__}: {exc})")
            return i, None
        finally:
            with runtime.PROG.lock:
                runtime.PROG.finished += 1

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for fut in as_completed([pool.submit(work, i) for i in range(n)]):
            i, ans = fut.result()
            with lock:
                if ans:
                    answers[i] = ans
                else:
                    failed.append(i)
                done += 1
                k = done
            if k % 10 == 0 or k == n:
                write_submission(questions, answers)
                log(f"Обработано {k}/{n}; уровень сейчас {choose_level()}, "
                    f"chat≈{runtime.TIMING.get('chat_full', 0):.1f}/{runtime.TIMING.get('chat_compact', 0):.1f} c")
    # Повторные проходы по вопросам с запасным ответом, пока есть время
    rounds = 0
    while failed and remaining() > 60 and rounds < 20:
        rounds += 1
        before = len(failed)
        log(f"Проход {rounds + 1}: повторяем {before} вопросов")
        runtime.PROG.n, runtime.PROG.finished = len(failed), 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for fut in as_completed([pool.submit(work, i) for i in sorted(failed)]):
                i, ans = fut.result()
                if ans:
                    answers[i] = ans
                    failed.remove(i)
        write_submission(questions, answers)
        if len(failed) == before:  # ничего не получилось — подождать, прежде чем пробовать снова
            time.sleep(min(20.0, max(0.0, remaining() - 60)))
    runtime.FLAGS.stats["fallback"] = len(failed)
    write_submission(questions, answers)
    run_stats(n, len(failed), rounds)
    log(f"Готово: {runtime.FLAGS.stats}, tiny={runtime.FLAGS.tiny}, chat≈{runtime.TIMING.get('chat_full', 0):.1f} c (полный) / "
        f"{runtime.TIMING.get('chat_compact', 0):.1f} c (компактный) / {runtime.TIMING.get('chat_min', 0):.1f} c (минимальный), "
        f"dense={runtime.FLAGS.dense}, compact={runtime.FLAGS.compact}, minimal={runtime.FLAGS.minimal}, rerank_off={runtime.FLAGS.rerank_off}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        log(f"Критическая ошибка: {type(exc).__name__}: {exc}")
        raise
