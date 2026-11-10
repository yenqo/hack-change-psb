"""Конвейер ответа на один вопрос: разбор диалога, подбор законов и документов, генерация,
проверка чисел и исправление, запасной ответ."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import runtime
from config import (CANDIDATES, COMPANY_FULL_LIMIT, LAW_BUDGET_CHARS, LAW_TOP, SMALL_LAW_CHARS, debug_dump, log,
    remaining)
from textproc import expand_query
from dates import date_note
from kb import related_docs
from runtime import per_question_budget
from llm import APIError, chat, rerank
from prompts import SYSTEM_TEXT, SYSTEM_TEXT_MINI, SYSTEM_TEXT_SHORT, SYSTEM_TINY, build_user_message
from checks import clean_answer, drop_sentences, unverified

if TYPE_CHECKING:
    from kb import Chunk, Doc, KB
    from index import Index


TURN_RE = re.compile(r"^\s*(Пользователь|Ассистент|Клиент|Юрист|Сотрудник|User|Assistant|Бот|Система)\s*:\s*",
                     re.I | re.M)
ASSISTANT_ROLES = {"ассистент", "assistant", "бот", "система"}


def parse_dialog(q: str) -> tuple[str, list[tuple[str, str]]]:
    """Возвращает (текущий вопрос, история [(роль, текст)])."""
    marks = list(TURN_RE.finditer(q))
    if not marks or marks[0].start() > 0 and q[: marks[0].start()].strip():
        if len(marks) < 2:
            return q.strip(), []
    turns = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(q)
        role = "assistant" if m.group(1).lower() in ASSISTANT_ROLES else "user"
        turns.append((role, q[m.end():end].strip()))
    if len(turns) < 2:
        return (turns[0][1] if turns else q.strip()), []
    last_user = max((i for i, (r, _) in enumerate(turns) if r == "user"), default=-1)
    if last_user < 0:
        return q.strip(), []
    return turns[last_user][1], turns[:last_user]


def fallback_answer(kb: KB, index: Index, question: str) -> str:
    current, history = parse_dialog(question)
    ranked = index.bm25(expand_query(current + " " + " ".join(t for _, t in history)))
    for ci, _ in ranked[:1]:
        c = kb.chunks[ci]
        text = re.sub(r"\[Пометка редакции:[^\]]*\]", " ", c.text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 450:
            text = text[:450].rsplit(" ", 1)[0] + "…"
        return f"По данным базы знаний ({c.label}): {text}"
    return ("В базе знаний не найдено документов по этому вопросу, поэтому ответить по документам нельзя. "
            "Уточните вопрос или обратитесь к юристу.")


@dataclass
class Prepared:
    current: str
    history: list
    qvec: list | None = None


def select_law(kb: KB, index: Index, prep: Prepared, use_rerank: bool, budget_chars: int = LAW_BUDGET_CHARS):
    query = prep.current
    if prep.history:
        query += " " + " ".join(t for r, t in prep.history if r == "user")
    ranked = index.hybrid(query, prep.qvec, {"law"})[:CANDIDATES]
    if not ranked:
        return [], []
    # Если по BM25 вопрос явно про документы компании, законов даём меньше: меньше шума и быстрее.
    budget = budget_chars
    allbm = index.bm25(expand_query(query))
    if allbm:
        best_law = max((sc for ci, sc in allbm if kb.docs[kb.chunks[ci].doc].kind == "law"), default=0.0)
        best_comp = max((sc for ci, sc in allbm if kb.docs[kb.chunks[ci].doc].kind == "company"), default=0.0)
        top3_law = any(kb.docs[kb.chunks[ci].doc].kind == "law" for ci, _ in allbm[:3])
        if best_comp > 0 and best_law < 0.6 * best_comp and not top3_law:
            budget = min(budget, 5000)
    order = ranked
    if use_rerank:
        rr = rerank(prep.current if not prep.history else query[:2000],
                    [kb.chunks[i].search[:1200] for i in ranked], top_n=len(ranked))
        if rr:
            rr_sorted = [ranked[i] for i, _ in sorted(rr, key=lambda x: -x[1])]
            order = rr_sorted + [i for i in ranked if i not in set(rr_sorted)]
    top = order[:LAW_TOP]
    # Короткие статьи, попавшие в топ-4, берём целиком; длинные — найденными пунктами.
    whole = []
    for ci in order[:4]:
        d = kb.chunks[ci].doc
        if len(kb.docs[d].body) <= min(SMALL_LAW_CHARS, budget) and d not in whole:
            whole.append(d)
    items: dict[int, list[Chunk] | None] = {}
    for d in whole:
        size = len(kb.docs[d].body)
        if size <= budget:
            items[d] = None
            budget -= size
    for ci in top:
        c = kb.chunks[ci]
        if c.doc in items and items[c.doc] is None:
            continue
        if len(c.text) > budget:
            continue
        items.setdefault(c.doc, [])
        if c not in items[c.doc]:  # type: ignore[operator]
            items[c.doc].append(c)  # type: ignore[union-attr]
            budget -= len(c.text)
    # Для больших статей из топ-3 добавляем короткие пункты верхнего уровня: там обычно общие правила
    # (например, общая ставка налога), которые лексически плохо находятся по вопросу.
    seen_docs = []
    for ci in order[:3]:
        d = kb.chunks[ci].doc
        if d in seen_docs or (d in items and items[d] is None):
            continue
        seen_docs.append(d)
        extra = 0
        for c in kb.chunks:
            if c.doc != d or c.label.count(",") != 2 or len(c.text) > 700 or "(продолжение" in c.text:
                continue
            if c in (items.get(d) or []) or len(c.text) > budget or extra + len(c.text) > 2500:
                continue
            items.setdefault(d, [])
            items[d].append(c)  # type: ignore[union-attr]
            budget -= len(c.text)
            extra += len(c.text)
    # порядок статей — по лучшему рангу
    rank_of = {}
    for r, ci in enumerate(order):
        rank_of.setdefault(kb.chunks[ci].doc, r)
    law_items = sorted(items.items(), key=lambda x: rank_of.get(x[0], 999))
    return law_items, order


def company_context(kb: KB, index: Index, prep: Prepared, compact: bool = False, minimal: bool = False) -> list[Doc]:
    docs = kb.company
    if not compact and not minimal and sum(len(d.text) for d in docs) <= COMPANY_FULL_LIMIT:
        return docs
    limit = 5000 if minimal else (10000 if compact else COMPANY_FULL_LIMIT)
    max_docs = 2 if minimal else (4 if compact else 8)
    query = prep.current + " " + " ".join(t for _, t in prep.history)
    ranked = index.hybrid(query, prep.qvec, {"company"})
    chosen: list[Doc] = []
    size = 0
    for ci in ranked:
        d = kb.docs[kb.chunks[ci].doc]
        if d in chosen:
            continue
        group = [d] + [o for o in related_docs(kb, d) if o not in chosen]
        for g in group:
            if g in chosen or size + len(g.text) > limit or len(chosen) >= max_docs + 2:
                continue
            chosen.append(g)
            size += len(g.text)
        if len(chosen) >= max_docs:
            break
    return [d for d in docs if d in chosen]  # исходный порядок документов


def answer_tiny(kb: KB, index: Index, prep: Prepared) -> str:
    """Последняя ступень: запрос размером со стартовый пример организаторов."""
    query = prep.current + " " + " ".join(t for r, t in prep.history if r == "user")
    parts, size = [], 0
    for ci in index.hybrid(query, prep.qvec, {"company", "law"})[:6]:
        c = kb.chunks[ci]
        if size + len(c.text) > 2800:
            continue
        parts.append(f"[{c.label}]\n{c.text}")
        size += len(c.text)
    hist = "\n".join(("Пользователь: " if r == "user" else "Ассистент: ") + t for r, t in prep.history)
    user = "Фрагменты:\n" + "\n\n".join(parts) + (f"\n\nИстория диалога:\n{hist}" if hist else "") + \
           f"\n\nВопрос:\n{prep.current}"
    try:
        ans = clean_answer(chat([{"role": "system", "content": SYSTEM_TINY}, {"role": "user", "content": user}]))
    except APIError as exc:
        if not exc.transient:  # отклонён даже короткий запрос — дальше уменьшать некуда
            with runtime.FLAGS.lock:
                runtime.FLAGS.chat_fail_streak += 1
        raise
    if not ans:
        raise APIError("пустой ответ после очистки", transient=True)
    runtime.FLAGS.inc("level0")
    return ans


def answer_one(kb: KB, index: Index, qtext: str, prep: Prepared, level: int) -> str:
    if runtime.FLAGS.tiny:
        return answer_tiny(kb, index, prep)
    if runtime.FLAGS.minimal:
        level = 1
    elif runtime.FLAGS.compact:
        level = min(level, 2)
    law_budget = {4: LAW_BUDGET_CHARS, 3: 8000, 2: 4000}.get(level, 2000)
    law_items, _ = select_law(kb, index, prep, use_rerank=level >= 4, budget_chars=law_budget)
    comp = company_context(kb, index, prep, compact=level <= 2, minimal=level <= 1)
    dnote = date_note(qtext, kb)
    user_msg = build_user_message(kb, comp, law_items, dnote, prep.current, prep.history)
    system = SYSTEM_TEXT if level >= 3 else (SYSTEM_TEXT_SHORT if level == 2 else SYSTEM_TEXT_MINI)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
    t0 = time.monotonic()
    try:
        ans = clean_answer(chat(messages))
    except APIError as exc:
        if not exc.transient and level >= 2:
            # Прокси отклонил запрос (размер или параметры): уменьшаем запрос для всех последующих вопросов.
            with runtime.FLAGS.lock:
                if level >= 3 and not runtime.FLAGS.compact:
                    runtime.FLAGS.compact = True
                    log(f"Прокси отклонил полный запрос ({exc}) — переходим на компактный режим")
                elif level == 2 and not runtime.FLAGS.minimal:
                    runtime.FLAGS.minimal = True
                    log(f"Прокси отклонил компактный запрос ({exc}) — переходим на минимальный режим")
            return answer_one(kb, index, qtext, prep, 2 if level >= 3 else 1)
        if not exc.transient and level == 1:
            with runtime.FLAGS.lock:
                if not runtime.FLAGS.tiny:
                    runtime.FLAGS.tiny = True
                    log(f"Прокси отклонил минимальный запрос ({exc}) — переходим на короткий запрос")
            return answer_tiny(kb, index, prep)
        raise
    runtime.TIMING.add({4: "chat_full", 3: "chat_full", 2: "chat_compact"}.get(level, "chat_min"), time.monotonic() - t0)
    if not ans:
        raise APIError("пустой ответ после очистки", transient=True)
    draft = ans
    bad = unverified(ans, user_msg, qtext)
    full = runtime.TIMING.get("chat_full", 8.0)
    can_repair = level >= 4 and remaining() > 60 and per_question_budget() >= 2.2 * full
    if bad and can_repair:
        spans = [sp for sp, _ in bad]
        log(f"Неподтверждённые числа {spans[:5]} — запрашиваю исправление")
        fix = (f"Черновик ответа:\n{ans}\n\nАвтоматическая проверка: числа или даты {', '.join(spans[:8])} не найдены "
               "в документах и не получаются простым расчётом из чисел документов и вопроса. Перепиши ответ: убери или "
               "замени их точными значениями из документов. Если число получено расчётом, приведи формулу с исходными "
               "числами из документов. Остальные требования к ответу сохраняются. Верни только итоговый текст ответа.")
        try:
            fixed = clean_answer(chat(messages + [{"role": "assistant", "content": ans},
                                                  {"role": "user", "content": fix}]))
            if fixed:
                ans = fixed
                runtime.FLAGS.inc("repaired")
                bad = unverified(ans, user_msg, qtext)
        except APIError as exc:
            log(f"Исправление не удалось: {exc}")
    # Сроки и даты модель копирует из текста, а не считает: неподтверждённые — почти наверняка ошибка.
    strict = [sp for sp, kind in bad if kind in ("term", "date")]
    if strict:
        trimmed = drop_sentences(ans, strict)
        if trimmed != ans:
            runtime.FLAGS.inc("trimmed")
            log(f"Удалены предложения с неподтверждёнными сроками/датами: {strict[:5]}")
            ans = trimmed
    runtime.FLAGS.inc(f"level{level}")
    debug_dump({"question": qtext, "current": prep.current, "level": level,
                "laws": [(kb.docs[d].label, "целиком" if chs is None else [c.label for c in chs]) for d, chs in law_items],
                "company": [d.label for d in comp], "date_note": dnote, "draft": draft, "unverified": bad,
                "final": ans, "prompt_chars": len(user_msg)})
    return ans
