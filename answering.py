"""Генерация ответа на одном уровне контекста: сообщение модели, проверка чисел и ссылок, исправление,
понижение уровня при отказах прокси (B5), короткий запрос (последняя ступень)."""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

import runtime
from checks import (clean_answer, drop_ref_sentences, drop_sentences, trim_after_source, unverified,
                    unverified_refs)
from config import (ANSWER_TEMPLATE, CHECK_FIXES, CHECK_REFS, CONCISE, DATE_CONDITIONS, LAW_BUDGET_CHARS, REWRITE,
                    REWRITE_PROMPT, SOFT_DEGRADE, debug_dump, log, remaining)
from dates import date_conditions, date_note
from dialog import Prepared, search_query, user_text
from llm import APIError, chat
from prompts import (REPAIR_REFS, SYSTEM_TEXT, SYSTEM_TEXT_MINI, SYSTEM_TEXT_SHORT, SYSTEM_TINY, build_user_message,
                     repair_refs_message)
from retrieval import company_context, maybe_rewrite, select_law
from runtime import per_question_budget

if TYPE_CHECKING:
    from index import Index
    from kb import KB


def answer_tiny(kb: KB, index: Index, prep: Prepared, trace=None, info: dict | None = None) -> str:
    """Последняя ступень: запрос размером со стартовый пример организаторов."""
    query = search_query(prep)
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
    if trace:
        trace("generate")
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
    if info is not None:
        info.update(level=0, chunks=[c for c in index.hybrid(query, prep.qvec, {"company", "law"})[:6]])
    return ans


_SIZE_ERROR = re.compile(r"(?i)context|too long|too large|maximum|max_?tokens|token limit|length|слишком|превыш")


def _size_error(exc: APIError) -> bool:
    """4xx «запрос слишком велик» (413 или упоминание длины контекста в тексте ошибки прокси)."""
    return exc.status == 413 or bool(_SIZE_ERROR.search(exc.detail or ""))


def _maybe_recover() -> None:
    """B5: режим, пониженный из-за серии прочих 4xx, через 10 вопросов снова пробуем полный."""
    f = runtime.FLAGS
    with f.lock:
        if f.degraded_at >= 0 and (f.compact or f.minimal or f.tiny):
            with runtime.PROG.lock:
                done = runtime.PROG.finished
            if done - f.degraded_at >= 10:
                f.compact = f.minimal = f.tiny = False
                f.degraded_at = -1
                log("Пробуем снова полный режим запроса")


def _soft_4xx(kb: KB, index: Index, qtext: str, prep: Prepared, level: int, exc: APIError, trace, info,
              retried: bool) -> str:
    """B5: прочий 4xx (не «слишком велик»): повтор того же уровня; не помогло — этот вопрос уровнем ниже;
    глобально понижаем только после 3 таких вопросов подряд и с последующей повторной пробой."""
    if not retried:
        log(f"Прокси отклонил запрос ({exc}) — повторяем тот же уровень")
        time.sleep(1.0)
        return answer_one(kb, index, qtext, prep, level, trace, info, _retry=True)
    f = runtime.FLAGS
    with f.lock:
        if not prep.soft4xx_counted:   # ступени понижения одного вопроса — один вопрос в серии
            prep.soft4xx_counted = True
            f.soft4xx_streak += 1
        if f.soft4xx_streak >= 3:
            f.soft4xx_streak = 0
            if level >= 3:
                f.compact = True
            elif level == 2:
                f.minimal = True
            else:
                f.tiny = True
            with runtime.PROG.lock:
                f.degraded_at = runtime.PROG.finished
            log(f"Три вопроса подряд с отказом прокси ({exc}) — временно понижаем режим")
    f.inc("soft4xx")
    if level >= 3:
        return answer_one(kb, index, qtext, prep, 2, trace, info)
    if level == 2:
        return answer_one(kb, index, qtext, prep, 1, trace, info)
    return answer_tiny(kb, index, prep, trace, info)


def answer_one(kb: KB, index: Index, qtext: str, prep: Prepared, level: int, trace=None, info: dict | None = None,
               _retry: bool = False) -> str:
    """trace(stage) — необязательный колбэк этапов (search / generate / verify) для веб-интерфейса;
    info — необязательный словарь, куда кладутся сведения об ответе (уровень, контекст, черновик, проверки).
    Ни то, ни другое на текст ответа не влияет."""
    if trace:
        trace("search")
    if SOFT_DEGRADE:
        _maybe_recover()
    if runtime.FLAGS.tiny:
        return answer_tiny(kb, index, prep, trace, info)
    if runtime.FLAGS.minimal:
        level = 1
    elif runtime.FLAGS.compact:
        level = min(level, 2)
    sel: dict = {}
    law_items, comp, dnote, user_msg = _prepare_message(kb, index, qtext, prep, level, sel)
    system = SYSTEM_TEXT if level >= 3 else (SYSTEM_TEXT_SHORT if level == 2 else SYSTEM_TEXT_MINI)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
    if trace:
        trace("generate")
    t0 = time.monotonic()
    try:
        ans = clean_answer(chat(messages))
    except APIError as exc:
        lower = _on_reject(kb, index, qtext, prep, level, exc, trace, info, _retry)
        if lower is None:
            raise
        return lower
    runtime.TIMING.add({4: "chat_full", 3: "chat_full", 2: "chat_compact"}.get(level, "chat_min"), time.monotonic() - t0)
    if SOFT_DEGRADE and level >= 3:
        with runtime.FLAGS.lock:
            runtime.FLAGS.soft4xx_streak = 0
    if not ans:
        raise APIError("пустой ответ после очистки", transient=True)
    draft = ans
    if trace:
        trace("verify")
    ctx_docs = list(comp) + [kb.docs[d] for d, _ in law_items]
    ans, bad, refs = _verify_and_repair(kb, ans, messages, user_msg, qtext, ctx_docs, level)
    ans = _trim_answer(ans, bad, refs)
    runtime.FLAGS.inc(f"level{level}")
    record = {"question": qtext, "current": prep.current, "level": level,
              "laws": [(kb.docs[d].label, "целиком" if chs is None else [c.label for c in chs]) for d, chs in law_items],
              "company": [d.label for d in comp], "date_note": dnote, "draft": draft, "unverified": bad,
              "final": ans, "prompt_chars": len(user_msg), **sel,
              "law_chunks": [c.idx for _, chs in law_items if chs for c in chs],
              "law_whole": [d for d, chs in law_items if chs is None], "user_msg": user_msg,
              "law_items_idx": [[d, None if chs is None else [c.idx for c in chs]] for d, chs in law_items]}
    if CHECK_REFS:
        record["unverified_refs"] = refs
    debug_dump(record)
    if info is not None:
        info.update(record, law_doc_idx=[d for d, _ in law_items], company_doc_idx=[d.idx for d in comp])
    return ans


def _prepare_message(kb: KB, index: Index, qtext: str, prep: Prepared, level: int, sel: dict):
    """Контекст уровня level → (законы, документы компании, справка по датам, сообщение пользователя).
    sel — сведения об отборе (диагностика и поле context веба)."""
    law_budget = {4: LAW_BUDGET_CHARS, 3: 8000, 2: 4000}.get(level, 2000)
    if REWRITE:  # R7
        maybe_rewrite(kb, index, prep, level, sel)
    law_items, _ = select_law(kb, index, prep, use_rerank=level >= 4, budget_chars=law_budget, diag=sel)
    comp = company_context(kb, index, prep, compact=level <= 2, minimal=level <= 1, diag=sel)
    date_text = user_text(prep) if CHECK_FIXES else qtext   # пакет 2: даты из реплик ассистента — не даты события
    dnote = date_note(date_text, kb)
    if DATE_CONDITIONS:  # R4: условия документов-изменений и редакции законов в контексте
        extra = date_conditions(date_text, kb, [(kb.docs[d], chs) for d, chs in law_items])
        dnote = "\n".join(x for x in (dnote, extra) if x)
    user_msg = build_user_message(kb, comp, law_items, dnote, prep.current, prep.history,
                                  xref=sel.get("law_xref"), absent=sel.get("law_absent"),
                                  standalone=prep.standalone if REWRITE and REWRITE_PROMPT else None)
    return law_items, comp, dnote, user_msg


def _on_reject(kb: KB, index: Index, qtext: str, prep: Prepared, level: int, exc: APIError, trace, info,
               retried: bool) -> str | None:
    """Отказ прокси на запрос уровня level → ответ уровнем ниже; None — ошибку пробросить (временная или ниже некуда)."""
    if SOFT_DEGRADE and not exc.transient and not _size_error(exc):
        return _soft_4xx(kb, index, qtext, prep, level, exc, trace, info, retried)
    if not exc.transient and level >= 2:
        # Прокси отклонил запрос (размер или параметры): уменьшаем запрос для всех последующих вопросов.
        with runtime.FLAGS.lock:
            if level >= 3 and not runtime.FLAGS.compact:
                runtime.FLAGS.compact = True
                log(f"Прокси отклонил полный запрос ({exc}) — переходим на компактный режим")
            elif level == 2 and not runtime.FLAGS.minimal:
                runtime.FLAGS.minimal = True
                log(f"Прокси отклонил компактный запрос ({exc}) — переходим на минимальный режим")
        return answer_one(kb, index, qtext, prep, 2 if level >= 3 else 1, trace, info)
    if not exc.transient and level == 1:
        with runtime.FLAGS.lock:
            if not runtime.FLAGS.tiny:
                runtime.FLAGS.tiny = True
                log(f"Прокси отклонил минимальный запрос ({exc}) — переходим на короткий запрос")
        return answer_tiny(kb, index, prep, trace, info)
    return None


def _verify_and_repair(kb: KB, ans: str, messages: list[dict], user_msg: str, qtext: str, ctx_docs: list,
                       level: int) -> tuple[str, list, list]:
    """Проверка чисел (и ссылок 7.1) по контексту; неподтверждённое — один запрос на исправление, если есть время.
    → (ответ, неподтверждённые числа [(фрагмент, вид)], неподтверждённые ссылки)."""
    bad = unverified(ans, user_msg, qtext)
    # 7.1 (флаг): ссылки на пункты и документы, которых нет в базе
    refs = [sp for sp, _ in unverified_refs(kb, ans, qtext, ctx_docs)] if CHECK_REFS else []
    full = runtime.TIMING.get("chat_full", 8.0)
    can_repair = level >= 4 and remaining() > 60 and per_question_budget() >= 2.2 * full
    if (bad or refs) and can_repair:
        spans = [sp for sp, _ in bad]
        if bad:
            log(f"Неподтверждённые числа {spans[:5]} — запрашиваю исправление")
            fix = (f"Черновик ответа:\n{ans}\n\nАвтоматическая проверка: числа или даты {', '.join(spans[:8])} не найдены "
                   "в документах и не получаются простым расчётом из чисел документов и вопроса. Перепиши ответ: убери или "
                   "замени их точными значениями из документов. Если число получено расчётом, приведи формулу с исходными "
                   "числами из документов. Остальные требования к ответу сохраняются. Верни только итоговый текст ответа.")
            if refs:
                fix += " " + REPAIR_REFS.format(refs=", ".join(refs[:8]))
        else:
            log(f"Неподтверждённые ссылки {refs[:5]} — запрашиваю исправление")
            fix = repair_refs_message(ans, refs)
        try:
            fixed = clean_answer(chat(messages + [{"role": "assistant", "content": ans},
                                                  {"role": "user", "content": fix}]))
            if fixed:
                ans = fixed
                runtime.FLAGS.inc("repaired")
                bad = unverified(ans, user_msg, qtext)
                if CHECK_REFS:
                    refs = [sp for sp, _ in unverified_refs(kb, ans, qtext, ctx_docs)]
        except APIError as exc:
            log(f"Исправление не удалось: {exc}")
    return ans, bad, refs


def _trim_answer(ans: str, bad: list, refs: list) -> str:
    """Удаление оставшегося неподтверждённого и обрезка после источника (B14)."""
    # Сроки и даты модель копирует из текста, а не считает: неподтверждённые — почти наверняка ошибка.
    strict = [sp for sp, kind in bad if kind in ("term", "date")]
    if strict:
        trimmed = drop_sentences(ans, strict)
        if trimmed != ans:
            runtime.FLAGS.inc("trimmed")
            log(f"Удалены предложения с неподтверждёнными сроками/датами: {strict[:5]}")
            ans = trimmed
    if refs:
        trimmed = drop_ref_sentences(ans, refs)
        if trimmed != ans:
            runtime.FLAGS.inc("ref_trimmed")
            log(f"Удалены предложения с неподтверждёнными ссылками: {refs[:5]}")
            ans = trimmed
    if ANSWER_TEMPLATE and CONCISE:   # B14 (формат подключается к промпту только вместе с B9)
        cut = trim_after_source(ans)
        if cut != ans:
            runtime.FLAGS.inc("template_trimmed")
            ans = cut
    return ans
