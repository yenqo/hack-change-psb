"""Конвейер ответа на один вопрос: разбор диалога, подбор законов и документов, генерация,
проверка чисел и исправление, запасной ответ."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

import runtime
from config import (CHECK_REFS, CROSS_REFS, DATE_CONDITIONS, GENERAL_POINTS, MAX_WORKERS, MODE, REWRITE, REWRITE_BM25, REWRITE_COS, DEBUG, SOFT_DEGRADE, CANDIDATES, COMPANY_FULL_LIMIT, LAW_BUDGET_CHARS, LAW_TOP, SMALL_LAW_CHARS, debug_dump, log,
    remaining)
from textproc import STEMMER, STOPWORDS, WORD_RE, expand_query, norm
from dates import date_conditions, date_note, event_periods
from kb import cross_refs, doc_clauses, doc_ids, doc_markup, related_docs
from runtime import per_question_budget
from llm import APIError, chat, embed, rerank
from prompts import (REPAIR_REFS, SYSTEM_REWRITE, SYSTEM_TEXT, SYSTEM_TEXT_MINI, SYSTEM_TEXT_SHORT, SYSTEM_TINY, build_user_message, rewrite_message,
                     repair_refs_message)
from checks import (_CLAUSE_REF, _applies, _clause_key, _expand_nums, _mentions, _quote_span, clean_answer,
                    drop_ref_sentences, drop_sentences, kb_clauses, unverified, unverified_refs)

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
    # R7: переформулирование (один раз на вопрос; финальный проход B8 берёт готовое)
    rewrite_done: bool = False
    standalone: str | None = None
    extra: list | None = None      # [(поисковый запрос, его эмбеддинг или None)]


def _search(index: Index, query: str, prep: Prepared, kinds: set[str]) -> list[int]:
    if prep.extra:
        return index.hybrid_multi([(query, prep.qvec)] + prep.extra, kinds)
    return index.hybrid(query, prep.qvec, kinds)


def weak_search(kb: KB, index: Index, prep: Prepared) -> bool:
    """R7: лучший результат поиска слабый и по законам, и по документам компании (вопрос «не узнан» поиском)."""
    if prep.qvec is not None and index.vecs is not None:
        best = max((sc for _, sc in index.dense(prep.qvec)[:1]), default=0.0)
        return best < REWRITE_COS
    query = prep.current + " " + " ".join(t for r, t in prep.history if r == "user")
    best = max((sc for _, sc in index.bm25(expand_query(query))[:1]), default=0.0)
    return best < REWRITE_BM25


def _parse_rewrite(text: str) -> tuple[str, list[str]] | None:
    import json
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    q = data.get("question") if isinstance(data, dict) else None
    qs = data.get("queries") if isinstance(data, dict) else None
    if not isinstance(q, str) or not q.strip():
        return None
    qs = [x.strip()[:300] for x in qs if isinstance(x, str) and x.strip()][:3] if isinstance(qs, list) else []
    return q.strip()[:600], qs


def maybe_rewrite(kb: KB, index: Index, prep: Prepared, level: int, info: dict | None = None) -> None:
    """R7: вызов LLM только когда он вероятно нужен — есть история диалога или поиск слабый. Вход — история и вопрос,
    без документов. При нехватке времени (уровень ниже полного, мало времени на вопрос) пропускается первым."""
    if prep.rewrite_done:
        return
    f = runtime.FLAGS
    if level < 4 or f.compact or f.minimal or f.tiny:
        return
    if not prep.history and not weak_search(kb, index, prep):
        return
    # пропускается первым: прогноз B7 для полного уровня должен сходиться и с лишним вызовом на каждый оставшийся вопрос
    with runtime.PROG.lock:
        left = max(1, runtime.PROG.n - runtime.PROG.finished)
    extra = runtime.TIMING.get("rewrite", runtime.TIMING.get("chat_full", 8.0))
    need = left * (runtime.question_cost(4) + extra) / max(1, MAX_WORKERS) * 1.2
    measured = MODE == "web" or "chat_full" in runtime.TIMING.ema   # до первых замеров прогноз — по умолчаниям
    if not measured or remaining() < 120 or need > remaining() - 15:
        f.inc("rewrite_skipped")
        return
    prep.rewrite_done = True
    t0 = time.monotonic()
    try:
        raw = chat([{"role": "system", "content": SYSTEM_REWRITE},
                    {"role": "user", "content": rewrite_message(prep.current, prep.history)}],
                   max_tokens=300, timeout=30.0)
    except APIError as exc:
        f.inc("rewrite_failed")
        log(f"Переформулирование не удалось: {exc}")
        return
    runtime.TIMING.add("rewrite", time.monotonic() - t0)
    parsed = _parse_rewrite(raw)
    if not parsed:
        f.inc("rewrite_failed")
        return
    standalone, queries = parsed
    queries = [standalone] + [q for q in queries if q != standalone]
    vecs: list = [None] * len(queries)
    if index.vecs is not None and prep.qvec is not None:
        try:
            vecs = embed(queries)
        except APIError:
            vecs = [None] * len(queries)
    prep.standalone = standalone
    prep.extra = list(zip(queries, vecs))
    f.inc("rewrite")
    if info is not None:
        info["rewrite"] = {"standalone": standalone, "queries": queries[1:], "sec": round(time.monotonic() - t0, 2)}


def select_law(kb: KB, index: Index, prep: Prepared, use_rerank: bool, budget_chars: int = LAW_BUDGET_CHARS,
               diag: dict | None = None):
    """diag — необязательный словарь для диагностики: ранжирование законов до и после реранкера (на отбор не влияет)."""
    query = prep.current
    if prep.history:
        query += " " + " ".join(t for r, t in prep.history if r == "user")
    ranked = _search(index, query, prep, {"law"})[:CANDIDATES]
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
    # B12: урезанный бюджет (вопрос о документах компании) не распространяется на общие пункты больших статей
    gp_room = (budget_chars - budget) if GENERAL_POINTS else 0
    order = ranked
    if use_rerank:
        rr = rerank(prep.standalone or (prep.current if not prep.history else query[:2000]),
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
    reasons: dict[int, str] = {}   # фрагмент → почему попал в контекст (для поля context веб-части)
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
            reasons[c.idx] = "search"
    # Для больших статей из топ-3 добавляем короткие пункты верхнего уровня: там обычно общие правила
    # (например, общая ставка налога), которые лексически плохо находятся по вопросу.
    seen_docs = []
    for ci in order[:3]:
        d = kb.chunks[ci].doc
        if d in seen_docs or (d in items and items[d] is None):
            continue
        seen_docs.append(d)
        extra = 0
        room = gp_room if gp_room and _topic_hit(kb.docs[d], query) else 0   # B12: только статья по теме вопроса
        for c in kb.chunks:
            if c.doc != d or c.label.count(",") != 2 or len(c.text) > 700 or "(продолжение" in c.text:
                continue
            if c in (items.get(d) or []) or len(c.text) > budget + room or extra + len(c.text) > 2500:
                continue
            items.setdefault(d, [])
            items[d].append(c)  # type: ignore[union-attr]
            if len(c.text) > budget:   # B12: недостающее — из запаса урезанного бюджета
                room -= len(c.text) - budget
                gp_room -= len(c.text) - budget
                budget = 0
            else:
                budget -= len(c.text)
            extra += len(c.text)
            reasons[c.idx] = "general_point"
    xref: dict[int, str] = {}
    absent: list[tuple[str, str]] = []
    if CROSS_REFS:  # R6: пункты, на которые ссылаются найденные, — на 1 шаг, не больше 3, в пределах бюджета
        budget = _add_cross_refs(kb, order, items, reasons, budget, xref, absent)
    # порядок статей — по лучшему рангу
    rank_of = {}
    for r, ci in enumerate(order):
        rank_of.setdefault(kb.chunks[ci].doc, r)
    law_items = sorted(items.items(), key=lambda x: rank_of.get(x[0], 999))
    if diag is not None:
        diag.update(law_hybrid=list(ranked[:40]), law_order=list(order[:40]), rerank_used=order is not ranked,
                    law_reason=reasons)
        if CROSS_REFS:
            diag.update(law_xref=xref, law_absent=absent)
    return law_items, order


_TOPIC_GENERIC = {"налог", "налогов", "ставк", "глав", "кодекс", "стат", "закон", "федеральн", "российск", "федерац", "организац"}


def _topic_hit(d: Doc, query: str) -> bool:
    """B12: вопрос про тему статьи — есть общие основы слов с её названием (с темой главы НК), с учётом синонимов."""
    title = norm((d.search_extra or d.title).split("«", 1)[-1])
    stems = {STEMMER.stem(w) for w in WORD_RE.findall(title) if not w.isdigit() and len(w) > 2 and w not in STOPWORDS}
    q = {STEMMER.stem(w) for w in WORD_RE.findall(norm(expand_query(query))) if len(w) > 2 and w not in STOPWORDS}
    return bool((stems - _TOPIC_GENERIC) & q)


XREF_MAX = 3          # фрагментов, добавленных по отсылкам
XREF_ABSENT_MAX = 5  # статей «в базе отсутствует» в контексте
XREF_SOURCES = 5     # отсылки разбираются только у лучших по рангу фрагментов в контексте


def _add_cross_refs(kb: KB, order: list[int], items: dict, reasons: dict, budget: int, xref: dict, absent: list) -> int:
    """Источники — найденные поиском фрагменты в контексте (у статей целиком — их пункты из выдачи поиска), по рангу.
    xref: добавленный фрагмент → пункт-источник (для пометки «[упомянут в …]»); absent: (акт, статья), которых нет в базе."""
    refs = cross_refs(kb)
    in_ctx = {c.idx for chs in items.values() if chs for c in chs}
    sources = [ci for ci in order[:LAW_TOP] if kb.chunks[ci].doc in items and
               (items[kb.chunks[ci].doc] is None or ci in in_ctx)][:XREF_SOURCES]
    added = 0
    for ci in sources:
        src = kb.chunks[ci]
        for ref in refs.get(ci, []):
            for a in ref["absent"] if ci == order[0] else []:   # «нет в базе» — только у лучшего фрагмента: меньше шума
                if a not in absent and len(absent) < XREF_ABSENT_MAX:
                    absent.append(a)
            for t in ref["targets"]:
                tc = kb.chunks[t]
                if added >= XREF_MAX or t in in_ctx or len(tc.text) > budget:
                    continue
                if tc.doc in items and items[tc.doc] is None:   # статья и так целиком
                    continue
                items.setdefault(tc.doc, []).append(tc)
                in_ctx.add(t)
                budget -= len(tc.text)
                reasons[t] = "cross_ref"
                base = kb.docs[src.doc].label
                xref[t] = src.label[len(base) + 2:] if tc.doc == src.doc and src.label.startswith(base + ", ") else src.label
                added += 1
    return budget


def company_context(kb: KB, index: Index, prep: Prepared, compact: bool = False, minimal: bool = False,
                    diag: dict | None = None) -> list[Doc]:
    docs = kb.company
    if not compact and not minimal and sum(len(d.text) for d in docs) <= COMPANY_FULL_LIMIT:
        if diag is not None:
            diag["company_reason"] = {d.idx: "company_full" for d in docs}
        return docs
    limit = 5000 if minimal else (10000 if compact else COMPANY_FULL_LIMIT)
    max_docs = 2 if minimal else (4 if compact else 8)
    query = prep.current + " " + " ".join(t for _, t in prep.history)
    ranked = _search(index, query, prep, {"company"})
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
            if diag is not None:
                diag.setdefault("company_reason", {})[g.idx] = "search" if g is d else "related"
        if len(chosen) >= max_docs:
            break
    return [d for d in docs if d in chosen]  # исходный порядок документов


def answer_tiny(kb: KB, index: Index, prep: Prepared, trace=None, info: dict | None = None) -> str:
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
    law_budget = {4: LAW_BUDGET_CHARS, 3: 8000, 2: 4000}.get(level, 2000)
    sel: dict = {}
    if REWRITE:  # R7
        maybe_rewrite(kb, index, prep, level, sel)
    law_items, _ = select_law(kb, index, prep, use_rerank=level >= 4, budget_chars=law_budget, diag=sel)
    comp = company_context(kb, index, prep, compact=level <= 2, minimal=level <= 1, diag=sel)
    dnote = date_note(qtext, kb)
    if DATE_CONDITIONS:  # R4: условия документов-изменений и редакции законов в контексте
        extra = date_conditions(qtext, kb, [(kb.docs[d], chs) for d, chs in law_items])
        dnote = "\n".join(x for x in (dnote, extra) if x)
    user_msg = build_user_message(kb, comp, law_items, dnote, prep.current, prep.history,
                                  xref=sel.get("law_xref"), absent=sel.get("law_absent"),
                                  standalone=prep.standalone if REWRITE else None)
    system = SYSTEM_TEXT if level >= 3 else (SYSTEM_TEXT_SHORT if level == 2 else SYSTEM_TEXT_MINI)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
    if trace:
        trace("generate")
    t0 = time.monotonic()
    try:
        ans = clean_answer(chat(messages))
    except APIError as exc:
        if SOFT_DEGRADE and not exc.transient and not _size_error(exc):
            return _soft_4xx(kb, index, qtext, prep, level, exc, trace, info, _retry)
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
        raise
    runtime.TIMING.add({4: "chat_full", 3: "chat_full", 2: "chat_compact"}.get(level, "chat_min"), time.monotonic() - t0)
    if SOFT_DEGRADE and level >= 3:
        with runtime.FLAGS.lock:
            runtime.FLAGS.soft4xx_streak = 0
    if not ans:
        raise APIError("пустой ответ после очистки", transient=True)
    draft = ans
    if trace:
        trace("verify")
    bad = unverified(ans, user_msg, qtext)
    # 7.1 (флаг): ссылки на пункты и документы, которых нет в базе
    ctx_docs = list(comp) + [kb.docs[d] for d, _ in law_items]
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


# ============================================================================
# Структурированный результат (доработка 7.4): статус и цитаты по готовому тексту ответа.
# Текст ответа для submission.csv не меняется.
# ============================================================================

STATUSES = ("answer", "partial", "no_info", "premise_error", "clarify", "out_of_scope", "error")


@dataclass
class Result:
    text: str
    status: str = "answer"
    citations: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    timing_ms: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)
    contract_version: str = ""
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"contract_version": self.contract_version, "status": self.status, "text": self.text,
                "citations": self.citations, "warnings": self.warnings, "context": self.context,
                "timing_ms": self.timing_ms, "debug": self.debug}


def extract_citations(kb: KB, text: str, qtext: str, ids: dict[int, str]) -> list[dict]:
    """Ссылки ответа «документ + пункт» → цитаты со смещениями в тексте документа (get_document)."""
    mentions = _mentions(kb, text, qtext)
    periods = event_periods(qtext)
    cites: dict[tuple[int, str | None], dict] = {}

    def add(d: Doc, clause: str | None) -> None:
        key = (d.idx, clause)
        if key in cites:
            return
        found = True
        if clause is None:
            first = d.text.split("\n", 1)[0]
            s, e = 0, len(first)
            if d.kind == "law":
                m = re.search(r"(?m)^Статья\s+\d.*$", d.text)
                if m:
                    s, e = m.start(), m.end()
        else:
            hit = next((c for c in kb_clauses(d) if c["clause"] == clause), None)
            if hit is None:
                found = False
                s, e = 0, len(d.text.split("\n", 1)[0])
            else:
                s, e = _quote_span(d, hit["start"], hit["end"])
        cites[key] = {"doc_id": ids[d.idx], "doc_title": d.title, "doc_kind": d.kind, "clause": clause,
                      "quote": d.text[s:e], "start": s, "end": e, "found": found,
                      "applies_to_event_date": _applies(d, periods)}

    for s, e, d in mentions:
        add(d, None)
    last_point: dict[int, str] = {}
    for m in _CLAUSE_REF.finditer(text):
        before = [x for x in mentions if x[0] <= m.start()]
        after = [x for x in mentions if x[0] >= m.end() and x[0] - m.end() < 60]
        if after and re.fullmatch(r"\s{0,3}", text[m.end():after[0][0]]):
            target = after[0][2]           # «п. 2.4 Регламента», «п. 2.2 договора» — документ сразу после ссылки
        elif before and text.rfind("(", 0, m.start()) > before[-1][1] and after:
            target = after[0][2]           # «(п. 3.1, договор № …)» — документ после ссылки
        elif before:
            target = before[-1][2]
        elif after:
            target = after[0][2]
        else:
            continue
        others = [x[2] for x in reversed(before)] + [x[2] for x in after]
        for num in _expand_nums(m.group(2)):
            doc = target
            key = _clause_key(doc, m.group(1), num, last_point.get(doc.idx))
            if not any(c["clause"] == key for c in kb_clauses(doc)):
                # в ближайшем документе такого пункта нет — берём ближайший упомянутый документ, где он есть
                for o in others:
                    k2 = _clause_key(o, m.group(1), num, last_point.get(o.idx))
                    if any(c["clause"] == k2 for c in kb_clauses(o)):
                        doc, key = o, k2
                        break
            if doc.kind == "law" and m.group(1).lower() not in ("пп", "подп"):
                last_point[doc.idx] = num
            add(doc, key)
    out = list(cites.values())
    # документ без пункта не нужен, если на него есть ссылка с пунктом
    with_clause = {c["doc_id"] for c in out if c["clause"]}
    return [c for c in out if c["clause"] or c["doc_id"] not in with_clause]


_NO_INFO = re.compile(r"(?i)в базе( знаний)? нет|нет документов|не найден\w* (документ|сведени)|"
                      r"ответить по документам нельзя|не могу ответить по документам")
_SCOPE = re.compile(r"(?i)юридическ\w+ ассистент|не относится к (юридическ|теме)|вне (моей )?компетенц")
_CLARIFY = re.compile(r"(?i)\bуточните\b|просьба уточнить|нужно уточнить, (о каком|какой)|какой именно")
_PREMISE = re.compile(r"(?i)\bа не\s+(?:[«\"]?\d)|не соответствует|неверн|ошибочн|не подтвержда|предпосылк")
_PARTIAL = re.compile(r"(?i)(в базе|в документах|в договоре|в предоставленной базе)[^.]{0,60}(нет|отсутству|не указ|не установ)|"
                      r"не хватает|частичн|отсутствуют в базе")


def classify_status(text: str, citations: list[dict]) -> str:
    has_cite = any(c["found"] for c in citations)
    if not has_cite and _SCOPE.search(text) and not _NO_INFO.search(text):
        return "out_of_scope"
    if not has_cite and _NO_INFO.search(text):
        return "no_info"
    if _CLARIFY.search(text):
        return "clarify"
    if _PREMISE.search(text) and has_cite:
        return "premise_error"
    if has_cite and (_PARTIAL.search(text) or _NO_INFO.search(text)):
        return "partial"
    if not has_cite and _SCOPE.search(text):
        return "out_of_scope"
    return "answer"


def build_warnings(kb: KB, qtext: str, info: dict, citations: list[dict], ids: dict[int, str]) -> list[dict]:
    out = []
    if info.get("date_note"):
        out.append({"code": "edition_note", "message": info["date_note"]})
    by_id = {v: k for k, v in ids.items()}
    cited = {by_id[c["doc_id"]] for c in citations}
    for di in sorted(cited):  # документы компании всегда в контексте целиком — предупреждаем только о процитированных
        d = kb.docs[di]
        if d.removed:
            out.append({"code": "removed_instructions", "doc_id": ids[di],
                        "message": f"В документе «{d.label}» удалена встроенная инструкция для ИИ: она не является "
                                   "положением документа и не учитывалась"})
    if event_periods(qtext):
        for di in sorted(cited):
            d = kb.docs[di]
            if d.kind == "law":
                out.append({"code": "law_current_edition_only", "doc_id": ids[di],
                            "message": f"В базе только действующая редакция ({d.label}); на дату события могла "
                                       "действовать иная"})
    amounts = [sp for sp, kind in info.get("unverified", []) if kind == "amount"]
    if amounts:
        out.append({"code": "unverified_numbers", "message": "Числа не найдены в документах: " + ", ".join(amounts[:8])})
    missing = [c for c in citations if not c["found"]]
    for c in missing:
        out.append({"code": "clause_not_found", "doc_id": c["doc_id"],
                    "message": f"Пункт {c['clause']} не найден в документе"})
    return out


def _clause_of(kb: KB, c: Chunk) -> str | None:
    lab, base = c.label, kb.docs[c.doc].label
    return lab[len(base) + 2:] if lab.startswith(base + ", ") else None


def _frag(kb: KB, c: Chunk, rank: int | None, reason: str) -> dict:
    return {"clause": _clause_of(kb, c), "start": c.start, "end": c.end, "rank": rank, "reason": reason,
            "edition_note": "[Пометка редакции:" in c.text}


def build_context(kb: KB, index: Index, info: dict, prep: Prepared, qtext: str, ids: dict[int, str],
                  citations: list[dict]) -> dict:
    """Что ушло в контекст модели — из тех же объектов, из которых строилось сообщение (info из answer_one).
    Только идентификаторы и смещения в тексте get_document; тексты не передаются."""
    periods = event_periods(qtext)
    cited = {c["doc_id"] for c in citations}
    order = info.get("law_order") or []
    rank = {ci: i + 1 for i, ci in enumerate(order)}
    reasons = {int(k): v for k, v in (info.get("law_reason") or {}).items()}
    out: list[dict] = []

    def doc_entry(d: Doc, included: str, reason: str) -> dict:
        return {"doc_id": ids[d.idx], "doc_title": d.title, "doc_kind": d.kind, "included": included, "reason": reason,
                "applies_to_event_date": _applies(d, periods), "cited": ids[d.idx] in cited}

    if info.get("level") == 0:   # короткий запрос: только найденные фрагменты
        by_doc: dict[int, list] = {}
        for i, ci in enumerate(info.get("chunks") or []):
            by_doc.setdefault(kb.chunks[ci].doc, []).append(_frag(kb, kb.chunks[ci], i + 1, "search"))
        for di, frs in by_doc.items():
            out.append({**doc_entry(kb.docs[di], "fragments", "search"), "fragments": frs})
    else:
        # документы компании: целиком; совпадения поиска — для подсветки
        q = prep.current + " " + " ".join(t for _, t in prep.history)
        hits: dict[int, list] = {}
        for r, ci in enumerate(index.hybrid(q, prep.qvec, {"company"})[:10], 1):
            c = kb.chunks[ci]
            hits.setdefault(c.doc, []).append({"clause": _clause_of(kb, c), "start": c.start, "end": c.end, "rank": r})
        creason = {int(k): v for k, v in (info.get("company_reason") or {}).items()}
        for di in info.get("company_doc_idx") or []:
            out.append({**doc_entry(kb.docs[di], "whole", creason.get(di, "company_full")),
                        "search_hits": hits.get(di, [])})
        # законы: статьи целиком или найденные фрагменты
        for di, chs in info.get("law_items_idx") or []:
            d = kb.docs[di]
            if chs is None:
                sh = [{"clause": _clause_of(kb, kb.chunks[ci]), "start": kb.chunks[ci].start,
                       "end": kb.chunks[ci].end, "rank": rank[ci]} for ci in order if kb.chunks[ci].doc == di]
                out.append({**doc_entry(d, "whole", "short_article"), "search_hits": sh})
            else:
                frs = [_frag(kb, kb.chunks[ci], rank.get(ci), reasons.get(ci, "search")) for ci in chs]
                main = next((r for r in ("search", "general_point", "cross_ref") if any(f["reason"] == r for f in frs)),
                            "search")
                out.append({**doc_entry(d, "fragments", main), "fragments": frs})
    removed = []
    for e in out:
        d = kb.docs[next(k for k, v in ids.items() if v == e["doc_id"])]
        for r in doc_markup(d)["removed_instructions"]:
            removed.append({"doc_id": e["doc_id"], "start": r["start"], "end": r["end"], "note": r["note"]})
    return {"documents": out, "date_note": info.get("date_note") or "", "removed_from_context": removed}


def answer(kb: KB, index: Index, qtext: str, level: int = 4, trace=None, debug: bool = False,
           ids: dict[int, str] | None = None) -> Result:
    """Один вопрос (текст в формате платформы, с историей диалога) → Result. Тот же конвейер answer_one, что и в
    пакетном режиме; исключений наружу не бросает: при сбое — status=error и запасной ответ."""
    ids = ids or doc_ids(kb)
    marks: dict[str, float] = {}
    t0 = time.monotonic()

    def stage(name: str) -> None:
        marks[name] = time.monotonic()
        if trace:
            try:
                trace(name)
            except Exception:  # noqa: BLE001 — колбэк интерфейса не должен ломать ответ
                pass

    prep = Prepared(*parse_dialog(qtext))
    info: dict = {}
    try:
        if index.vecs is not None:
            q = (prep.current + " " + " ".join(t for r, t in prep.history if r == "user"))[:2000]
            try:
                prep.qvec = embed([q])[0]
            except APIError:
                prep.qvec = None
        text = answer_one(kb, index, qtext, prep, level, stage, info)
        status = None
        warns: list[dict] = []
    except Exception as exc:  # noqa: BLE001
        text = fallback_answer(kb, index, qtext)
        status = "error"
        code = "llm_unavailable" if isinstance(exc, APIError) else "internal_error"
        warns = [{"code": code, "message": "Модель не ответила; показан ближайший фрагмент базы знаний"},
                 {"code": "fallback", "message": type(exc).__name__}]
    stage("done")
    citations = extract_citations(kb, text, qtext, ids) if status != "error" else []
    context: dict = {}
    if status is None:
        status = classify_status(text, citations)
        warns = build_warnings(kb, qtext, info, citations, ids)
        try:
            context = build_context(kb, index, info, prep, qtext, ids, citations)
        except Exception:  # noqa: BLE001 — описание контекста не должно ломать ответ
            context = {}
    order = ["search", "generate", "verify", "done"]
    seen = [n for n in order if n in marks]
    timing = {}
    for a, b in zip(seen, seen[1:]):
        timing[a] = round((marks[b] - marks[a]) * 1000)
    timing["total"] = round((marks["done"] - t0) * 1000)
    dbg = {}
    if debug or DEBUG:
        dbg = {k: info.get(k) for k in ("level", "laws", "company", "date_note", "draft", "unverified", "prompt_chars")}
    return Result(text=text, status=status, citations=citations, warnings=warns, timing_ms=timing, debug=dbg,
                  context=context)
