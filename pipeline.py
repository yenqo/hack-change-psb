"""Конвейер ответа на один вопрос: разбор диалога, подбор законов и документов, генерация,
проверка чисел и исправление, запасной ответ."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

import runtime
from config import (DEBUG, CANDIDATES, COMPANY_FULL_LIMIT, LAW_BUDGET_CHARS, LAW_TOP, SMALL_LAW_CHARS, debug_dump, log,
    remaining)
from textproc import expand_query
from dates import date_note, event_periods
from kb import doc_clauses, doc_ids, related_docs
from runtime import per_question_budget
from llm import APIError, chat, embed, rerank
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


def answer_one(kb: KB, index: Index, qtext: str, prep: Prepared, level: int, trace=None, info: dict | None = None) -> str:
    """trace(stage) — необязательный колбэк этапов (search / generate / verify) для веб-интерфейса;
    info — необязательный словарь, куда кладутся сведения об ответе (уровень, контекст, черновик, проверки).
    Ни то, ни другое на текст ответа не влияет."""
    if trace:
        trace("search")
    if runtime.FLAGS.tiny:
        return answer_tiny(kb, index, prep, trace, info)
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
    if trace:
        trace("generate")
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
            return answer_one(kb, index, qtext, prep, 2 if level >= 3 else 1, trace, info)
        if not exc.transient and level == 1:
            with runtime.FLAGS.lock:
                if not runtime.FLAGS.tiny:
                    runtime.FLAGS.tiny = True
                    log(f"Прокси отклонил минимальный запрос ({exc}) — переходим на короткий запрос")
            return answer_tiny(kb, index, prep, trace, info)
        raise
    runtime.TIMING.add({4: "chat_full", 3: "chat_full", 2: "chat_compact"}.get(level, "chat_min"), time.monotonic() - t0)
    if not ans:
        raise APIError("пустой ответ после очистки", transient=True)
    draft = ans
    if trace:
        trace("verify")
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
    record = {"question": qtext, "current": prep.current, "level": level,
              "laws": [(kb.docs[d].label, "целиком" if chs is None else [c.label for c in chs]) for d, chs in law_items],
              "company": [d.label for d in comp], "date_note": dnote, "draft": draft, "unverified": bad,
              "final": ans, "prompt_chars": len(user_msg)}
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

    def to_dict(self) -> dict:
        return {"contract_version": self.contract_version, "status": self.status, "text": self.text,
                "citations": self.citations, "warnings": self.warnings, "timing_ms": self.timing_ms,
                "debug": self.debug}


# Упоминания документов компании в тексте ответа: (шаблон, признак метки документа в базе)
_COMPANY_MENTIONS = [
    (r"доп\w*\.?\s+соглашени\w*\s*(?:№\s*)?1\b", "Доп соглашение 1"),
    (r"ПТ-05(?:/2023)?|Альфа-Трейдинг", "ПТ-05-2023"),
    (r"А-08(?:/2024)?|Северн\w*\s+Склад", "А-08-2024"),
    (r"П-17(?:/2024)?|Альфа-Трейд(?!инг)", "П-17-2024"),
    (r"К-12\b|комплаенс", "К-12"),
    (r"ЮО-7\b|служебн\w*\s+записк", "Служебная записка"),
    (r"Регламент\w*", "Регламент"),
]
_LAW_MENTION = re.compile(r"(ТК|НК|КоАП)\s*РФ\W{0,3}(?:ст\.|стать\w*)\s*(\d+(?:\.\d+)?)|"
                          r"(?:ст\.|стать\w*)\s*(\d+(?:\.\d+)?)\s*(ТК|НК|КоАП)\s*РФ|"
                          r"(\d+-ФЗ)\W{0,3}(?:ст\.|стать\w*)\s*(\d+)|(?:ст\.|стать\w*)\s*(\d+)(?:,?\s*ч\.\s*\d+)?\s*"
                          r"(?:Федеральн\w+\s+закона\s+)?(?:№\s*)?(\d+-ФЗ)", re.I)
_CLAUSE_REF = re.compile(r"(?<![\w.])(пп?|ч|подп)\.\s*(\d+(?:\.\d+)*(?:\s*(?:,|и|–|-|—)\s*\d+(?:\.\d+)*)*)", re.I)


def _expand_nums(raw: str) -> list[str]:
    out = []
    for part in re.split(r"\s*(?:,|\bи\b)\s*", raw):
        m = re.fullmatch(r"(\d+(?:\.\d+)*)\s*[–—-]\s*(\d+(?:\.\d+)*)", part.strip())
        if m:
            a, b = m.group(1).split("."), m.group(2).split(".")
            if len(a) == len(b) and a[:-1] == b[:-1] and int(a[-1]) <= int(b[-1]) <= int(a[-1]) + 20:
                out += [".".join(a[:-1] + [str(k)]) for k in range(int(a[-1]), int(b[-1]) + 1)]
            else:
                out += [m.group(1), m.group(2)]
        elif re.fullmatch(r"\d+(?:\.\d+)*", part.strip()):
            out.append(part.strip())
    return out


def _mentions(kb: KB, text: str, qtext: str) -> list[tuple[int, int, Doc]]:
    """Упоминания документов в ответе: (начало, конец, документ). Общие слова «договор», «Регламент» без
    уточнения относятся к последнему упомянутому договору / редакции регламента, иначе — по дате события."""
    raw: list[tuple[int, int, str, str | None]] = []   # (начало, конец, ключ, уточнение)
    for pat, key in _COMPANY_MENTIONS:
        for m in re.finditer(pat, text, re.I):
            qual = None
            if key == "Регламент":
                after = text[m.end(): m.end() + 70]
                if re.search(r"^\W*(?:\w+\s+){0,4}?(?:ред\w*\.?\s*(?:от\s*)?10\.01\.2023|2023|утрат)", after, re.I):
                    qual = "2023"
                elif re.search(r"^\W*(?:\w+\s+){0,4}?(?:ред\w*\.?\s*(?:от\s*)?01\.03\.2025|2025|действующ)", after, re.I):
                    qual = "2025"
            raw.append((m.start(), m.end(), key, qual))
    for m in re.finditer(r"(?<![\w-])договор\w*(?!\s+(?:поставки|аренды)?\s*№)", text, re.I):
        raw.append((m.start(), m.end(), "договор", None))
    raw.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    found: list[tuple[int, int, Doc]] = []
    periods = event_periods(qtext)
    last_contract = last_reg = None
    for s, e, key, qual in raw:
        if any(a <= s < b for a, b, _ in found):
            continue
        doc = None
        if key == "Регламент":
            regs = [d for d in kb.company if "Регламент" in d.label]
            if qual:
                doc = next((d for d in regs if qual in d.label), None)
            elif last_reg is not None:
                doc = last_reg
            else:
                fit = [d for d in regs if _applies(d, periods)]
                doc = fit[0] if len(fit) == 1 else next((d for d in regs if "2025" in d.label), None)
            last_reg = doc or last_reg
        elif key == "договор":
            doc = last_contract
        else:
            doc = next((d for d in kb.company if key in d.label), None)
            if doc is not None and doc_type_is_contract(doc):
                last_contract = doc
        if doc is not None:
            found.append((s, e, doc))
    for m in _LAW_MENTION.finditer(text):
        g = m.groups()
        code, art = (g[0], g[1]) if g[0] else (g[3], g[2]) if g[3] else (g[4], g[5]) if g[4] else (g[7], g[6])
        code = code.upper().replace("КОАП", "КоАП")
        for d in kb.laws:
            if (code.endswith("ФЗ") and code in d.label and d.label.endswith(f"ст. {art}")) or \
                    d.label == f"{code} РФ, ст. {art}":
                found.append((m.start(), m.end(), d))
                break
    return sorted(found, key=lambda x: x[0])


def doc_type_is_contract(d: Doc) -> bool:
    return d.kind == "company" and d.title.upper().startswith("ДОГОВОР")


def _clause_key(d: Doc, kind: str, num: str, prev_point: str | None) -> str:
    if d.kind == "company":
        return num
    word = "п." if d.label.startswith("НК") else "ч."
    if kind.lower() in ("пп", "подп") and prev_point:
        return f"{word} {prev_point}, пп. {num}"
    return f"{word} {num}"


def _quote_span(d: Doc, start: int, end: int, limit: int = 600) -> tuple[int, int]:
    if end - start > limit:
        cut = d.text.rfind(" ", start, start + limit)
        end = cut if cut > start + limit // 2 else start + limit
    return start, end


def _applies(d: Doc, periods: list) -> bool | None:
    if not periods or not (d.valid_from or d.valid_to):
        return None
    vf = d.valid_from or date(1900, 1, 1)
    vt = d.valid_to or date(2999, 12, 31)
    return all(vf <= end and start <= vt for start, end, _ in periods)


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


_KB_CLAUSES: dict[int, list] = {}


def kb_clauses(d: Doc) -> list[dict]:
    if id(d) not in _KB_CLAUSES:
        _KB_CLAUSES[id(d)] = doc_clauses(d)
    return _KB_CLAUSES[id(d)]


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
    if status is None:
        status = classify_status(text, citations)
        warns = build_warnings(kb, qtext, info, citations, ids)
    order = ["search", "generate", "verify", "done"]
    seen = [n for n in order if n in marks]
    timing = {}
    for a, b in zip(seen, seen[1:]):
        timing[a] = round((marks[b] - marks[a]) * 1000)
    timing["total"] = round((marks["done"] - t0) * 1000)
    dbg = {}
    if debug or DEBUG:
        dbg = {k: info.get(k) for k in ("level", "laws", "company", "date_note", "draft", "unverified", "prompt_chars")}
    return Result(text=text, status=status, citations=citations, warnings=warns, timing_ms=timing, debug=dbg)
