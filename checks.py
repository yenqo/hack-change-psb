"""Проверки ответа: числа и даты против контекста (в т. ч. расчётные), удаление предложений, очистка."""

from __future__ import annotations

import re
from datetime import date
from typing import TYPE_CHECKING

from config import ANSWER_MAX_CHARS, CHECK_FIXES, FIX_DOCNUM
from textproc import norm
from dates import DATE_NUM, MONTH_WORD, event_periods, month_no
from kb import doc_clauses
from llm import THINK_RE

if TYPE_CHECKING:
    from kb import KB, Doc


NUM_WORDS: dict[str, tuple[str, float]] = {}


def _reg(kind: str, value: float, forms: str) -> None:
    for f in forms.split():
        NUM_WORDS[f] = (kind, value)


for _v, _f in [(1, "один одна одно одного одной одному одним одних одну"), (2, "два две двух двум двумя"),
               (3, "три трех трем тремя"), (4, "четыре четырех четырем четырьмя"), (5, "пять пяти пятью"),
               (6, "шесть шести шестью"), (7, "семь семи семью"), (8, "восемь восьми восемью"),
               (9, "девять девяти девятью")]:
    _reg("u", _v, _f)
for _v, _f in [(10, "десять десяти десятью"), (11, "одиннадцать одиннадцати"), (12, "двенадцать двенадцати"),
               (13, "тринадцать тринадцати"), (14, "четырнадцать четырнадцати"), (15, "пятнадцать пятнадцати"),
               (16, "шестнадцать шестнадцати"), (17, "семнадцать семнадцати"), (18, "восемнадцать восемнадцати"),
               (19, "девятнадцать девятнадцати"), (20, "двадцать двадцати"), (30, "тридцать тридцати"),
               (40, "сорок сорока"), (50, "пятьдесят пятидесяти"), (60, "шестьдесят шестидесяти"),
               (70, "семьдесят семидесяти"), (80, "восемьдесят восьмидесяти"), (90, "девяносто девяноста")]:
    _reg("t", _v, _f)
for _v, _f in [(100, "сто ста"), (200, "двести двухсот"), (300, "триста трехсот"), (400, "четыреста четырехсот"),
               (500, "пятьсот пятисот"), (600, "шестьсот шестисот"), (700, "семьсот семисот"),
               (800, "восемьсот восьмисот"), (900, "девятьсот девятисот")]:
    _reg("h", _v, _f)
_reg("s", 1e3, "тысяча тысячи тысяч тысячу тысячей тысячам тысячами")
_reg("s", 1e6, "миллион миллиона миллионов миллионе миллионом миллионам")
_reg("s", 1e9, "миллиард миллиарда миллиардов")
_reg("u", 1.5, "полтора полутора")

DATE_TEXT = re.compile(r"(?<!\d)(\d{1,2})\s+" + MONTH_WORD + r"\s+(\d{4})", re.I)
REF_RE = re.compile(
    r"(?:(?<![а-яё])(?:пп|п|ч|ст|абз|подп|гл|разд)\.\s*|(?<![а-яё])(?:пункт|подпункт|част|стать|абзац|глав|раздел)\w*\s+)"
    r"\d+(?:[.\-]\d+)*(?:\s*(?:,|и|–|-|—)\s*\d+(?:[.\-]\d+)*)*", re.I)
DOCNUM_RE = re.compile(
    r"(?:№|\bN)\s*[\w\-/.]*\d[\w\-/]*|\b[A-ZА-ЯЁ]{1,4}-\d+(?:[/-]\d+)?\b|\b\d+-(?:ФЗ|П|ПП|ОД)\b|ИНН\s*\d+")
# B3: extract_numbers работает по тексту в нижнем регистре, а DOCNUM_RE ищет заглавные буквы — номера вида
# «П-17/2024» не вырезались и «17» проверялось как число. Исправление — за флагом SOLUTION_FIX_DOCNUM.
DOCNUM_RE_CI = re.compile(DOCNUM_RE.pattern, re.I)
_DOCNUM = DOCNUM_RE_CI if FIX_DOCNUM else DOCNUM_RE
LISTMARK_RE = re.compile(r"(?:(?<=\s)|^)\d{1,2}\)\s")
NUM_RE = re.compile(r"(?<![\w.,])(\d{1,3}(?:[  ]\d{3})+|\d+)(?:[.,](\d+))?(?![\d])"
                    r"(?:\s*(тыс\w*\.?|млн\.?|миллион\w*|млрд\.?|миллиард\w*))?", re.I)
NUMWORD_RE = re.compile(r"[а-яё]+", re.I)
UNIT_STEMS = [("рабоч", "wday"), ("календар", "cday"), ("дн", "day"), ("ден", "day"), ("сут", "day"),
              ("месяц", "month"), ("мес", "month"), ("недел", "week"), ("год", "year"), ("лет", "year"),
              ("час", "hour"), ("минут", "minute"), ("руб", "rub"), ("процент", "pct"), ("числ", "dom")]
SKIP_TOKENS = {"го", "й", "е", "ти", "х", "ми", "м", "ый", "ой", "ти"}


def _mult(word: str | None) -> float:
    if not word:
        return 1.0
    w = norm(word)
    if w.startswith("тыс"):
        return 1e3
    if w.startswith("млрд") or w.startswith("миллиард"):
        return 1e9
    if w.startswith("млн") or w.startswith("миллион"):
        return 1e6
    return 1.0


def _unit_after(t: str, pos: int) -> str | None:
    """Единица измерения сразу после числа (числительные в скобках пропускаются)."""
    seg = t[pos: pos + 70]
    if CHECK_FIXES:   # пакет 2: в формуле «2 000 000 × 0,1%» единица соседнего числа не наша
        seg = re.split(r"[×*=+÷/]|\s[-−]\s|\d", seg, maxsplit=1)[0]
    for tok in re.findall(r"[а-яa-z]+|%|₽", seg):
        if tok in NUM_WORDS or tok in SKIP_TOKENS:
            continue
        if tok == "%":
            return "pct"
        if tok == "₽":
            return "rub"
        if tok in ("раз", "раза"):
            return "times"
        for stem, unit in UNIT_STEMS:
            if tok.startswith(stem):
                return unit
        return None
    return None


def _strip_dates(text: str) -> tuple[set[tuple[int, int, int]], str]:
    found = set()
    for m in DATE_NUM.finditer(text):
        d, mo, y = (int(x) for x in m.groups())
        found.add((d, mo, y))
    for m in DATE_TEXT.finditer(text):
        mo = month_no(m.group(2))
        if mo:
            found.add((int(m.group(1)), mo, int(m.group(3))))
    text = DATE_NUM.sub(" ", text)
    text = DATE_TEXT.sub(" ", text)
    return found, text


def extract_numbers(text: str, answer: bool = False) -> tuple[list[tuple[float, str, str | None]], set]:
    """Числа текста: (значение, исходная запись, единица). Даты возвращаются отдельно."""
    dates, t = _strip_dates(norm(text))
    if answer:
        t = REF_RE.sub(" ", t)
        t = _DOCNUM.sub(" ", t)
        t = LISTMARK_RE.sub(" ", t)
    else:  # в контексте номера документов оставляем и как числа
        t = _DOCNUM.sub(lambda m: m.group(0) + " " + " ".join(re.findall(r"\d+", m.group(0))), t)
    out = []
    for m in NUM_RE.finditer(t):
        intpart = re.sub(r"[  ]", "", m.group(1))
        v = float(intpart + ("." + m.group(2) if m.group(2) else "")) * _mult(m.group(3))
        out.append((v, m.group(0).strip(), _unit_after(t, m.end())))
    # числительные словами: «трех месяцев», «пятнадцати миллионов рублей»
    total, cur, seq, last_end = 0.0, 0.0, [], 0
    for m in list(NUMWORD_RE.finditer(t)) + [None]:
        info = NUM_WORDS.get(m.group(0)) if m else None
        if info:
            kind, v = info
            if kind == "s":
                total += (cur or 1.0) * v
                cur = 0.0
            else:
                cur += v
            seq.append(m.group(0))
            last_end = m.end()
            continue
        if seq:
            out.append((total + cur, " ".join(seq), _unit_after(t, last_end)))
            total, cur, seq = 0.0, 0.0, []
    return out, dates


def _key(v: float) -> float:
    return round(v, 6)


def _unit_ok(au: str, cu: str | None) -> bool:
    if au in ("wday", "cday"):
        return cu in (au, "day")
    if au == "day":
        return cu in ("day", "wday", "cday")
    return au == cu


def unverified(answer: str, context: str, question: str) -> list[tuple[str, str]]:
    """Числа и даты ответа, которых нет в контексте/вопросе (с той же единицей) и которые не получаются расчётом."""
    ctx_items, ctx_dates = extract_numbers(context + "\n" + question)
    units: dict[float, set] = {}
    for v, _, u in ctx_items:
        units.setdefault(_key(v), set()).add(u)
    for d, mo, y in ctx_dates:
        for v in (d, mo, y):
            units.setdefault(_key(v), set()).add(None)
    q_vals = [v for v, _, _ in extract_numbers(question)[0]]
    # для проверки расчётов сначала числа вопроса и документов компании, затем остальные
    primary = context.split("=== ЗАКОНЫ", 1)[0] + "\n" + question
    first = sorted({v for v, _, _ in extract_numbers(primary)[0] if 0 < abs(v) < 1e12})
    rest = sorted({v for v, _, _ in ctx_items if 0 < abs(v) < 1e12} - set(first))
    base = (first + rest)[:500]
    ans_items, ans_dates = extract_numbers(answer, answer=True)
    bad = [(f"{d:02d}.{mo:02d}.{y}", "date") for d, mo, y in ans_dates if (d, mo, y) not in ctx_dates]
    for v, span, unit in ans_items:
        keys = [_key(v * k) for k in (1, 1e3, 1e6, 1e-3, 1e-6)]
        if CHECK_FIXES and unit in TERM_UNITS and v == 1 and not re.search(r"\d", span):   # «за один день»
            continue
        if v.is_integer() and 1990 <= v <= 2100 and not (CHECK_FIXES and unit is not None):  # год (не «2 050 рублей»)
            if any(k in units for k in keys):
                continue
            bad.append((span, "date"))
            continue
        if unit is None:
            if any(k in units for k in keys) or _derivable(v, base, q_vals):
                continue
        else:
            if any(_unit_ok(unit, cu) for k in keys for cu in units.get(k, ())):
                continue
            if (unit in ("rub", "pct") or v >= 100) and _derivable(v, base, q_vals):
                continue
        bad.append((span, "term" if unit in TERM_UNITS else "amount"))
    return bad


TERM_UNITS = {"day", "wday", "cday", "month", "week", "year", "hour", "minute", "dom", "times"}


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.01, abs(b) * 1e-6)


def _derivable(v: float, base: list[float], q: list[float]) -> bool:
    """Число получено простым расчётом: умножение/проценты из чисел документов и вопроса,
    сложение/вычитание — только с участием числа из вопроса."""
    if v == 0:
        return True
    cand = base
    for a in cand:
        for b in cand:
            if (_close(v, a * b) or _close(v, a * b / 100) or (b and _close(v, a / b))
                    or _close(v, a * (1 + b / 100)) or _close(v, a * (1 - b / 100))):
                return True
    for a in q:
        for b in cand + q:
            if _close(v, a + b) or _close(v, a - b) or _close(v, b - a):
                return True
        for b in q:
            for c in cand:
                if _close(v, a * b * c / 100) or _close(v, a * b * c):
                    return True
    return False


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z«\"(])")


def drop_sentences(answer: str, spans: list[str]) -> str:
    sents = SENT_SPLIT.split(answer)
    tail = r"(?![\d]|[.,]\d)" if CHECK_FIXES else r"(?![\d.,])"   # B1: дата или число перед точкой в конце предложения
    pats = [re.compile(r"(?<![\d.,])" + re.escape(sp) + tail, re.I) for sp in spans if sp]
    keep = [s for s in sents if not any(p.search(norm(s)) for p in pats)]
    if not keep or all(_is_source(s) for s in keep):   # от ответа осталась бы одна строка «Источник: …»
        return answer
    out = " ".join(keep).strip()
    return out if len(out.split()) >= 8 else answer


MD_RE = [(re.compile(r"\*\*|__|`"), ""), (re.compile(r"^#+\s*", re.M), ""),
         (re.compile(r"^\s*[-*•]\s+", re.M), ""), (re.compile(r"^\s*Ответ\s*:\s*", re.I), "")]


_SOURCE_START = re.compile(r"Источник\s*:", re.I)
# конец предложения: точка (не после сокращения вроде «г.», «п.», «ст.») или ?/! перед заглавной буквой или в конце текста
_SENT_END = re.compile(r"[.!?](?=\s+[А-ЯЁA-Z«(]|\s*$)")
_ABBR_BEFORE = re.compile(r"(?:^|[\s(«№])(?:г|гг|п|пп|ч|ст|д|ул|обл|ред|им|т|руб|коп|см|др|прим|абз)$", re.I)


def _is_source(sentence: str) -> bool:
    return bool(_SOURCE_START.match(sentence.strip()))


def trim_after_source(text: str) -> str:
    """B14: всё после предложения «Источник: …» — лишнее. Не трогает текст, если «Источник:» нет, если их несколько
    (ответ по нескольким договорам) или если ответ с источника начинается."""
    text = text or ""
    starts = list(_SOURCE_START.finditer(text))
    if len(starts) != 1 or not text[:starts[0].start()].strip():
        return text
    for m in _SENT_END.finditer(text, starts[0].end()):
        if m.group(0) == "." and _ABBR_BEFORE.search(text[:m.start()]):
            continue
        end = m.end()
        return text[:end].strip() if text[end:].strip() else text
    return text


def clean_answer(text: str) -> str:
    t = THINK_RE.sub("", text or "")
    for rx, rep in MD_RE:
        t = rx.sub(rep, t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > 2 and t[0] == t[-1] == '"':
        t = t[1:-1].strip()
    if len(t) > ANSWER_MAX_CHARS:
        cut = t[:ANSWER_MAX_CHARS]
        dot = cut.rfind(". ")
        t = cut[: dot + 1] if dot > ANSWER_MAX_CHARS // 2 else cut
    return t


# ============================================================================
# Ссылки в ответе: документы и пункты (цитаты для фасада и проверка 7.1)
# ============================================================================

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


_KB_CLAUSES: dict[int, list] = {}


def kb_clauses(d: Doc) -> list[dict]:
    if id(d) not in _KB_CLAUSES:
        _KB_CLAUSES[id(d)] = doc_clauses(d)
    return _KB_CLAUSES[id(d)]


# ---------------------------------------------------------------------------
# 7.1: проверка номеров пунктов и документов в ответе (флаг SOLUTION_CHECK_REFS)
# ---------------------------------------------------------------------------

_CODE_RE = re.compile(r"№\s*([A-Za-zА-ЯЁа-яё0-9][\w\-/.]*\d[\w\-/]*)|(?<![\w-])([А-ЯЁA-Z]{1,4}-\d+(?:/\d+)?)(?![\w-])|"
                      r"(?<![\w-])(\d+-ФЗ)\b")
_ART_REF = re.compile(r"(ТК|НК|КоАП)\s*РФ\W{0,3}(?:ст\.|стать\w*)\s*(\d+(?:\.\d+)?)|"
                      r"(?:ст\.|стать\w*)\s*(\d+(?:\.\d+)?)\s*(?:,\s*ч\.\s*\d+\s*)?(ТК|НК|КоАП)\s*РФ", re.I)
_OTHER_ART = re.compile(r"^\s*(?:,\s*)?(?:(?:ч|п)\.\s*\d+(?:[\s,–-]+\d+)*\s*)?(?:ст\.|стать\w*)\s*(\d+(?:\.\d+)?)", re.I)
_KB_NORM: dict[int, str] = {}


def _kb_norm(kb: KB) -> str:
    """Весь текст базы в нижнем регистре, «/» → «-» — для поиска номеров документов."""
    if id(kb) not in _KB_NORM:
        _KB_NORM[id(kb)] = "\n".join(d.text for d in kb.docs).lower().replace("ё", "е").replace("/", "-")
    return _KB_NORM[id(kb)]


def _article_known(kb: KB, num: str) -> bool:
    """Статья есть в базе или упоминается в тексте её статей (например, ст. 6 152-ФЗ из ст. 9)."""
    pat = re.compile(rf"(?:стать\w*|ст\.)\s*{re.escape(num)}(?![\d.])", re.I)
    return any(d.label.endswith(f"ст. {num}") or pat.search(d.text) for d in kb.docs)


def unverified_refs(kb: KB, answer: str, qtext: str, context_docs: list) -> list[tuple[str, str]]:
    """Ссылки ответа, которых нет в базе: (фрагмент ответа, "ref"). Пункт ищется в документе, к которому он
    относится по тексту; если документ не назван — хотя бы в одном документе контекста."""
    bad: list[tuple[str, str]] = []
    mentions = _mentions(kb, answer, qtext)
    for m in _CLAUSE_REF.finditer(answer):
        nxt = _OTHER_ART.match(answer[m.end(): m.end() + 40])
        if nxt or re.match(r"\s*(?:ч\.\s*\d+\s*)?(?:ст\.|стать)", answer[m.end(): m.end() + 12], re.I):
            # «п. 2–11 ч. 1 ст. 6» — пункты другой статьи: проверяем, что статья известна базе
            art = nxt.group(1) if nxt else None
            if art and not _article_known(kb, art):
                bad.append((answer[m.start(): m.end() + nxt.end()].strip(), "ref"))
            continue
        before = [x for x in mentions if x[0] <= m.start()]
        after = [x for x in mentions if x[0] >= m.end() and x[0] - m.end() < 60]
        if after and re.fullmatch(r"\s{0,3}", answer[m.end():after[0][0]]):
            cands = [after[0][2]]
        elif before and answer.rfind("(", 0, m.start()) > before[-1][1] and after:
            cands = [after[0][2]]
        elif before:
            cands = [before[-1][2]]
        elif after:
            cands = [after[0][2]]
        else:
            cands = []
        cands += [x[2] for x in reversed(before)] + [x[2] for x in after]
        pool = cands or list(context_docs)
        nums = _expand_nums(m.group(2))
        missing = []
        for num in nums:
            ok = False
            for d in pool or context_docs:
                clauses = kb_clauses(d)
                if d.kind == "law" and not clauses:
                    ok = True       # у статьи нет нумерованных частей (ТК) — проверить нечем
                    break
                key = _clause_key(d, m.group(1), num, None)
                if any(c["clause"] == key or c["clause"].startswith(key + ",") for c in clauses):
                    ok = True
                    break
                if m.group(1).lower() in ("пп", "подп") and any(c["clause"].endswith(f"пп. {num}") for c in clauses):
                    ok = True
                    break
            if not ok:
                missing.append(num)
        if missing:
            bad.append((m.group(0).strip(), "ref"))
    norm_kb = _kb_norm(kb)
    for m in _CODE_RE.finditer(answer):
        code = next(g for g in m.groups() if g)
        c = code.lower().replace("ё", "е").replace("/", "-").rstrip(".-")
        if c.isdigit() and len(c) <= 2:
            continue        # «№ 1» — номер доп. соглашения и т. п., сравнивать не с чем
        if c not in norm_kb:
            bad.append((code, "ref"))
    for m in _ART_REF.finditer(answer):
        num = m.group(2) or m.group(3)
        code = (m.group(1) or m.group(4)).upper().replace("КОАП", "КоАП")
        if not any(d.label == f"{code} РФ, ст. {num}" for d in kb.docs) and not _article_known(kb, num):
            bad.append((m.group(0).strip(), "ref"))
    seen, out = set(), []
    for sp, kind in bad:
        if sp not in seen:
            seen.add(sp)
            out.append((sp, kind))
    return out


def drop_ref_sentences(answer: str, spans: list[str]) -> str:
    """Удаляет предложения с неподтверждёнными ссылками (как drop_sentences для сроков, но по точному вхождению)."""
    sents = SENT_SPLIT.split(answer)
    keep = [s for s in sents if not any(sp and sp.lower() in s.lower() for sp in spans)]
    if not keep or all(_is_source(s) for s in keep):
        return answer
    out = " ".join(keep).strip()
    return out if len(out.split()) >= 8 else answer
