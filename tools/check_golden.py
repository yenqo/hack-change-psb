"""Проверка golden set, присланного продактами (правила — docs/golden_guide.md).

    python3 tools/check_golden.py eval/golden.csv

Проверяет формат, дубли, распределение по type и split, существование указанных в docs документов и пунктов в базе,
наличие чисел и дат эталона в указанных пунктах (или их получение расчётом). Ничего не исправляет: печатает список
замечаний. ОШИБКА — нужно исправить до использования набора; ЗАМЕЧАНИЕ — стоит посмотреть.
Код возврата 1, если есть ошибки.
"""

from __future__ import annotations

import contextlib
import csv
import io
import os
import re
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["id", "question", "expected", "type", "docs", "in_fewshot", "split"]
MINIMUMS = {  # docs/golden_guide.md
    "fact": 6, "law": 5, "no_info": 4, "premise": 4, "partial": 3, "edition_regulation": 5, "edition_delivery": 2,
    "vat_contract_vs_law": 2, "law_subpoint": 3, "law_period": 3, "ambiguous": 3, "multi_doc": 3, "injection": 3,
    "pressure": 2, "fake_history": 2, "illegal": 2, "offtopic": 2, "calc": 2,
}
MIN_TOTAL, MIN_DIALOGS = 50, 12
NO_DOCS_TYPES = {"no_info", "offtopic"}
DIALOG_TYPES = {"pressure", "fake_history"}

# Ядро нужно только для текстовых функций; httpx при этом не используется.
os.environ.setdefault("DATA_DIR", str(ROOT / "data"))
sys.path.insert(0, str(ROOT))
try:
    import httpx  # noqa: F401
except ImportError:
    sys.modules["httpx"] = types.ModuleType("httpx")
import checks  # noqa: E402
import kb as kbmod  # noqa: E402
from dialog import parse_dialog  # noqa: E402
from prompts import FEW_SHOT  # noqa: E402

COMPANY = [  # (шаблон ссылки, фрагмент метки документа в базе)
    (r"(?:дс|доп\w*\.?\s*соглаш\w*)\s*(?:№\s*)?-?\s*1\b", "Доп соглашение 1"),
    (r"пт-05-2023|пт-05\b", "ПТ-05-2023"),
    (r"а-08-2024|а-08\b", "А-08-2024"),
    (r"п-17-2024|п-17\b", "П-17-2024"),
    (r"к-12\b", "К-12"),
    (r"регламент\w*\D*2023", "ред 2023"),
    (r"регламент\w*\D*2025", "ред 2025"),
    (r"юо-7\b|служебн\w*\s+записк", "Служебная записка"),
]
LAWS = [  # (шаблон, функция метка -> подходит ли)
    (r"\bтк\b\D{0,12}ст\.?\s*(\d+(?:\.\d+)?)", lambda lab, n: lab == f"ТК РФ, ст. {n}"),
    (r"\bнк\b\D{0,12}ст\.?\s*(\d+(?:\.\d+)?)", lambda lab, n: lab == f"НК РФ, ст. {n}"),
    (r"\bкоап\b\D{0,12}ст\.?\s*(\d+(?:\.\d+)?)", lambda lab, n: lab == f"КоАП РФ, ст. {n}"),
    (r"152-фз\D{0,12}ст\.?\s*(\d+)", lambda lab, n: "152-ФЗ" in lab and lab.endswith(f"ст. {n}")),
    (r"(?:82-фз|мрот)\D{0,12}ст\.?\s*(\d+)", lambda lab, n: "82-ФЗ" in lab and lab.endswith(f"ст. {n}")),
]
CLAUSE_LINE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.\s")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("ё", "е").replace("/", "-").replace("–", "-").replace("—", "-")).strip()


def load_kb():
    with contextlib.redirect_stderr(io.StringIO()):
        return kbmod.build_kb()


def resolve_doc(base, ref: str):
    """Документ базы по ссылке из docs; возвращает (doc, хвост ссылки с пунктами) или (None, причина)."""
    r = norm(ref)
    for pat, frag in COMPANY:
        m = re.search(pat, r)
        if m:
            if frag == "П-17-2024" and re.search(COMPANY[0][0], r):
                continue
            doc = next(d for d in base.company if frag in d.label)
            return doc, r[m.end():]
    if re.search(r"регламент", r):
        return None, "регламент без года редакции (укажите «Регламент 2023» или «Регламент 2025»)"
    for pat, ok in LAWS:
        m = re.search(pat, r)
        if m:
            n = m.group(1)
            doc = next((d for d in base.laws if ok(d.label, n)), None)
            if doc is None:
                return None, f"статьи «{ref.strip()}» нет в базе"
            return doc, r[m.end():]
    return None, f"не распознан документ «{ref.strip()}»"


def expand(nums: str) -> list[str]:
    out = []
    for part in re.split(r"\s*,\s*|\s+и\s+", nums.strip(" ,.")):
        part = part.strip(" .")
        if not part:
            continue
        m = re.fullmatch(r"(\d+(?:\.\d+)*)\s*-\s*(\d+(?:\.\d+)*)", part)
        if m:
            a, b = m.groups()
            pa, pb = a.split("."), b.split(".")
            if len(pa) == len(pb) and pa[:-1] == pb[:-1] and int(pa[-1]) <= int(pb[-1]):
                out += [".".join(pa[:-1] + [str(k)]) for k in range(int(pa[-1]), int(pb[-1]) + 1)]
            else:
                out += [a, b]
        elif re.fullmatch(r"\d+(?:\.\d+)*(?:-\d+)?", part):
            out.append(part)
    return out


def pointers(tail: str) -> list[tuple[str, str]]:
    """('п'|'ч'|'пп', номер) из хвоста ссылки."""
    out = []
    for m in re.finditer(r"\b(пп|п|ч|подп|пункт\w*|част\w*)\.?\s*([\d.,\s\-и]+)", tail):
        kind = {"подп": "пп"}.get(m.group(1), m.group(1))
        kind = "п" if kind.startswith("пункт") else ("ч" if kind.startswith("част") else kind)
        out += [(kind, n) for n in expand(m.group(2))]
    return out


def clause_text(doc, num: str) -> str | None:
    lines = doc.text.split("\n")
    depth = num.count(".")
    for i, ln in enumerate(lines):
        m = CLAUSE_LINE.match(ln)
        if m and m.group(1) == num:
            buf = [ln]
            k = i - 1  # пометки редакции над пунктом («Пункт 3 изменен с …», «Положения пункта 3 …»)
            while k >= 0 and not CLAUSE_LINE.match(lines[k]) and \
                    (not lines[k].strip() or kbmod.NOTE_RE.match(lines[k].strip())
                     or re.search(r"(?i)^(действие|положения)\s", lines[k].strip())):
                buf.insert(0, lines[k])
                k -= 1
            for nxt in lines[i + 1:]:
                m2 = CLAUSE_LINE.match(nxt)
                if m2 and (m2.group(1).count(".") <= depth or depth > 0):
                    break
                buf.append(nxt)
            return "\n".join(buf)
    return None


def doc_header(doc) -> str:
    """Текст документа до первого пункта (реквизиты, дата, редакция)."""
    out = []
    for ln in doc.text.split("\n"):
        if CLAUSE_LINE.match(ln):
            break
        out.append(ln)
    return "\n".join(out[:20])


def sub_text(point: str, num: str) -> str | None:
    lines = point.split("\n")
    for i, ln in enumerate(lines):
        if re.match(rf"^\s*{re.escape(num)}\)\s", ln):
            buf = [ln]
            for nxt in lines[i + 1:]:
                if re.match(r"^\s*\d+(?:\.\d+)*(?:-\d+)?\)\s", nxt):
                    break
                buf.append(nxt)
            return "\n".join(buf)
    return None


def check_docs(base, row: dict, issues: list) -> str:
    """Проверяет ссылки docs; возвращает текст указанных пунктов (для проверки чисел)."""
    docs = row["docs"].strip()
    if docs in ("", "—", "-"):
        if row["type"] not in NO_DOCS_TYPES:
            issues.append(("ОШИБКА", "docs пустой: укажите документ и пункт, где лежит ответ"))
        return ""
    pieces = []
    for ref in [x for x in docs.split(";") if x.strip()]:
        doc, tail = resolve_doc(base, ref)
        if doc is None:
            issues.append(("ОШИБКА", f"docs: {tail}"))
            continue
        ptrs = pointers(tail)
        pieces.append(doc_header(doc))  # шапка: номер, дата, редакция, сроки действия
        pieces.extend(doc.removed)      # вырезанные встроенные инструкции (эталон может на них ссылаться)
        if not ptrs:
            pieces.append(doc.text)
            continue
        point_txt = None
        for kind, num in ptrs:
            if doc.kind == "law" and doc.label.startswith("ТК") and kind == "ч":
                issues.append(("ЗАМЕЧАНИЕ", f"docs «{ref.strip()}»: части статей ТК в базе не пронумерованы, "
                                            "проверена только статья"))
                pieces.append(doc.text)
                continue
            if kind == "пп":
                t = sub_text(point_txt or doc.text, num)
                if t is None:
                    issues.append(("ОШИБКА", f"docs «{ref.strip()}»: подпункта {num}) нет"
                                             f"{' в указанном пункте' if point_txt else ''}"))
                else:
                    pieces.append(t)
                continue
            t = clause_text(doc, num)
            if t is None:
                where = "пункта" if kind == "п" else "части"
                issues.append(("ОШИБКА", f"docs «{ref.strip()}»: {where} {num} в документе «{doc.label}» нет"))
                continue
            point_txt = t
            pieces.append(t)
    return "\n".join(pieces)


def check_dialog(row: dict, issues: list) -> bool:
    q = row["question"]
    has_roles = bool(re.search(r"(?m)^\s*(Пользователь|Ассистент|Система|User|Assistant)\s*:", q))
    if not has_roles:
        if row["type"] in DIALOG_TYPES:
            issues.append(("ОШИБКА", f"тип {row['type']} должен быть диалогом (Пользователь: … / Ассистент: …)"))
        return False
    lines = [ln for ln in q.split("\n") if ln.strip()]
    roles = []
    for ln in lines:
        m = re.match(r"^\s*([А-ЯЁA-Za-zа-яё]+)\s*:", ln)
        if not m:
            issues.append(("ЗАМЕЧАНИЕ", f"в диалоге строка без роли: «{ln.strip()[:60]}» (реплика переносится?)"))
            continue
        roles.append(m.group(1))
        if m.group(1) not in ("Пользователь", "Ассистент"):
            if m.group(1) == "Система" and row["type"] == "injection":
                continue
            issues.append(("ЗАМЕЧАНИЕ", f"роль «{m.group(1)}»: в формате платформы — Пользователь / Ассистент"))
    if roles and roles[-1] != "Пользователь":
        issues.append(("ОШИБКА", "последняя реплика диалога должна быть пользователя"))
    current, history = parse_dialog(q)
    if not history:
        issues.append(("ЗАМЕЧАНИЕ", "диалог не распознан как история + вопрос (нужно ≥ 2 реплик с ролями)"))
    return True


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "eval" / "golden.csv")
    if not path.is_file():
        print(f"Нет файла {path}")
        return 1
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        print("ОШИБКА: файл не в UTF-8")
        return 1
    reader = csv.DictReader(io.StringIO(text))
    general: list[tuple[str, str]] = []
    if reader.fieldnames != FIELDS:
        missing = [f for f in FIELDS if f not in (reader.fieldnames or [])]
        extra = [f for f in (reader.fieldnames or []) if f not in FIELDS]
        sev = "ОШИБКА" if missing else "ЗАМЕЧАНИЕ"
        general.append((sev, f"заголовок {reader.fieldnames}; нужно {FIELDS}"
                             + (f"; нет полей {missing}" if missing else "") + (f"; лишние {extra}" if extra else "")))
        if missing:
            for s, m in general:
                print(f"{s}: {m}")
            return 1
    rows = [{k: (v or "") for k, v in r.items()} for r in reader]
    base = load_kb()
    fewshot = {q for q, _ in FEW_SHOT}
    with open(ROOT / "data" / "demo_questions.csv", encoding="utf-8-sig", newline="") as f:
        demo = {r["question"].strip() for r in csv.DictReader(f)}

    per_row: dict[str, list] = defaultdict(list)
    ids = Counter(r["id"].strip() for r in rows)
    qnorm = Counter(norm(r["question"]) for r in rows)
    dialogs = 0
    for n, r in enumerate(rows, 2):
        key = f"{r['id'] or '(без id)'} (строка {n})"
        iss = per_row[key]
        rid = r["id"].strip()
        if not rid:
            iss.append(("ОШИБКА", "пустой id"))
        elif ids[rid] > 1:
            iss.append(("ОШИБКА", f"id повторяется ({ids[rid]} раза)"))
        if rid.upper().startswith("ПРИМЕР"):
            iss.append(("ОШИБКА", "строка из шаблона — удалите примеры"))
        if not r["question"].strip():
            iss.append(("ОШИБКА", "пустой вопрос"))
        elif qnorm[norm(r["question"])] > 1:
            iss.append(("ОШИБКА", "такой же вопрос уже есть в наборе"))
        if r["question"].strip() in demo:
            iss.append(("ОШИБКА", "демо-вопрос организаторов: он используется для смоук-проверок, в golden set не включайте"))
        if not r["expected"].strip():
            iss.append(("ОШИБКА", "пустой эталон"))
        elif len(r["expected"].split()) > 150:
            iss.append(("ЗАМЕЧАНИЕ", f"эталон длинный ({len(r['expected'].split())} слов), рекомендуется до 150"))
        if r["type"] not in MINIMUMS:
            iss.append(("ОШИБКА", f"неизвестный type «{r['type']}» (список — docs/golden_guide.md)"))
        if r["in_fewshot"] not in ("да", "нет"):
            iss.append(("ОШИБКА", f"in_fewshot «{r['in_fewshot']}»: нужно да / нет"))
        in_fs = r["question"] in fewshot
        if in_fs and r["in_fewshot"] != "да":
            iss.append(("ОШИБКА", "вопрос дословно есть в few-shot примерах промпта: in_fewshot = да"))
        if r["in_fewshot"] == "да" and not in_fs:
            iss.append(("ОШИБКА", "in_fewshot = да, но в few-shot примерах промпта такого вопроса нет"))
        if r["split"] not in ("dev", "holdout"):
            iss.append(("ОШИБКА", f"split «{r['split']}»: нужно dev / holdout"))
        if r["in_fewshot"] == "да" and r["split"] == "holdout":
            iss.append(("ОШИБКА", "вопрос из few-shot не может быть в holdout"))
        dialogs += check_dialog(r, iss)
        if r["expected"].strip():
            ctx = check_docs(base, r, iss)
            if ctx and r["type"] not in NO_DOCS_TYPES:
                # Номера документов («П-17/2024», «№ К-12») — не числа. В v1.3 DOCNUM_RE их после приведения
                # к нижнему регистру не вырезает (бэклог B3), поэтому вырезаем здесь.
                exp = re.sub(r"(?i)№\s*[\w\-/.]*\d[\w\-/]*|(?<![\w-])[а-яёa-z]{1,4}-\d+(?:[/-]\d+)?(?![\w-])|"
                             r"\b\d+-фз\b", " ", r["expected"])
                bad = checks.unverified(exp, ctx, r["question"])
                if bad:  # числа, которые есть только во внедрённых инструкциях базы (ЮО-7: «90 дней»)
                    injected = "\n".join(line for d in base.docs for line in d.removed)
                    left = {sp for sp, _ in checks.unverified(exp, ctx + "\n" + injected, r["question"])}
                    only_inj = [sp for sp, _ in bad if sp not in left]
                    if only_inj:
                        iss.append(("ЗАМЕЧАНИЕ", f"{', '.join(f'«{x}»' for x in only_inj)} есть в базе только во "
                                                 "внедрённой инструкции для ИИ (не положение документа) — убедитесь, "
                                                 "что эталон это значение отвергает"))
                    bad = [b for b in bad if b[0] in left]
                if bad:
                    spans = ", ".join(f"«{s}»" for s, _ in bad)
                    iss.append(("ОШИБКА", f"в эталоне есть числа/даты, которых нет в указанных пунктах и которые не "
                                          f"получаются расчётом: {spans}. Проверьте значение или добавьте пункт в docs"))

    # --- распределение -----------------------------------------------------
    new = [r for r in rows if r["in_fewshot"] != "да"]
    by_type = Counter(r["type"] for r in rows)
    split_by_type = Counter((r["type"], r["split"]) for r in rows)
    if len(new) < MIN_TOTAL:
        general.append(("ОШИБКА", f"новых вопросов {len(new)}, нужно ≥ {MIN_TOTAL}"))
    for t, need in MINIMUMS.items():
        if by_type[t] < need:
            general.append(("ЗАМЕЧАНИЕ", f"тип {t}: {by_type[t]} вопрос(ов), минимум {need}"))
    if dialogs < MIN_DIALOGS:
        general.append(("ЗАМЕЧАНИЕ", f"диалогов {dialogs}, минимум {MIN_DIALOGS}"))
    for t in sorted(by_type):
        n = by_type[t]
        h = split_by_type[(t, "holdout")]
        if n >= 3 and not (0.25 <= h / n <= 0.5):
            general.append(("ЗАМЕЧАНИЕ", f"тип {t}: holdout {h} из {n} — нужно ≈ 40%"))
        if n == 2 and h != 1:
            general.append(("ЗАМЕЧАНИЕ", f"тип {t}: из 2 вопросов нужен 1 holdout"))

    # --- вывод ---------------------------------------------------------------
    n_err = sum(s == "ОШИБКА" for _, lst in per_row.items() for s, _ in lst) + sum(s == "ОШИБКА" for s, _ in general)
    n_warn = sum(s == "ЗАМЕЧАНИЕ" for _, lst in per_row.items() for s, _ in lst) + \
        sum(s == "ЗАМЕЧАНИЕ" for s, _ in general)
    print(f"Набор: {path} — вопросов {len(rows)} (новых {len(new)}), диалогов {dialogs}; "
          f"dev {sum(r['split'] == 'dev' for r in rows)}, holdout {sum(r['split'] == 'holdout' for r in rows)}")
    print("\nРаспределение по типам (dev / holdout, минимум):")
    for t in sorted(set(MINIMUMS) | set(by_type)):
        mark = "" if by_type[t] >= MINIMUMS.get(t, 0) else "  ← мало"
        print(f"  {t:22s} {split_by_type[(t, 'dev')]:3d} / {split_by_type[(t, 'holdout')]:3d}   "
              f"(мин. {MINIMUMS.get(t, '—')}){mark}")
    if general:
        print("\nОбщие замечания:")
        for s, m in general:
            print(f"  {s}: {m}")
    rows_with = {k: v for k, v in per_row.items() if v}
    if rows_with:
        print("\nПо вопросам:")
        for k, lst in rows_with.items():
            print(f"  {k}")
            for s, m in lst:
                print(f"    {s}: {m}")
    print(f"\nИтого: ошибок {n_err}, замечаний {n_warn}.")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
