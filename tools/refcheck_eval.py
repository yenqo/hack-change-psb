"""Точность проверки ссылок 7.1 (refs.unverified_refs) на размеченном наборе — без вызовов модели.

    python tools/refcheck_eval.py

Набор: реальные ответы модели (eval/runs/*.json, docs/contract_examples/) считаются «верными» — каждое срабатывание
на них печатается для ручного разбора (ложное или настоящая ошибка модели). Из каждого ответа делаются «выдуманные»
мутанты: номер пункта, номер документа или статья закона заменяются на несуществующие. Метрики: доля ложных
срабатываний на исходных ответах и доля пойманных мутантов (по типам подмены).
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def collect() -> list[tuple[str, str]]:
    """(вопрос, ответ) из сохранённых реальных прогонов и примеров контракта."""
    out = []
    qs = {}
    for path in (ROOT / "eval" / "smoke_demo.csv",):
        with open(path, encoding="utf-8", newline="") as f:
            qs.update({r["id"]: r["question"] for r in csv.DictReader(f)})
    for run in sorted((ROOT / "eval" / "runs").glob("*.json")):
        if ".judge" in run.name or run.name.startswith("starter"):
            continue
        d = json.loads(run.read_text(encoding="utf-8"))
        for qid, ans in d.get("answers", {}).items():
            if qid in qs:
                out.append((qs[qid], ans))
    for ex in sorted((ROOT / "docs" / "contract_examples").glob("answer_*.json")):
        d = json.loads(ex.read_text(encoding="utf-8"))
        out.append((d["request"]["messages"][-1]["content"], d["response"]["text"]))
    seen, uniq = set(), []
    for q, a in out:
        if a not in seen and not a.startswith("По данным базы знаний"):
            seen.add(a)
            uniq.append((q, a))
    return uniq


MUTATIONS = [  # (тип, шаблон, замена)
    ("пункт договора", re.compile(r"(п\.\s*)(\d)\.(\d)\b"), lambda m: f"{m.group(1)}{m.group(2)}.{int(m.group(3)) + 17}"),
    ("номер документа", re.compile(r"\b(А-08|П-17|ПТ-05|К-12|ЮО-7)\b"),
     lambda m: {"А-08": "А-09", "П-17": "П-19", "ПТ-05": "ПТ-06", "К-12": "К-21", "ЮО-7": "ЮО-9"}[m.group(1)]),
    ("пункт из другого документа (8.2 есть только в П-17/2024)", re.compile(r"(п\.\s*)(\d\.\d)\b"),
     lambda m: f"{m.group(1)}8.2"),
    ("статья закона", re.compile(r"((?:ТК|НК|КоАП)\s*РФ,?\s*ст\.\s*)(\d+)"), lambda m: f"{m.group(1)}{int(m.group(2)) + 400}"),
    ("часть/пункт закона", re.compile(r"(ст\.\s*\d+,\s*(?:ч|п)\.\s*)(\d+)"), lambda m: f"{m.group(1)}{int(m.group(2)) + 30}"),
]


def main() -> int:
    import refs
    import kb as kbmod
    with contextlib.redirect_stderr(io.StringIO()):
        base = kbmod.build_kb()
    ctx = list(base.docs)  # контекст неизвестен — все документы (мягче, чем в ядре)
    data = collect()
    fp_cases = []
    for q, a in data:
        bad = refs.unverified_refs(base, a, q, ctx)
        if bad:
            fp_cases.append((a, bad))
    print(f"Исходных ответов: {len(data)}; со срабатыванием: {len(fp_cases)} ({len(fp_cases) / max(1, len(data)):.0%})")
    for a, bad in fp_cases:
        print(f"  ! {[s for s, _ in bad]} ← «{a[:160]}…»")
    print("\nМутанты (выдуманная ссылка должна быть поймана):")
    print("| подмена | мутантов | поймано |")
    print("|---|---|---|")
    total_n = total_hit = 0
    for kind, rx, repl in MUTATIONS:
        n = hit = 0
        for q, a in data:
            m = rx.search(a)
            if not m:
                continue
            mutant = a[:m.start()] + rx.sub(repl, a[m.start():m.end()], count=1) + a[m.end():]
            if mutant == a:
                continue
            n += 1
            before = {s for s, _ in refs.unverified_refs(base, a, q, ctx)}
            after = {s for s, _ in refs.unverified_refs(base, mutant, q, ctx)}
            hit += bool(after - before)
        total_n += n
        total_hit += hit
        print(f"| {kind} | {n} | {hit} ({hit / max(1, n):.0%}) |")
    print(f"| всего | {total_n} | {total_hit} ({total_hit / max(1, total_n):.0%}) |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
