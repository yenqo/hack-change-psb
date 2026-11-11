"""Аудит «лишнего» в ответах модели относительно эталона — без вызовов модели.

    python tools/answer_audit.py                                 # смоук-прогоны из eval/runs + примеры контракта
    python tools/answer_audit.py --labels smoke_x,smoke_y --set smoke

Для каждого ответа: числа и даты сверх эталона и вопроса, документы и пункты сверх эталона (ссылки разбираются тем же
кодом цитат, что в фасаде, — и в ответе, и в эталоне), служебные сведения базы (дата выгрузки и т. п.), длина
относительно эталона. Итог — доли ответов с каждым видом лишнего и примеры. Лишнее — не обязательно ошибка, но каждая
деталь вне эталона — повод для строгого судьи счесть ответ выдумкой.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

META_RE = re.compile(r"(?i)выгруз|действующ\w+ редакци\w+ стать|по состоянию на|гарант\b|в базе приведен")


def load_set(name: str) -> dict[str, dict]:
    path = ROOT / "eval" / ("smoke_demo.csv" if name == "smoke" else "golden.csv")
    with open(path, encoding="utf-8", newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def numbers(checks, text: str, answer: bool) -> set:
    items, dates = checks.extract_numbers(text, answer=answer)
    vals = {round(v, 4) for v, _, _ in items}
    return vals | {("date", d) for d in dates}


def audit(kb, ids, checks, pipeline, row: dict, answer: str) -> dict:
    q, exp = row["question"], row["expected"]
    allowed = numbers(checks, exp, True) | numbers(checks, q, False)
    extra_nums = sorted(str(x) for x in numbers(checks, answer, True) - allowed
                        if not (isinstance(x, float) and 1990 <= x <= 2100 and x in {round(y, 4) for y in allowed if isinstance(y, float)}))
    ca = pipeline.extract_citations(kb, answer, q, ids)
    ce = pipeline.extract_citations(kb, exp, q, ids)
    exp_docs = {c["doc_id"] for c in ce}
    exp_refs = {(c["doc_id"], c["clause"]) for c in ce if c["clause"]}
    extra_docs = sorted({c["doc_id"] for c in ca} - exp_docs)
    extra_refs = sorted(f"{d[:24]}:{cl}" for d, cl in {(c["doc_id"], c["clause"]) for c in ca if c["clause"]} - exp_refs
                        if d in exp_docs)
    # полнота: ключевые числа/даты и пункты эталона, которых нет в ответе
    ans_nums = numbers(checks, answer, True) | numbers(checks, answer, False)
    missing_nums = sorted(str(x) for x in numbers(checks, exp, True) - ans_nums)
    ans_refs = {(c["doc_id"], c["clause"]) for c in ca if c["clause"]}
    missing_refs = sorted(f"{d[:24]}:{cl}" for d, cl in exp_refs - ans_refs)
    return {"extra_nums": extra_nums, "extra_docs": extra_docs, "extra_refs": extra_refs,
            "meta": bool(META_RE.search(answer)), "words": len(answer.split()), "exp_words": len(exp.split()),
            "missing_nums": missing_nums, "missing_refs": missing_refs,
            "exp_n": len(numbers(checks, exp, True)) + len(exp_refs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="smoke_dense_current,smoke_sub5_repro,smoke_robust_build")
    ap.add_argument("--set", default="smoke")
    args = ap.parse_args()
    import checks
    import kb as kbmod
    import pipeline
    with contextlib.redirect_stderr(io.StringIO()):
        kb = kbmod.build_kb()
    ids = kbmod.doc_ids(kb)
    rows = load_set(args.set)
    results = []   # (метка, id, ответ, аудит)
    for lb in args.labels.split(","):
        run = json.loads((ROOT / "eval" / "runs" / f"{lb}.json").read_text(encoding="utf-8"))
        for qid, ans in run["answers"].items():
            if qid in rows and not ans.startswith("По данным базы знаний"):
                results.append((lb, qid, ans, audit(kb, ids, checks, pipeline, rows[qid], ans)))
    n = len(results)
    share = lambda f: sum(1 for *_, a in results if f(a)) / max(1, n)  # noqa: E731
    print(f"# Аудит лишнего в ответах ({n} ответов: {args.labels})\n")
    print("| вид лишнего | доля ответов |")
    print("|---|---|")
    print(f"| числа или даты сверх эталона и вопроса | {share(lambda a: a['extra_nums']):.0%} |")
    print(f"| документы сверх эталона | {share(lambda a: a['extra_docs']):.0%} |")
    print(f"| пункты сверх эталона (в тех же документах) | {share(lambda a: a['extra_refs']):.0%} |")
    print(f"| служебные сведения базы (выгрузка, «действующая редакция статьи») | {share(lambda a: a['meta']):.0%} |")
    print(f"| хоть что-то из перечисленного | {share(lambda a: a['extra_nums'] or a['extra_docs'] or a['extra_refs'] or a['meta']):.0%} |")
    print(f"| **полнота:** не хватает чисел/дат эталона | {share(lambda a: a['missing_nums']):.0%} |")
    print(f"| **полнота:** не хватает пунктов эталона | {share(lambda a: a['missing_refs']):.0%} |")
    ratio = [a["words"] / max(1, a["exp_words"]) for *_, a in results]
    print(f"\nДлина: ответ в среднем {statistics.mean(a['words'] for *_, a in results):.0f} слов, эталон — "
          f"{statistics.mean(a['exp_words'] for *_, a in results):.0f}; медиана отношения {statistics.median(ratio):.1f}×.\n")
    # устойчивость между прогонами: один и тот же вопрос
    by_q: dict[str, list] = {}
    for lb, qid, ans, a in results:
        by_q.setdefault(qid, []).append(a)
    varying = [qid for qid, lst in by_q.items() if len({bool(x["extra_docs"] or x["extra_nums"] or x["extra_refs"] or x["meta"])
                                                          for x in lst}) > 1]
    print(f"Вопросов, где лишнее есть в одних прогонах и нет в других: {len(varying)} из {len(by_q)} "
          f"({', '.join(sorted(varying, key=int))}).\n")
    print("## По вопросам\n")
    print("| id | тип | прогонов с лишним | что лишнее (примеры) |")
    print("|---|---|---|---|")
    for qid in sorted(by_q, key=int):
        lst = by_q[qid]
        bad = [x for x in lst if x["extra_docs"] or x["extra_nums"] or x["extra_refs"] or x["meta"]]
        ex = []
        for x in bad[:2]:
            parts = []
            if x["extra_docs"]:
                parts.append("док: " + ", ".join(d[:28] for d in x["extra_docs"]))
            if x["extra_refs"]:
                parts.append("пп: " + ", ".join(x["extra_refs"][:3]))
            if x["extra_nums"]:
                parts.append("числа: " + ", ".join(x["extra_nums"][:4]))
            if x["meta"]:
                parts.append("служебное")
            ex.append("; ".join(parts))
        print(f"| {qid} | {rows[qid]['type']} | {len(bad)}/{len(lst)} | {' ‖ '.join(ex)} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
