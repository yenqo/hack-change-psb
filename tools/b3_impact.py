"""Масштаб бага B3 офлайн: старая (v1.3) и исправленная проверка чисел на сохранённых ответах — без вызовов модели.

    python tools/b3_impact.py                       # все прогоны eval/runs/*.json (кроме стартера)
    python tools/b3_impact.py calib_v13_r1 calib_v13_r2 calib_v13_r3   # калибровочные прогоны (после golden set)

Для каждого ответа: черновик (debug.draft, если есть) и итог проверяются по контексту, собранному тем же путём,
что в ядре (документы компании целиком + отобранные законы по BM25 без эмбеддингов — приближение). Считаем ответы,
где v1.3 находил «неподтверждённое» число, которого нет при исправленном DOCNUM_RE: это лишний вызов исправления,
а для сроков и дат — ещё и удаление предложения.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    import checks
    import index as indexmod
    import kb as kbmod
    import dates
    import dialog
    import prompts
    import retrieval
    with contextlib.redirect_stderr(io.StringIO()):
        base = kbmod.build_kb()
    idx = indexmod.Index(base)
    labels = sys.argv[1:] or sorted(p.stem for p in (ROOT / "eval" / "runs").glob("*.json")
                                   if ".judge" not in p.name and not p.name.startswith("starter"))
    qs = {}
    for name in ("smoke_demo.csv", "golden.csv"):
        p = ROOT / "eval" / name
        if p.is_file():
            with open(p, encoding="utf-8", newline="") as f:
                qs.update({r["id"]: r["question"] for r in csv.DictReader(f)})
    rows = []
    for lb in labels:
        run = json.loads((ROOT / "eval" / "runs" / f"{lb}.json").read_text(encoding="utf-8"))
        for qid, final in run["answers"].items():
            q = qs.get(qid)
            if q is None or final.startswith("По данным базы знаний"):
                continue
            draft = (run.get("debug", {}).get(qid) or {}).get("draft") or final
            prep = dialog.Prepared(*dialog.parse_dialog(q))
            law_items, _ = retrieval.select_law(base, idx, prep, use_rerank=False)
            comp = retrieval.company_context(base, idx, prep)
            ctx = prompts.build_user_message(base, comp, law_items, dates.date_note(q, base), prep.current,
                                             prep.history)
            res = {}
            for mode, rx in (("old", checks.DOCNUM_RE), ("new", checks.DOCNUM_RE_CI)):
                checks._DOCNUM = rx
                res[mode] = checks.unverified(draft, ctx, q)
            only_old = [b for b in res["old"] if b not in res["new"]]
            rows.append((lb, qid, only_old, bool(res["old"]), bool(res["new"]),
                         [b for b in only_old if b[1] in ("term", "date")]))
    n = len(rows)
    false_any = [r for r in rows if r[2]]
    extra_repair = [r for r in rows if r[3] and not r[4]]
    drops = [r for r in rows if r[5]]
    print(f"Ответов: {n} (прогоны: {', '.join(labels)})")
    print(f"B3 давал ложное «неподтверждённое число»: {len(false_any)} ({len(false_any) / max(1, n):.1%})")
    print(f"  из них лишний вызов исправления (без B3 проверка чистая): {len(extra_repair)}")
    print(f"  из них ложный срок/дата → удаление предложения: {len(drops)}")
    for lb, qid, only_old, *_ in false_any[:20]:
        print(f"  {lb}/{qid}: {only_old}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
