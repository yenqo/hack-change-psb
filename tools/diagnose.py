"""Диагностика ответов (R1): поиск, факты сверх эталона и выдумки, длина — на сохранённых прогонах, без вызовов модели.

    python tools/diagnose.py --labels calib_demo_r1,calib_demo_r2,calib_demo_r3 --set smoke
    python tools/diagnose.py --labels <прогоны на golden set> --set golden

Нужны прогоны tools/eval.py (ответы, debug с ранжированием законов, сжатые контексты *.contexts.json.gz) и оценки
судьи (*.judge.json). Нужные документы и пункты — из поля docs набора (формат docs/golden_guide.md).
(а) Поиск: попал ли нужный документ/пункт в контекст; позиция нужного фрагмента закона до и после реранкера;
    для вопросов с баллом < 0,5 — причина: поиск (нужного нет в контексте) или генерация (было, но ответ неверный).
(б) Факты: детали сверх эталона (кандидаты в «выдумку» для судьи) и настоящие выдумки — числа и ссылки, которых нет
    в контексте; пересечение с оценками 0.
(в) Длина: слова у нас и у стартового примера, доля > 150 слов, связь длины и лишнего с оценкой.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import io
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def needed_items(base, resolve, pointers, docs_field: str):
    """Нужные документы и пункты из поля docs: [(doc, clause_key | None)]."""
    out = []
    if not docs_field or docs_field.strip() in ("—", "-"):
        return out
    for ref in docs_field.replace("Регламент ред.", "Регламент").split(";"):
        doc, tail = resolve(base, ref)
        if doc is None:
            continue
        ptrs = pointers(tail) if isinstance(tail, str) else []
        if not ptrs:
            out.append((doc, None))
        for kind, num in ptrs:
            if doc.kind == "company":
                out.append((doc, num))
            else:
                word = "п." if doc.label.startswith("НК") else "ч."
                out.append((doc, f"{word} {num}" if kind != "пп" else None))
    return out


def chunk_matches(chunk, doc, clause) -> bool:
    if chunk.doc != doc.idx:
        return False
    if clause is None:
        return True
    return f", {clause}" in chunk.label + ","


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--set", default="smoke", choices=["smoke", "golden"])
    ap.add_argument("--judge", default="", help="суффикс файла оценок: '' (j2) или '_j1'")
    args = ap.parse_args()
    import answer_audit as A
    import checks
    import check_golden as CG
    import kb as kbmod
    import pipeline
    with contextlib.redirect_stderr(io.StringIO()):
        base = kbmod.build_kb()
    ids = kbmod.doc_ids(base)
    by_label = {d.label: d for d in base.docs}
    rows = A.load_set(args.set)
    with open(ROOT / "eval" / "baseline_answers" / "starter.json", encoding="utf-8") as f:
        starter = json.load(f)
    recs = []
    for lb in args.labels.split(","):
        run = json.loads((ROOT / "eval" / "runs" / f"{lb}.json").read_text(encoding="utf-8"))
        jd = json.loads((ROOT / "eval" / "runs" / f"{lb}.judge{args.judge}.json").read_text(encoding="utf-8"))
        cpath = ROOT / "eval" / "runs" / f"{lb}.contexts.json.gz"
        ctxs = json.load(gzip.open(cpath, "rt", encoding="utf-8")) if cpath.is_file() else {}
        for qid, ans in run["answers"].items():
            if qid not in rows:
                continue
            row, dbg = rows[qid], run["debug"].get(qid) or {}
            fallback = ans.startswith("По данным базы знаний")
            # (а) поиск
            need = needed_items(base, CG.resolve_doc, CG.pointers, row.get("docs", ""))
            company_ctx = {by_label[l].idx for l in dbg.get("company", []) if l in by_label}
            law_whole = set(dbg.get("law_whole") or [])
            law_chunks = [base.chunks[i] for i in dbg.get("law_chunks") or []]
            hybrid = [base.chunks[i] for i in dbg.get("law_hybrid") or []]
            order = [base.chunks[i] for i in dbg.get("law_order") or []]
            found, missing, pos = [], [], []
            for doc, clause in need:
                if doc.kind == "company":
                    ok = doc.idx in company_ctx
                else:
                    ok = doc.idx in law_whole or any(chunk_matches(c, doc, clause) for c in law_chunks)
                    rh = next((i + 1 for i, c in enumerate(hybrid) if chunk_matches(c, doc, clause)), None)
                    ro = next((i + 1 for i, c in enumerate(order) if chunk_matches(c, doc, clause)), None)
                    pos.append((doc.label + (f", {clause}" if clause else ""), rh, ro))
                (found if ok else missing).append(doc.label + (f", {clause}" if clause else ""))
            # (б) факты
            au = A.audit(base, ids, checks, pipeline, row, ans)
            ctx = ctxs.get(qid) or ""
            ctx_docs = [base.docs[i] for i in company_ctx] + [base.docs[i] for i in law_whole] + \
                [base.docs[c.doc] for c in law_chunks]
            real_nums = [s for s, _ in checks.unverified(ans, ctx, row["question"])] if ctx else None
            real_refs = [s for s, _ in checks.unverified_refs(base, ans, row["question"], ctx_docs)] if ctx_docs else None
            q = jd["q"].get(qid, {})
            recs.append({
                "label": lb, "id": qid, "type": row["type"], "fewshot": row["in_fewshot"] == "да", "fallback": fallback,
                "score": q.get("score"), "rounds": q.get("rounds", []), "judge_fab": q.get("fabricated"),
                "need": len(need), "found": found, "missing": missing, "pos": pos,
                "rerank_used": dbg.get("rerank_used"), "level": dbg.get("level"),
                "extra": bool(au["extra_nums"] or au["extra_docs"] or au["extra_refs"] or au["meta"]),
                "extra_detail": au, "real_nums": real_nums, "real_refs": real_refs,
                "words": len(ans.split()), "starter_words": len((starter.get(row["question"]) or "").split()),
            })
    report(recs)
    return 0


def pct(x: float) -> str:
    return f"{x:.0%}"


def report(recs: list[dict]) -> None:
    n = len(recs)
    print(f"# R1. Диагностика ({n} ответов)\n")
    # (а) поиск
    print("## (а) Поиск\n")
    need = [r for r in recs if r["need"]]
    miss = [r for r in need if r["missing"]]
    print(f"Ответов с известными нужными документами/пунктами: {len(need)}; нужное попало в контекст полностью — "
          f"{len(need) - len(miss)} ({pct((len(need) - len(miss)) / max(1, len(need)))}).")
    print(f"Реранкер работал: {sum(1 for r in recs if r['rerank_used'])} из {n} ответов (локально реранкера нет — на "
          "платформе позиции «после» будут другими).\n")
    if miss:
        print("Чего не хватило в контексте:\n")
        for r in miss:
            print(f"- {r['label']}/{r['id']} [{r['type']}]: {', '.join(r['missing'])}")
        print()
    pos = [(r["id"], p) for r in recs for p in r["pos"]]
    if pos:
        print("Позиция нужного фрагмента закона в ранжировании (до / после реранкера; — — не найден в топ-40):\n")
        seen = set()
        for qid, (name, rh, ro) in pos:
            if (qid, name) in seen:
                continue
            seen.add((qid, name))
            hs = [p[1] for q, p in pos if q == qid and p[0] == name]
            print(f"- вопрос {qid}: {name} — {', '.join(str(h) if h else '—' for h in hs)} (по прогонам)")
        print()
    low = [r for r in recs if r["score"] is not None and r["score"] < 0.5 and not r["fewshot"]]
    by_search = [r for r in low if r["missing"] or r["fallback"]]
    print(f"Ответов вне few-shot с баллом < 0,5: {len(low)}; причина — поиск (нужного нет в контексте / запасной ответ): "
          f"{len(by_search)}; генерация (нужное было): {len(low) - len(by_search)}.\n")
    for r in low:
        why = "поиск" if r in by_search else "генерация"
        detail = []
        if r["judge_fab"]:
            detail.append("судья: выдумка")
        if r["extra"]:
            detail.append("есть лишнее")
        if r["real_nums"]:
            detail.append(f"числа вне контекста: {r['real_nums'][:3]}")
        if r["real_refs"]:
            detail.append(f"ссылки вне базы: {r['real_refs'][:3]}")
        print(f"- {r['label']}/{r['id']} [{r['type']}] балл {r['score']:.2f} — {why}; {'; '.join(detail) or '—'}")
    # (б) факты
    print("\n## (б) Факты сверх эталона и выдумки\n")
    ex = [r for r in recs if r["extra"]]
    real = [r for r in recs if r["real_nums"] or r["real_refs"]]
    zeros = [r for r in recs if r["rounds"] and min(r["rounds"]) == 0.0]
    print(f"| показатель | ответов |\n|---|---|")
    print(f"| с деталями сверх эталона (кандидаты в «выдумку» для судьи) | {len(ex)} ({pct(len(ex) / n)}) |")
    print(f"| с настоящими выдумками (числа/ссылки, которых нет в контексте) | {len(real)} ({pct(len(real) / n)}) |")
    print(f"| судья поставил 0 хотя бы в одном раунде | {len(zeros)} ({pct(len(zeros) / n)}) |")
    print(f"| … из них с деталями сверх эталона | {sum(r['extra'] for r in zeros)} |")
    print(f"| … из них с настоящими выдумками | {sum(bool(r['real_nums'] or r['real_refs']) for r in zeros)} |")
    print(f"| судья отметил выдумку | {sum(bool(r['judge_fab']) for r in recs)} ({pct(sum(bool(r['judge_fab']) for r in recs) / n)}) |")
    if real:
        print("\nНастоящие выдумки (нет в контексте):\n")
        for r in real:
            print(f"- {r['label']}/{r['id']}: числа {r['real_nums']}, ссылки {r['real_refs']}")
    # (в) длина
    print("\n## (в) Длина\n")
    ours = [r["words"] for r in recs]
    st = [r["starter_words"] for r in recs if r["starter_words"]]
    print(f"Слов: у нас медиана {statistics.median(ours):.0f} (от {min(ours)} до {max(ours)}), у стартового примера "
          f"медиана {statistics.median(st):.0f}; наших ответов длиннее 150 слов — {sum(w > 150 for w in ours)}.\n")
    scored = [r for r in recs if r["score"] is not None]
    if scored:
        terc = sorted(scored, key=lambda r: r["words"])
        k = len(terc) // 3
        groups = [("короткие", terc[:k]), ("средние", terc[k:2 * k]), ("длинные", terc[2 * k:])]
        print("| группа | слов | средний балл | доля с лишним | доля «выдумок» судьи |\n|---|---|---|---|---|")
        for name, g in groups:
            print(f"| {name} | {min(r['words'] for r in g)}–{max(r['words'] for r in g)} | "
                  f"{statistics.mean(r['score'] for r in g):.2f} | {pct(sum(r['extra'] for r in g) / len(g))} | "
                  f"{pct(sum(bool(r['judge_fab']) for r in g) / len(g))} |")
        wx = [r["score"] for r in scored if r["extra"]]
        nx = [r["score"] for r in scored if not r["extra"]]
        print(f"\nСредний балл: с лишним {statistics.mean(wx):.2f} ({len(wx)} отв.), без лишнего "
              f"{statistics.mean(nx) if nx else float('nan'):.2f} ({len(nx)} отв.).")


if __name__ == "__main__":
    sys.exit(main())
