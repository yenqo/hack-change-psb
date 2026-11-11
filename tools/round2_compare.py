"""Сравнение задачи второго раунда с калибровкой на временном стенде (docs/AGENT_TASKS_ROUND2.md, раздел 2.3).

    python tools/round2_compare.py --task R4_demo_r1,R4_demo_r2 \
        --scenarios eval/scenarios/R4_agent.csv --sc-off R4_sc_f0_r1,R4_sc_f0_r2 --sc-on R4_sc_f1_r1,R4_sc_f1_r2
    python tools/round2_compare.py --task R6_demo_f1_r1,R6_demo_f1_r2 --base R6_demo_f0_r1,R6_demo_f0_r2 --org ...

--base: парный замер (после R4) — сравнение не с калибровкой, а с прогонами без флага из той же сессии;
--org: ещё и оценки по эталонам организаторов (файлы *.judge_org_j2.json / *.judge_org_j1.json).

Демо: итог по 10 вопросам вне few-shot (j2 — основной, j1 — справочно) против среднего калибровки и порога шума;
регрессии по вопросам (оценка ниже минимума калибровки хотя бы в одном прогоне) с причиной; новые «выдумки» судьи;
настоящие выдумки (числа/даты ответа, которых нет в контексте, — проверка ядра); время p50/p95, доля кэша, размер запроса.
Сценарии агента: средний балл судьи без флага и с флагом; ключевые элементы ответа — шаблоны из <сценарии>.keys.json
({"id": {"must": [...], "first_word": "Да|Нет"}}).
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import eval as E  # noqa: E402

CALIB = ["calib_demo_r1", "calib_demo_r2", "calib_demo_r3"]
THRESHOLD = 0.02


def load(label: str, suffix: str = ""):
    run = json.loads((E.RUNS / f"{label}.json").read_text(encoding="utf-8"))
    jd = json.loads((E.RUNS / f"{label}.judge{suffix}.json").read_text(encoding="utf-8"))
    return run, jd


def demo_part(task: list[str], base: list[str] | None = None, org: bool = False) -> None:
    import checks
    paired = base is not None
    CAL = base or CALIB
    bname = "без флага" if paired else "калибровка"
    E.GOLDEN = E.SETS["smoke"]
    rows = E.load_golden()
    nonfs = [r for r in rows if r["in_fewshot"] == "нет"]
    print("## Демо: 10 вопросов вне few-shot" + (" (парный замер: без флага и с флагом в одной сессии)" if paired else "") + "\n")
    print(f"| эталоны | судья | {bname} (среднее; прогоны) | задача (прогоны) | разница | порог |\n|---|---|---|---|---|---|")
    judges = [("задания", "", "j2 (основной)"), ("задания", "_j1", "j1 (справочно)")]
    if org:
        judges += [("организаторов", "_org_j2", "j2"), ("организаторов", "_org_j1", "j1")]
    for refs, suf, name in judges:
        cal = [E.metrics(nonfs, *load(lb, suf))["final"] for lb in CAL]
        tsk = [E.metrics(nonfs, *load(lb, suf))["final"] for lb in task]
        diff = statistics.mean(tsk) - statistics.mean(cal)
        print(f"| {refs} | {name} | {statistics.mean(cal):.3f} ({' / '.join(f'{x:.3f}' for x in cal)}) | "
              f"{' / '.join(f'{x:.3f}' for x in tsk)} | {diff:+.3f} | {THRESHOLD} |")
    if org:
        print("\nВсе 18 вопросов (справочно; 8 из них — в few-shot):\n")
        print(f"| эталоны | судья | {bname} | задача | разница |\n|---|---|---|---|---|")
        for refs, suf, name in judges:
            cal = statistics.mean(E.metrics(rows, *load(lb, suf))["final"] for lb in CAL)
            tsk = statistics.mean(E.metrics(rows, *load(lb, suf))["final"] for lb in task)
            print(f"| {refs} | {name} | {cal:.3f} | {tsk:.3f} | {tsk - cal:+.3f} |")
    print("\n**Где сработал флаг** (по отладочным записям прогонов задачи):\n")
    fired = False
    for lb in task:
        run, _ = load(lb)
        for qid, d in run["debug"].items():
            x, a, rw = d.get("law_xref") or {}, d.get("law_absent") or [], d.get("rewrite")
            if x or a:
                fired = True
                print(f"- {qid} ({lb}): добавлено {len(x)} — " + "; ".join(f"{k}←{v}" for k, v in x.items()) +
                      (f"; нет в базе: {[f'{act[:30]} ст. {art}' for act, art in a]}" if a else ""))
            if rw:
                fired = True
                print(f"- {qid} ({lb}): переформулировано за {rw.get('sec')} c — «{rw.get('standalone')}»; "
                      f"запросы: {rw.get('queries')}")
    if not fired:
        print("- нигде")
    print(f"\n**Регрессии** (оценка j2 ниже минимума «{bname}» хотя бы в одном прогоне задачи):\n")
    cal_min = {r["id"]: min(load(lb)[1]["q"][r["id"]]["score"] for lb in CAL) for r in rows}
    cal_fab = {r["id"]: any(load(lb)[1]["q"][r["id"]]["fabricated"] for lb in CAL) for r in rows}
    found = False
    for r in rows:
        for lb in task:
            run, jd = load(lb)
            q = jd["q"][r["id"]]
            if q["score"] is not None and q["score"] < cal_min[r["id"]]:
                found = True
                tag = " (few-shot)" if r["in_fewshot"] == "да" else ""
                print(f"- {r['id']}{tag} в {lb}: {q['score']:.2f} (минимум «{bname}» {cal_min[r['id']]:.2f}) — "
                      f"{' | '.join(q['reasons'])[:300]}")
    if not found:
        print("- нет")
    print(f"\n**Новые «выдумки» судьи** (вопросы, где в прогонах «{bname}» выдумки не было):\n")
    new_fab = [(r["id"], lb) for r in rows for lb in task if load(lb)[1]["q"][r["id"]]["fabricated"] and not cal_fab[r["id"]]]
    print("- " + (", ".join(f"{q} ({lb})" for q, lb in new_fab) if new_fab else "нет"))
    print("\n**Настоящие выдумки** (числа/даты ответа, которых нет в контексте модели):\n")
    real = []
    for lb in task:
        run, _ = load(lb)
        ctx_path = E.RUNS / f"{lb}.contexts.json.gz"
        ctxs = json.load(gzip.open(ctx_path, "rt", encoding="utf-8")) if ctx_path.is_file() else {}
        for r in rows:
            ctx = ctxs.get(r["id"])
            if ctx:
                bad = checks.unverified(run["answers"][r["id"]], ctx, r["question"])
                if bad:
                    real.append(f"{r['id']} ({lb}): {[s for s, _ in bad]}")
    print("- " + ("; ".join(real) if real else "нет"))
    print("\n**Защитные метрики** (все 18 вопросов):\n")
    print("| прогоны | время чата на вопрос p50 / p95, с | доля кэша на входе | символов в запросе (среднее) |\n|---|---|---|---|")
    for name, lbs in ((bname, CAL), ("задача", task)):
        p50, p95, cache, chars = [], [], [], []
        for lb in lbs:
            run, jd = load(lb)
            m = E.metrics(rows, run, jd)
            p50.append(m["p50"]); p95.append(m["p95"]); cache.append(m["cache_share"])
            chars += [d.get("prompt_chars") or 0 for d in run["debug"].values()]
        print(f"| {name} | {statistics.mean(p50):.1f} / {statistics.mean(p95):.1f} | "
              f"{' / '.join(f'{c:.0%}' for c in cache)} | {statistics.mean(chars):.0f} |")


_KB = None


def _label(ci) -> str:
    """Название фрагмента по номеру (для отчёта о том, что добавил флаг)."""
    global _KB
    if _KB is None:
        import contextlib
        import io
        import kb as kbmod
        with contextlib.redirect_stderr(io.StringIO()):
            _KB = kbmod.build_kb()
    return _KB.chunks[int(ci)].label


def scenario_part(path: str, off: list[str], on: list[str]) -> None:
    E.GOLDEN = (ROOT / path).resolve()
    rows = E.load_golden()
    keys_path = Path(str(E.GOLDEN).replace(".csv", ".keys.json"))
    keys = json.loads(keys_path.read_text(encoding="utf-8")) if keys_path.is_file() else {}
    print(f"\n## Целевые сценарии агента ({Path(path).name}) — проверка механизма, не оценка качества\n")
    print("| сценарий | j2 без флага | j2 с флагом | j1 без флага | j1 с флагом | ключевые элементы без / с флагом | "
          "что добавил флаг (прогон 1) |\n|---|---|---|---|---|---|---|")

    def key_ok(qid: str, ans: str) -> bool:
        k = keys.get(qid, {})
        ok = all(re.search(p, ans, re.I) for p in k.get("must", []))
        if k.get("first_word"):
            ok = ok and re.match(rf"\W*({k['first_word']})\b", ans, re.I) is not None
        return ok

    tot = {"off": [], "on": []}
    for r in rows:
        cells = []
        for suf in ("", "_j1"):
            for lbs in (off, on):
                cells.append(statistics.mean(load(lb, suf)[1]["q"][r["id"]]["score"] for lb in lbs))
        k_off = sum(key_ok(r["id"], load(lb)[0]["answers"][r["id"]]) for lb in off)
        k_on = sum(key_ok(r["id"], load(lb)[0]["answers"][r["id"]]) for lb in on)
        tot["off"].append(k_off / len(off)); tot["on"].append(k_on / len(on))
        dbg = load(on[0])[0]["debug"].get(r["id"], {})
        what = (", ".join(f"{_label(k)} ← {v}" for k, v in (dbg.get("law_xref") or {}).items()) or "—") + \
            (f"; нет в базе: ст. {', '.join(a for _, a in dbg['law_absent'])}" if dbg.get("law_absent") else "") + \
            (f"; «{dbg['rewrite'].get('standalone')}»" if dbg.get("rewrite") else "")
        print(f"| {r['id']} | {cells[0]:.2f} | {cells[1]:.2f} | {cells[2]:.2f} | {cells[3]:.2f} | "
              f"{k_off}/{len(off)} → {k_on}/{len(on)} | {what} |")
    for suf, name in (("", "j2"), ("_j1", "j1")):
        a = statistics.mean(E.metrics(rows, *load(lb, suf))["final"] for lb in off)
        b = statistics.mean(E.metrics(rows, *load(lb, suf))["final"] for lb in on)
        print(f"\nИтог по сценариям ({name}): без флага {a:.3f}, с флагом {b:.3f} ({b - a:+.3f}).", end="")
    print(f"\nКлючевые элементы: без флага {statistics.mean(tot['off']):.0%}, с флагом {statistics.mean(tot['on']):.0%}.")
    print("\nОтветы с флагом (первый прогон):\n")
    for r in rows:
        print(f"- **{r['id']}** {r['question']}\n  → {load(on[0])[0]['answers'][r['id']]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--scenarios")
    ap.add_argument("--sc-off")
    ap.add_argument("--sc-on")
    ap.add_argument("--base", help="прогоны без флага из той же сессии (парный замер) вместо калибровки")
    ap.add_argument("--org", action="store_true", help="добавить оценки по эталонам организаторов")
    args = ap.parse_args()
    demo_part(args.task.split(","), args.base.split(",") if args.base else None, args.org)
    if args.scenarios:
        scenario_part(args.scenarios, args.sc_off.split(","), args.sc_on.split(","))
    return 0


if __name__ == "__main__":
    sys.exit(main())
