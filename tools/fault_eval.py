"""Устойчивость к сбоям прокси (B5): прогон 18 демо-вопросов на заглушке со случайными отказами — без токенов.

    python tools/fault_eval.py

Сравнивает поведение v1.3 (SOLUTION_SOFT_DEGRADE выключен) и B5 (включён) на сценариях tests/golden_runner.py:
flaky4xx (разовые 400 «перегружен» с вероятностью p), flaky_mix (+ 429/503), size400 (настоящий «слишком велик»
на полный запрос), normal. Метрики: доля вопросов на полном уровне (level4), на пониженных (компактный/минимальный/
короткий), запасные ответы, вызовов чата на вопрос, итоговые флаги режима.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import golden_runner  # noqa: E402

SEEDS = [1, 2, 3, 4, 5]
CASES = [("normal", None), ("flaky4xx", 0.05), ("flaky4xx", 0.10), ("flaky4xx", 0.20), ("flaky_mix", 0.10),
         ("size400", None)]


def run(scenario: str, p: float | None, seed: int, soft: bool) -> dict:
    env = {"SOLUTION_SOFT_DEGRADE": "1" if soft else "0", "FAULT_SEED": str(seed)}
    if p is not None:
        env["FAULT_P"] = str(p)
    res = golden_runner.run_subprocess("root", scenario, env)
    st = res["stats"]
    n = len(res["answers"])
    fallback = sum(a.startswith("По данным базы знаний") for _, a in res["answers"])
    return {"full": st.get("level4", 0) / n, "lower": (st.get("level2", 0) + st.get("level1", 0) +
                                                        st.get("level0", 0)) / n,
            "fallback": fallback / n, "calls": len(res["chat"]) / n, "end_mode":
            "tiny" if res["flags"]["tiny"] else "minimal" if res["flags"]["minimal"] else
            "compact" if res["flags"]["compact"] else "full", "answers": [a for _, a in res["answers"]]}


def main() -> int:
    print("| сценарий | режим | полный уровень | пониженный | запасные | вызовов чата/вопр. | режим в конце |")
    print("|---|---|---|---|---|---|---|")
    same_normal = None
    for sc, p in CASES:
        seeds = SEEDS if p is not None else [1]
        for soft in (False, True):
            rs = [run(sc, p, s, soft) for s in seeds]
            if sc == "normal":
                if soft:
                    same_normal = rs[0]["answers"] == base_normal
                else:
                    base_normal = rs[0]["answers"]
            ends = ", ".join(sorted({r["end_mode"] for r in rs}))
            name = f"{sc}" + (f" p={p:.0%}" if p is not None else "")
            print(f"| {name} | {'B5' if soft else 'v1.3'} | {statistics.mean(r['full'] for r in rs):.0%} | "
                  f"{statistics.mean(r['lower'] for r in rs):.0%} | {statistics.mean(r['fallback'] for r in rs):.0%} | "
                  f"{statistics.mean(r['calls'] for r in rs):.2f} | {ends} |")
    print(f"\nnormal: ответы B5 совпадают с v1.3 — {same_normal}")
    print(f"(по {len(SEEDS)} сидам для flaky-сценариев; 18 вопросов, MAX_WORKERS=1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
