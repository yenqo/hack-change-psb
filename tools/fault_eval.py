"""Устойчивость к сбоям и нагрузке (B5–B8): симуляция прогонов на заглушке прокси в виртуальном времени — без токенов.

    python tools/fault_eval.py                     # обе таблицы для всех конфигураций
    python tools/fault_eval.py --part resilience   # сбои прокси и реранкера (50 вопросов, MAX_WORKERS=2)
    python tools/fault_eval.py --part load         # 50/100/200 вопросов × MAX_WORKERS 1/2 (B7)
    python tools/fault_eval.py --configs v1.3,B5

Модель: полный запрос ≈ 8 c (медиана на dev-стенде), компактный и минимальный — быстрее, джиттер ±30%;
эмбеддинги 0,4 c; реранкер 1 c. Бюджет — как на платформе (990 c из 20 мин). Сценарии — tests/golden_runner.py.
Метрики: доля итоговых ответов на полном уровне (4), на пониженных (3–0), запасные; вызовов чата на вопрос;
общее виртуальное время прогона.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import golden_runner  # noqa: E402

OFF = {"SOLUTION_SOFT_DEGRADE": "0", "SOLUTION_RERANK_RECHECK": "0", "SOLUTION_LEVEL_FORECAST": "0",
       "SOLUTION_FINAL_PASS": "0"}
CONFIGS = {
    "v1.3": OFF,
    "B5": {**OFF, "SOLUTION_SOFT_DEGRADE": "1"},
    "B5+B6": {**OFF, "SOLUTION_SOFT_DEGRADE": "1", "SOLUTION_RERANK_RECHECK": "1"},
    "B5+B6+B7": {**OFF, "SOLUTION_SOFT_DEGRADE": "1", "SOLUTION_RERANK_RECHECK": "1", "SOLUTION_LEVEL_FORECAST": "1"},
    "B5–B8": {"SOLUTION_SOFT_DEGRADE": "1", "SOLUTION_RERANK_RECHECK": "1", "SOLUTION_LEVEL_FORECAST": "1",
              "SOLUTION_FINAL_PASS": "1"},
}
# задачи второго раунда — поверх принятой конфигурации (B5–B8)
CONFIGS["B5–B8+R4"] = {**CONFIGS["B5–B8"], "SOLUTION_DATE_CONDITIONS": "1"}
CONFIGS["B5–B8+R6"] = {**CONFIGS["B5–B8"], "SOLUTION_CROSS_REFS": "1"}
CONFIGS["B5–B8+R7"] = {**CONFIGS["B5–B8"], "SOLUTION_REWRITE": "1"}
SEEDS = [1, 2, 3]
RESILIENCE = [  # (название, сценарий, доп. окружение)
    ("без сбоев", "normal", {}),
    ("прокси ×2 медленнее", "normal", {"SIM_SLOW": "2"}),
    ("прокси ×3 медленнее", "normal", {"SIM_SLOW": "3"}),
    ("разовые 400, 10%", "flaky4xx", {"FAULT_P": "0.1"}),
    ("400 + 429 + 503, 10%", "flaky_mix", {"FAULT_P": "0.1"}),
    ("таймауты реранкера, 30%", "rerank_timeout", {"SIM_RERANK_TIMEOUT_P": "0.3"}),
    ("медленный реранкер (~7 с)", "rerank_slow", {}),
]
LOAD = [(n, w) for n in (50, 100, 200) for w in (1, 2)]


def run(scenario: str, env: dict, seed: int) -> dict:
    res = golden_runner.run_subprocess("root", scenario, {"SIM": "1", "FAULT_SEED": str(seed), **env})
    sim = res["sim"]
    n = sim["n"]
    lv = sim["levels"]
    return {"full": lv["4"] / n, "lower": (lv["3"] + lv["2"] + lv["1"] + lv["0"]) / n, "fallback": lv["fallback"] / n,
            "calls": len(res["chat"]) / n, "time": sim["elapsed"], "rerank_off": res["flags"]["rerank_off"],
            "rerank_calls": len(res["rerank"]) / n, "rerank_ok": res["stats"].get("rerank_ok", 0) / n}


def row(name: str, cfg: str, rs: list[dict]) -> str:
    m = lambda k: statistics.mean(r[k] for r in rs)  # noqa: E731
    return (f"| {name} | {cfg} | {m('full'):.0%} | {m('lower'):.0%} | {m('fallback'):.0%} | {m('calls'):.2f} | "
            f"{m('rerank_ok'):.2f} из {m('rerank_calls'):.2f} | {m('time'):.0f} c (макс. {max(r['time'] for r in rs):.0f}) |")


HEAD = ("| сценарий | конфигурация | полный уровень | пониженный | запасные | вызовов чата/вопр. | реранкер: успешно из вызовов/вопр. "
        "| время прогона |\n|---|---|---|---|---|---|---|---|")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["resilience", "load", "all"], default="all")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    args = ap.parse_args()
    cfgs = [c for c in args.configs.split(",") if c]
    seeds = list(range(1, args.seeds + 1))
    if args.part in ("resilience", "all"):
        print("### Сбои прокси и реранкера (50 вопросов, MAX_WORKERS=2, бюджет 990 c)\n")
        print(HEAD)
        for name, sc, env in RESILIENCE:
            for cfg in cfgs:
                rs = [run(sc, {**CONFIGS[cfg], **env, "SIM_N": "50", "SIM_WORKERS": "2"}, s) for s in seeds]
                print(row(name, cfg, rs), flush=True)
        print()
    if args.part in ("load", "all"):
        print("### Нагрузка: число вопросов × MAX_WORKERS (без сбоев; прокси ×1 и ×2)\n")
        print(HEAD)
        for slow in ("1", "2"):
            for n, w in LOAD:
                for cfg in cfgs:
                    rs = [run("normal", {**CONFIGS[cfg], "SIM_N": str(n), "SIM_WORKERS": str(w), "SIM_SLOW": slow}, s)
                          for s in seeds]
                    print(row(f"{n} вопр., W={w}, ×{slow}", cfg, rs), flush=True)
    print(f"\n({len(seeds)} сида на ячейку; лимит платформы — 1200 c, бюджет решения — 990 c)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
