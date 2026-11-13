"""Отпечаток поведения ядра для рефакторинга без изменения поведения.

    python tools/fingerprint.py record --out /tmp/fp_before     # до изменений
    python tools/fingerprint.py check  --ref /tmp/fp_before     # после: побайтное сравнение, код 1 при расхождении

Golden master (tests/test_golden_master.py) проверяет только путь v1.3: все флаги после v1.3 в тестах выключены.
Отпечаток снимает то же самое (точные тексты запросов к чату, запросы к эмбеддеру и реранкеру, ответы, счётчики)
для текущих настроек по умолчанию («prod») и для «всё включено» («all»: плюс выключенные флаги B3, 7.1, R4, R6,
B19–B20) на расширенном наборе вопросов: демо + целевые сценарии eval/scenarios (там срабатывают R7 и B12).
Заглушка прокси — из tests/golden_runner.py; на запрос переформулирования (R7) она отвечает разбираемым JSON,
чтобы ветка R7 выполнялась целиком. Дополнительно — веб-путь: ответы фасада LegalAssistant (статус, цитаты,
контекст, предупреждения, отладка) без времён.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

CONFIGS = {
    "prod": {},
    "all": {"SOLUTION_CROSS_REFS": "1", "SOLUTION_DATE_CONDITIONS": "1", "SOLUTION_CHUNK_FIXES": "1",
            "SOLUTION_CHECK_REFS": "1", "SOLUTION_FIX_DOCNUM": "1"},
    # ветки «выключено» у принятых флагов (кроме тех, что выключает golden master v1.3)
    "alt": {"SOLUTION_REWRITE_PROMPT": "0", "SOLUTION_FEWSHOT": "demo", "SOLUTION_SOURCE_RULE": "0",
            "SOLUTION_CHECK_FIXES": "0", "SOLUTION_GENERAL_POINTS": "0", "SOLUTION_ANSWER_TEMPLATE": "0"},
}
ALT_SCENARIOS = ["normal", "fabricated"]
# Флаги, которые tests/harness.py фиксирует на v1.3: для отпечатка — значения по умолчанию из config.py
PROD_DEFAULTS = {"SOLUTION_PREBUILT_INDEX": "1", "SOLUTION_SOFT_DEGRADE": "1", "SOLUTION_RERANK_RECHECK": "1",
                 "SOLUTION_LEVEL_FORECAST": "1", "SOLUTION_FINAL_PASS": "1", "SOLUTION_CONCISE": "1",
                 "SOLUTION_DATE_CONDITIONS": "0", "SOLUTION_CROSS_REFS": "0", "SOLUTION_REWRITE": "1",
                 "SOLUTION_GENERAL_POINTS": "1", "SOLUTION_ANSWER_TEMPLATE": "1", "SOLUTION_FEWSHOT": "fictional",
                 "SOLUTION_SOURCE_RULE": "1", "SOLUTION_CHECK_FIXES": "1", "SOLUTION_CHUNK_FIXES": "0"}
SCENARIOS = ["normal", "fabricated", "reject_full", "reject_compact", "tiny", "transient", "no_dense"]
_TIME_KEY = re.compile(r"(?i)time|elapsed|latency|_ms$|_sec$|seconds|^t_")


def questions_file(work: Path) -> Path:
    """Демо + вопросы целевых сценариев (без повторов)."""
    seen, rows = set(), []
    files = [ROOT / "data" / "demo_questions.csv"] + sorted((ROOT / "eval" / "scenarios").glob("*.csv"))
    for p in files:
        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                q = r["question"]
                if q not in seen:
                    seen.add(q)
                    rows.append(q)
    out = work / "questions.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question"])
        for i, q in enumerate(rows, 1):
            w.writerow([str(i), q])
    return out


def strip_times(x):
    if isinstance(x, dict):
        return {k: strip_times(v) for k, v in x.items() if not _TIME_KEY.search(str(k))}
    if isinstance(x, list):
        return [strip_times(v) for v in x]
    return x


# ---------------------------------------------------------------------------- дочерний процесс
def _rewrite_aware(proxy_cls):
    class Proxy(proxy_cls):
        def _chat(self, payload, timeout, attempts):
            msgs = payload["messages"]
            rewrite = getattr(self.core, "SYSTEM_REWRITE", None)
            if rewrite and msgs and msgs[0]["content"] == rewrite:
                self.chat_calls.append({"messages": [[m["role"], self._keep(m["content"])] for m in msgs],
                                        "max_tokens": payload.get("max_tokens"), "kind": "rewrite"})
                user = msgs[-1]["content"]
                tail = user.strip().splitlines()[-1][:200] if user.strip() else "вопрос"
                return self._reply(json.dumps({"question": "Переформулировано: " + tail,
                                               "queries": [tail[:80], "срок договора поставки"]}, ensure_ascii=False))
            return super()._chat(payload, timeout, attempts)
    return Proxy


def child_batch(scenario: str, out: str) -> None:
    sys.path.insert(0, str(TESTS))
    import golden_runner
    golden_runner.FakeProxy = _rewrite_aware(golden_runner.FakeProxy)
    res = golden_runner.run("root", scenario)
    res.pop("_elapsed", None)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)


def child_web(out: str) -> None:
    sys.path.insert(0, str(TESTS))
    import golden_runner
    import harness
    os.environ["SOLUTION_MODE"] = "web"
    extra = {"LLM_BASE_URL": "http://stub.invalid/v1", "LLM_API_KEY": "test-key", "LLM_MODEL": "stub-model",
             "SOLUTION_MODE": "web"}
    core = harness.load_core("root", extra)
    proxy = _rewrite_aware(golden_runner.FakeProxy)(core, "normal")
    harness.patch_everywhere(core, "post_json", proxy)
    import assistant
    a = assistant.LegalAssistant(answer_timeout=600, debug=True)
    a.warmup()
    with open(os.environ["GOLDEN_QUESTIONS"], encoding="utf-8", newline="") as f:
        qs = [r["question"] for r in csv.DictReader(f)]
    answers = []
    for q in qs:
        r = a.answer_text(q)
        answers.append(strip_times(r.to_dict() if hasattr(r, "to_dict") else r))
    res = {"answers": answers, "chat": proxy.chat_calls, "rerank": proxy.rerank_calls,
           "embed": sorted(proxy.embed_calls), "texts": dict(sorted(proxy.texts.items())),
           "documents": strip_times(a.list_documents())}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
    os._exit(0)  # пул фасада не ждём


# ---------------------------------------------------------------------------- запуск
def runs() -> list[tuple[str, str, dict]]:
    out = []
    for cname, cenv in CONFIGS.items():
        env = dict(PROD_DEFAULTS, **cenv)
        for sc in (ALT_SCENARIOS if cname == "alt" else SCENARIOS):
            out.append((f"{cname}_{sc}", "batch:" + sc, env))
        out.append((f"{cname}_web", "web", env))
    return out


def run_one(kind: str, env_extra: dict, qfile: Path, out: Path) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SOLUTION_")}
    env.update(env_extra, PYTHONHASHSEED="0", GOLDEN_QUESTIONS=str(qfile))
    args = [sys.executable, __file__, "_child", kind, str(out)]
    proc = subprocess.run(args, env=env, capture_output=True, text=True, timeout=1200, cwd=str(ROOT))
    if proc.returncode != 0 or not out.is_file():
        raise SystemExit(f"{kind}: прогон упал\n{proc.stderr[-3000:]}")


def record(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        qfile = questions_file(Path(tmp))
        for name, kind, env in runs():
            run_one(kind, env, qfile, dest / f"{name}.json")
            print("записан", name, flush=True)


def first_diff(a, b, path="") -> str | None:
    if type(a) is not type(b):
        return f"{path}: тип {type(a).__name__} → {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}/{k}: ключ {'добавлен' if k not in a else 'исчез'}"
            d = first_diff(a[k], b[k], f"{path}/{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: длина {len(a)} → {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {str(a)[:200]!r} → {str(b)[:200]!r}"


def check(ref: Path) -> int:
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        qfile = questions_file(Path(tmp))
        for name, kind, env in runs():
            out = Path(tmp) / f"{name}.json"
            run_one(kind, env, qfile, out)
            a = json.loads((ref / f"{name}.json").read_text(encoding="utf-8"))
            b = json.loads(out.read_text(encoding="utf-8"))
            d = first_diff(a, b)
            print(("OK  " if d is None else "DIFF") + f" {name}" + ("" if d is None else f"  {d}"), flush=True)
            bad += d is not None
    print("совпадает" if not bad else f"расхождений: {bad}")
    return 1 if bad else 0


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "_child":
        kind, out = sys.argv[2], sys.argv[3]
        if kind == "web":
            child_web(out)
        else:
            child_batch(kind.split(":", 1)[1], out)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["record", "check"])
    ap.add_argument("--out")
    ap.add_argument("--ref")
    a = ap.parse_args()
    if a.cmd == "record":
        record(Path(a.out or "fp"))
    else:
        sys.exit(check(Path(a.ref)))


if __name__ == "__main__":
    main()
