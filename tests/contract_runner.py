"""Прогон фасада assistant.LegalAssistant на демо-вопросах с заглушкой прокси (отдельный процесс).

    python tests/contract_runner.py --scenario normal --out res.json [--mode dialogs|text]

Сценарии: normal (как golden master), broken (post_json бросает произвольное исключение), slow (ответ дольше
тайм-аута фасада). Результат: ответы фасада (to_dict), каталог документов и все документы.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import golden_runner  # noqa: E402
import harness  # noqa: E402


def platform_to_messages(q: str) -> list[dict]:
    """Строка платформы → история чата (для проверки обратного преобразования фасада)."""
    sys.path.insert(0, str(harness.ROOT))
    from dialog import parse_dialog
    current, history = parse_dialog(q)
    if not history:
        return [{"role": "user", "content": q}]
    return [{"role": r, "content": t} for r, t in history] + [{"role": "user", "content": current}]


def run(scenario: str, mode: str) -> dict:
    work = Path(tempfile.mkdtemp(prefix="contract_"))
    try:
        data = work / "data"
        data.mkdir()
        shutil.copy(harness.DATA / "knowledge-base.zip", data / "knowledge-base.zip")
        os.chdir(work)
        core = harness.load_core("root", {"LLM_BASE_URL": "http://stub.invalid/v1", "LLM_API_KEY": "test-key",
                                          "LLM_MODEL": "stub-model", "DATA_DIR": str(data), "SOLUTION_MODE": "web"})
        import assistant
        proxy = golden_runner.FakeProxy(core, "normal")
        if scenario == "broken":
            def boom(*a, **k):
                raise RuntimeError("сбой транспорта")
            harness.patch_everywhere(core, "post_json", boom)
        elif scenario == "slow":
            def slow(*a, **k):
                time.sleep(3)
                return proxy(*a, **k)
            harness.patch_everywhere(core, "post_json", slow)
        else:
            harness.patch_everywhere(core, "post_json", proxy)
        bot = assistant.LegalAssistant(answer_timeout=1.0 if scenario == "slow" else 60.0)
        health = bot.warmup()
        with open(harness.DATA / "demo_questions.csv", encoding="utf-8-sig", newline="") as f:
            demo = [(r["id"], r["question"]) for r in csv.DictReader(f)]
        if scenario == "slow":
            demo = demo[:1]
        answers, stages, sent = [], {}, {}
        for qid, q in demo:
            seen: list[str] = []
            n_before = len(proxy.chat_calls)
            if mode == "dialogs":
                res = bot.answer(platform_to_messages(q), on_stage=seen.append)
            else:
                res = bot.answer_text(q, on_stage=seen.append)
            answers.append([qid, res.to_dict()])
            stages[qid] = seen
            calls = proxy.chat_calls[n_before:]
            if calls:  # сообщение пользователя первого вызова чата по этому вопросу
                sent[qid] = proxy.texts[calls[0]["messages"][-1][1]]
        docs = bot.list_documents()
        full = {d["doc_id"]: bot.get_document(d["doc_id"]) for d in docs}
        extra = {"empty": bot.answer([]).to_dict(), "unknown_doc": bot.get_document("нет-такого"),
                 "to_platform": [bot.to_platform_text(platform_to_messages(q)) == q for _, q in demo]}
        return {"health": health, "answers": answers, "stages": stages, "sent": sent, "documents": docs, "full": full,
                "extra": extra, "health_after": bot.health()}
    finally:
        os.chdir(harness.ROOT)
        shutil.rmtree(work, ignore_errors=True)


def run_subprocess(scenario: str, mode: str = "dialogs") -> dict:
    import subprocess
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out = tmp.name
    try:
        env = dict(os.environ, PYTHONHASHSEED="0")
        proc = subprocess.run([sys.executable, __file__, "--scenario", scenario, "--mode", mode, "--out", out],
                              env=env, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"contract_runner {scenario} упал:\n{proc.stderr[-4000:]}")
        with open(out, encoding="utf-8") as f:
            res = json.load(f)
        res["_stderr"] = proc.stderr
        return res
    finally:
        os.unlink(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["normal", "broken", "slow"], required=True)
    ap.add_argument("--mode", choices=["dialogs", "text"], default="dialogs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = run(args.scenario, args.mode)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    os._exit(0)  # фоновые потоки тайм-аутов не ждём


if __name__ == "__main__":
    main()
