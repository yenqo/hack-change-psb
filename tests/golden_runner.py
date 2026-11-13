"""Сквозной прогон `main()` на демо-вопросах с подменённым HTTP-слоем (golden master).

Запускается отдельным процессом (чистое глобальное состояние ядра, PYTHONHASHSEED=0):
    python tests/golden_runner.py --target baseline --scenario normal --out run.json

Записывает точные тексты всех сообщений, отправленных в чат, параметры вызовов, запросы к реранкеру
и эмбеддеру, итоговые ответы из submission.csv и счётчики ядра.

Сценарии заглушки:
  normal          — всё отвечает; часть ответов с markdown, обрезкой по длине и пустым ответом при лимите токенов
  fabricated      — в черновиках выдуманные числа и даты; исправление помогает не всегда
  reject_full     — 4xx на полный запрос → компактный режим
  reject_compact  — 4xx на полный и компактный → минимальный режим
  tiny            — 4xx на всё, кроме самого короткого запроса
  transient       — 429 и тайм-ауты чата на первой попытке части вопросов; тайм-аут реранкера
  no_dense        — эмбеддинги недоступны (5xx), реранкер отвечает 404
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

SCENARIOS = ["normal", "fabricated", "reject_full", "reject_compact", "tiny", "transient", "no_dense"]
# Сценарии сбоев для проверки устойчивости (B5; без эталонов, только для tools/fault_eval.py и тестов, цель root):
#   flaky4xx — случайный 400 «прокси перегружен» с вероятностью FAULT_P на каждый вызов чата;
#   flaky_mix — то же плюс случайные 429 и 503;
#   size400 — 400 «maximum context length» на полный запрос (настоящий «слишком велик»).
#   rerank_slow — реранкер отвечает ~7 с; rerank_timeout — реранкер с вероятностью SIM_RERANK_TIMEOUT_P висит до тайм-аута.
FAULT_SCENARIOS = ["flaky4xx", "flaky_mix", "size400", "rerank_slow", "rerank_timeout"]

# Пороги размера запроса для сценариев с отказом 4xx (символы сообщения пользователя).
# На демо-вопросах полный запрос 21–31 тыс., компактный 10,7–15,5 тыс., минимальный 5,7–8,7 тыс.
FULL_LIMIT = 20000
COMPACT_LIMIT = 10000
PROBE = "Ответь одним словом: готов."
PROBE_RERANK_QUERY = "налог на прибыль"
FABRICATED = " Срок рассмотрения составляет 37 рабочих дней. Штраф — 123 456 рублей. Документ подписан 31.12.2019."


class FakeProxy:
    def __init__(self, core, scenario: str) -> None:
        self.core = core
        self.scenario = scenario
        self.lock = threading.Lock()
        self.texts: dict[str, str] = {}
        self.chat_calls: list[dict] = []
        self.rerank_calls: list[dict] = []
        self.embed_calls: list[str] = []
        self.failed_once: set[str] = set()
        self.rerank_seen = 0
        import random
        self.rng = random.Random(int(os.environ.get("FAULT_SEED", "1")))
        self.sim_rng = random.Random(int(os.environ.get("FAULT_SEED", "1")) + 100)
        self.sim_slow = float(os.environ.get("SIM_SLOW", "1"))
        self.sim_rerank_timeout_p = float(os.environ.get("SIM_RERANK_TIMEOUT_P", "0.3"))
        self.clock = None
        self.fault_p = float(os.environ.get("FAULT_P", "0.1"))
        self.probe_rerank = 0

    # -- запись -------------------------------------------------------------
    def _keep(self, text: str) -> str:
        h = harness.sha(text)
        self.texts[h] = text
        return h

    # -- точка подмены post_json -------------------------------------------
    def __call__(self, path: str, key: str, payload: dict, timeout: float = 90.0, attempts: int = 3,
                 max_wait: float = 60.0) -> dict:
        assert key == "test-key", "ядро должно передавать ключ из окружения"
        if self.clock is not None:
            lat, fault = self._sim_latency(path, payload, timeout)
            self.clock.advance(lat)
            if fault is not None:
                raise fault
        with self.lock:
            if path == "/chat/completions":
                return self._chat(payload, timeout, attempts)
            if path == "/embeddings":
                return self._embed(payload)
            if path == "/rerank":
                return self._rerank(payload)
        raise AssertionError(f"неожиданный путь {path}")

    def _sim_latency(self, path: str, payload: dict, timeout: float):
        """Модель задержек (виртуальные секунды): полный запрос ≈ 8 c, меньше — быстрее; SIM_SLOW — замедление."""
        jitter = self.sim_rng.uniform(0.75, 1.35)
        if path == "/chat/completions":
            msgs = payload["messages"]
            if len(msgs) == 1 and msgs[0]["content"] == PROBE:
                return 2.0 * self.sim_slow, None
            size = sum(len(m["content"]) for m in msgs[1:])
            lat = (1.5 + 2.6 * size / 10000) * self.sim_slow * jitter
            if lat > timeout:
                return timeout, self._err("ReadTimeout /chat/completions", transient=True)
            return lat, None
        if path == "/embeddings":
            return 0.4 * jitter, None
        if path == "/rerank":
            if self.scenario == "rerank_slow":
                return 7.0 * jitter, None
            if self.scenario == "rerank_timeout" and self.sim_rng.random() < self.sim_rerank_timeout_p:
                return 12.0, self._err("ReadTimeout /rerank", transient=True)
            return 1.0 * jitter, None
        return 0.0, None

    def _err(self, msg: str, transient: bool):
        return self.core.APIError(msg, transient=transient)

    # -- чат ----------------------------------------------------------------
    def _chat(self, payload: dict, timeout: float, attempts: int) -> dict:
        messages = payload["messages"]
        self.chat_calls.append({
            "messages": [[m["role"], self._keep(m["content"])] for m in messages],
            "model": payload.get("model"), "temperature": payload.get("temperature"),
            "max_tokens": payload.get("max_tokens"), "timeout": timeout, "attempts": attempts,
            "keys": sorted(payload.keys()),
        })
        h = harness.sha(json.dumps(messages, ensure_ascii=False, sort_keys=True))
        n = int(h[:8], 16)
        is_probe = len(messages) == 1 and messages[0]["content"] == PROBE
        if is_probe:
            return self._reply("Готов.")
        user = messages[1]["content"] if len(messages) > 1 else messages[0]["content"]
        sc = self.scenario
        if sc == "reject_full" and len(user) > FULL_LIMIT:
            raise self._err("HTTP 400 /chat/completions", transient=False)
        if sc == "reject_compact" and len(user) > COMPACT_LIMIT:
            raise self._err("HTTP 400 /chat/completions", transient=False)
        if sc == "tiny" and messages[0]["content"] != self.core.SYSTEM_TINY:
            raise self._err("HTTP 413 /chat/completions", transient=False)
        if sc in ("flaky4xx", "flaky_mix"):
            x = self.rng.random()
            if x < self.fault_p:
                raise self.core.APIError("HTTP 400 /chat/completions", transient=False, status=400,
                                         detail='{"error": {"message": "upstream overloaded, please retry"}}')
            if sc == "flaky_mix" and x < 2 * self.fault_p:
                raise self._err("HTTP 429 /chat/completions" if x < 1.5 * self.fault_p else "HTTP 503 /chat/completions",
                                transient=True)
        if sc == "size400" and len(user) > FULL_LIMIT:
            raise self.core.APIError("HTTP 400 /chat/completions", transient=False, status=400,
                                     detail='{"error": {"message": "This model\'s maximum context length is 16384 tokens"}}')
        if sc == "transient" and n % 5 in (0, 1) and h not in self.failed_once:
            self.failed_once.add(h)
            raise self._err("HTTP 429 /chat/completions" if n % 5 == 0 else "ReadTimeout /chat/completions",
                            transient=True)
        if len(messages) == 4:  # запрос на исправление после проверки чисел
            if n % 2 == 0:
                return self._reply(self._compose(messages[:2]))
            return self._reply(messages[2]["content"])
        if sc == "normal" and n % 13 == 2 and payload.get("max_tokens") == self.core.MAX_TOKENS:
            return self._reply("", finish="length")
        text = self._compose(messages)
        if sc == "fabricated":
            text += FABRICATED
        if sc == "normal" and n % 6 == 1:
            return self._reply(text + " Незаконченное предложение без точки", finish="length")
        return self._reply(text)

    @staticmethod
    def _reply(content: str, finish: str = "stop") -> dict:
        return {"choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": finish}]}

    @staticmethod
    def _compose(messages: list[dict]) -> str:
        """Ответ из строк самого запроса: числа и даты в нём подтверждаются контекстом."""
        h = harness.sha(json.dumps(messages, ensure_ascii=False, sort_keys=True))
        n = int(h[:8], 16)
        user = messages[-1]["content"]
        lines = [ln.strip() for ln in user.split("\n")
                 if 40 <= len(ln.strip()) <= 260 and not ln.strip().startswith(("===", "[", "<"))]
        if not lines:
            return f"Ответ {h[:6]}: в приложенных документах сведений по вопросу нет, обратитесь к юристу."
        a, b = lines[n % len(lines)], lines[(n // 7) % len(lines)]
        text = f"Ответ {h[:6]}. {a} {b}"
        if n % 5 == 0:
            text = "**Ответ:** " + text
        return text

    # -- эмбеддинги и реранкер ---------------------------------------------
    def _embed(self, payload: dict) -> dict:
        texts = payload["input"]
        self.embed_calls.append(harness.sha(json.dumps([payload.get("model"), texts], ensure_ascii=False)))
        if self.scenario == "no_dense":
            raise self._err("HTTP 503 /embeddings", transient=True)
        return {"data": [{"index": i, "embedding": harness.fake_vector(t)} for i, t in enumerate(texts)]}

    def _rerank(self, payload: dict) -> dict:
        q, docs, top_n = payload["query"], payload["documents"], payload["top_n"]
        if q == PROBE_RERANK_QUERY:  # пробный вызов PLATFORM PROBE (7.5): отвечаем, в сценарии не учитываем
            self.probe_rerank += 1
            scored = sorted(((i, harness.fake_rerank_score(q, d)) for i, d in enumerate(docs)), key=lambda x: -x[1])
            return {"id": "stub", "results": [{"index": i, "relevance_score": s} for i, s in scored[:top_n]]}
        self.rerank_calls.append({"query": self._keep(q), "docs": harness.sha(json.dumps(docs, ensure_ascii=False)),
                                  "n_docs": len(docs), "top_n": top_n, "model": payload.get("model")})
        self.rerank_seen += 1
        if self.scenario == "transient" and self.rerank_seen == 1:
            raise self._err("ReadTimeout /rerank", transient=True)
        if self.scenario == "no_dense":
            raise self._err("HTTP 404 /rerank", transient=False)
        scored = sorted(((i, harness.fake_rerank_score(q, d)) for i, d in enumerate(docs)), key=lambda x: -x[1])
        return {"id": "stub", "results": [{"index": i, "relevance_score": s} for i, s in scored[:top_n]]}


def run(which: str, scenario: str) -> dict:
    work = Path(tempfile.mkdtemp(prefix="golden_"))
    try:
        data = work / "data"
        data.mkdir()
        sim = os.environ.get("SIM") == "1"
        extra = {"LLM_BASE_URL": "http://stub.invalid/v1", "LLM_API_KEY": "test-key", "LLM_MODEL": "stub-model"}
        clock = None
        if sim:
            # симуляция долгого прогона: N вопросов, MAX_WORKERS, бюджет как на платформе, виртуальное время
            n_q = int(os.environ.get("SIM_N", "18"))
            with open(harness.DATA / "demo_questions.csv", encoding="utf-8-sig", newline="") as f:
                demo = [r["question"] for r in csv.DictReader(f)]
            with open(data / "questions.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "question"])
                for i in range(n_q):
                    q = demo[i % len(demo)]
                    w.writerow([str(i + 1), q if i < len(demo) else f"{q} [{i // len(demo)}]"])
            extra.update({"MAX_WORKERS": os.environ.get("SIM_WORKERS", "2"), "SOLUTION_DEBUG": "1",
                          "SOLUTION_TIME_BUDGET": os.environ.get("SIM_BUDGET", str(16 * 60 + 30))})
            import simclock
            clock = simclock.SimClock()
            clock.install()
        else:
            # GOLDEN_QUESTIONS — свой набор вопросов (tools/fingerprint.py); по умолчанию демо
            shutil.copy(os.environ.get("GOLDEN_QUESTIONS") or harness.DATA / "demo_questions.csv",
                        data / "questions.csv")
        shutil.copy(harness.DATA / "knowledge-base.zip", data / "knowledge-base.zip")
        harness.TEST_ENV["DATA_DIR"] = str(data)
        os.chdir(work)
        core = harness.load_core(which, extra)
        proxy = FakeProxy(core, scenario)
        proxy.clock = clock
        harness.patch_everywhere(core, "post_json", proxy)
        t0 = time.monotonic()
        core.main()
        elapsed = time.monotonic() - t0
        with open(work / "submission.csv", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        flags = core.FLAGS
        return {
            "scenario": scenario,
            "header": rows[0],
            "answers": [[r[0], r[1]] for r in rows[1:]],
            "chat": proxy.chat_calls,
            "rerank": proxy.rerank_calls,
            "embed": sorted(proxy.embed_calls),
            "stats": dict(sorted(flags.stats.items())),
            "flags": {"compact": flags.compact, "minimal": flags.minimal, "tiny": flags.tiny,
                      "rerank_off": flags.rerank_off, "dense": flags.dense},
            "texts": dict(sorted(proxy.texts.items())),
            "probe_rerank_calls": proxy.probe_rerank,
            "sim": sim_summary(work, rows, clock) if clock is not None else None,
            "_elapsed": round(elapsed, 1),
        }
    finally:
        os.chdir(harness.ROOT)
        shutil.rmtree(work, ignore_errors=True)


def sim_summary(work: Path, rows: list, clock) -> dict:
    """Итог симуляции: виртуальное время и уровень, на котором дан итоговый ответ на каждый вопрос."""
    by_q: dict[str, list] = {}
    dbg = work / "debug.jsonl"
    if dbg.is_file():
        for line in dbg.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            by_q.setdefault(rec["question"], []).append(rec)
    with open(work / "data" / "questions.csv", encoding="utf-8", newline="") as f:
        qtext = {r["id"]: r["question"] for r in csv.DictReader(f)}
    levels = []
    for qid, ans in rows[1:]:
        if ans.startswith("По данным базы знаний") or ans.startswith("В базе знаний не найдено") or \
                ans.startswith("Не удалось"):
            levels.append("fallback")
            continue
        recs = [r for r in by_q.get(qtext[qid], []) if r.get("final") == ans]
        levels.append(str(recs[-1]["level"]) if recs else "0")
    dist = {k: levels.count(k) for k in ("4", "3", "2", "1", "0", "fallback")}
    return {"elapsed": round(clock.elapsed(), 1), "levels": dist, "n": len(levels)}


def run_subprocess(which: str, scenario: str, env_extra: dict | None = None) -> dict:
    """Прогон в отдельном процессе с фиксированным PYTHONHASHSEED. В результат добавляется stderr прогона."""
    import subprocess
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out = tmp.name
    try:
        env = dict(os.environ, PYTHONHASHSEED="0", **(env_extra or {}))
        proc = subprocess.run([sys.executable, __file__, "--target", which, "--scenario", scenario, "--out", out],
                              env=env, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"golden_runner {which}/{scenario} упал:\n{proc.stderr[-4000:]}")
        with open(out, encoding="utf-8") as f:
            res = json.load(f)
        res["_stderr"] = proc.stderr
        return res
    finally:
        os.unlink(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["root", "baseline"], required=True)
    ap.add_argument("--scenario", choices=SCENARIOS + FAULT_SCENARIOS, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = run(args.target, args.scenario)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
