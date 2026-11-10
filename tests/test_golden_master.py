"""Сквозной golden master: 18 демо-вопросов через `main()` с заглушками прокси.

Сообщения в чат, запросы к реранкеру и эмбеддеру и итоговые ответы должны совпасть с v1.3 байт в байт.
Каждый сценарий — отдельный процесс (tests/golden_runner.py).
"""

from __future__ import annotations

import difflib
import gzip
import json
import unittest

import golden_runner
import harness


def load_fixture(scenario: str) -> dict:
    with gzip.open(harness.FIXTURES / f"golden_{scenario}.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


class GoldenMasterTest(unittest.TestCase):
    """Поведение по умолчанию (как на платформе). Пробный вызов /rerank из PLATFORM PROBE (7.5) не входит в v1.3 —
    он исключается из сравнения; всё остальное совпадает байт в байт."""
    maxDiff = None
    ENV: dict = {}

    def compare(self, scenario: str) -> None:
        exp = load_fixture(scenario)
        got = golden_runner.run_subprocess(harness.target(), scenario, self.ENV)
        got.pop("_elapsed", None)
        got.pop("_stderr", None)
        got.pop("probe_rerank_calls", None)
        probe_q = harness.sha(golden_runner.PROBE_RERANK_QUERY)
        got["rerank"] = [r for r in got["rerank"] if r["query"] != probe_q]
        if not any(m[1] == probe_q for c in got["chat"] for m in c["messages"]):
            got["texts"].pop(probe_q, None)
        self.assertEqual(got["header"], ["id", "answer"])
        # Сначала — понятная разница в первом несовпавшем сообщении
        for i, (g, e) in enumerate(zip(got["chat"], exp["chat"])):
            if g == e:
                continue
            for (gr, gh), (er, eh) in zip(g["messages"], e["messages"]):
                if gh != eh:
                    diff = "\n".join(difflib.unified_diff(exp["texts"][eh].splitlines(), got["texts"][gh].splitlines(),
                                                          "v1.3", harness.target(), lineterm="", n=2))
                    self.fail(f"{scenario}: вызов чата №{i}, сообщение {gr} отличается:\n{diff[:6000]}")
            self.assertEqual(g, e, f"{scenario}: параметры вызова чата №{i}")
        self.assertEqual(len(got["chat"]), len(exp["chat"]), f"{scenario}: число вызовов чата")
        for (gid, ga), (eid, ea) in zip(got["answers"], exp["answers"]):
            self.assertEqual((gid, ga), (eid, ea), f"{scenario}: ответ на вопрос {eid}")
        for key in ("answers", "rerank", "embed", "stats", "flags", "texts"):
            self.assertEqual(got[key], exp[key], f"{scenario}: {key}")

    def test_normal(self) -> None:
        self.compare("normal")

    def test_fabricated(self) -> None:
        self.compare("fabricated")

    def test_reject_full(self) -> None:
        self.compare("reject_full")

    def test_reject_compact(self) -> None:
        self.compare("reject_compact")

    def test_tiny(self) -> None:
        self.compare("tiny")

    def test_transient(self) -> None:
        self.compare("transient")

    def test_no_dense(self) -> None:
        self.compare("no_dense")


class GoldenMasterProbeOffTest(GoldenMasterTest):
    """С выключенными блоками probe (SOLUTION_PROBE=0) — строго как v1.3, без пробного вызова."""
    ENV = {"SOLUTION_PROBE": "0"}


class ProbeOutputTest(unittest.TestCase):
    """Блоки PLATFORM PROBE и RUN STATS: есть, это JSON, без ключа и адреса прокси (7.5)."""

    def blocks(self, scenario: str, env: dict | None = None) -> tuple[dict, dict]:
        res = golden_runner.run_subprocess(harness.target(), scenario, env)
        err = res["_stderr"]
        self.assertNotIn("test-key", err)
        self.assertNotIn("stub.invalid", err)
        out = {}
        for title in ("PLATFORM PROBE", "RUN STATS"):
            m = __import__("re").search(rf"=== {title} ===\n(.*?)\n=== /{title} ===", err, __import__("re").S)
            self.assertIsNotNone(m, f"нет блока {title}")
            out[title] = json.loads(m.group(1))
        return out["PLATFORM PROBE"], out["RUN STATS"]

    def test_no_extra_calls_by_default(self) -> None:
        """По умолчанию (как на платформе) probe не делает ни одного лишнего вызова."""
        res = golden_runner.run_subprocess(harness.target(), "normal")
        self.assertEqual(res["probe_rerank_calls"], 0)

    def test_rerank_probe_local(self) -> None:
        if harness.target() == "baseline":
            self.skipTest("в v1.3 блоков нет")
        probe, _ = self.blocks("normal", {"SOLUTION_PROBE_RERANK": "1"})
        self.assertTrue(probe["rerank"]["ok"])
        self.assertEqual(probe["rerank"]["score_field"], "relevance_score")

    def test_blocks_normal(self) -> None:
        if harness.target() == "baseline":
            self.skipTest("в v1.3 блоков нет")
        probe, stats = self.blocks("normal")
        self.assertFalse(probe["rerank"]["checked"])
        self.assertEqual(probe["kb"]["chunks"], 336)
        self.assertEqual(probe["kb"]["company"], 8)
        self.assertEqual(probe["questions"]["n"], 18)
        self.assertEqual(probe["embeddings"]["dim"], 64)
        self.assertTrue(probe["env_present"]["LLM_API_KEY"])
        self.assertEqual(stats["questions"], 18)
        self.assertEqual(stats["stats"]["level4"], 18)
        self.assertIn("usage", stats)  # счётчики вызовов здесь пусты: post_json подменён; см. test_http

    def test_blocks_off(self) -> None:
        res = golden_runner.run_subprocess(harness.target(), "tiny", {"SOLUTION_PROBE": "0"})
        self.assertNotIn("PLATFORM PROBE", res["_stderr"])
        self.assertNotIn("RUN STATS", res["_stderr"])


class SubmissionFormatTest(unittest.TestCase):
    """Формат submission.csv в эталоне: все ID, без дублей, непустые ответы, ≤ 10 000 символов."""

    def test_format(self) -> None:
        ids = [q for q, _ in __import__("characterization").demo_questions()]
        for sc in golden_runner.SCENARIOS:
            answers = load_fixture(sc)["answers"]
            self.assertEqual([a[0] for a in answers], ids, sc)
            for qid, text in answers:
                self.assertTrue(text.strip(), f"{sc}/{qid}: пустой ответ")
                self.assertLessEqual(len(text), 10000, f"{sc}/{qid}")


if __name__ == "__main__":
    unittest.main()
