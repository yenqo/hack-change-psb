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
    maxDiff = None

    def compare(self, scenario: str) -> None:
        exp = load_fixture(scenario)
        got = golden_runner.run_subprocess(harness.target(), scenario)
        got.pop("_elapsed", None)
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
