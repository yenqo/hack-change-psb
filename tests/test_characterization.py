"""Фиксация поведения v1.3 на детерминированных частях (реальная база, без сети).

Цель проверки — CORE_TARGET (root — модули в корне, baseline — tools/baseline/solution_v1_3.py).
"""

from __future__ import annotations

import json
import unittest

import characterization
import harness


class CharacterizationTest(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.core = harness.load_core()
        cls.snap = characterization.snapshot(cls.core)
        with open(harness.FIXTURES / "characterization.json", encoding="utf-8") as f:
            cls.expected = json.load(f)

    def check(self, section: str) -> None:
        got = json.loads(json.dumps(self.snap[section], ensure_ascii=False))
        self.assertEqual(got, self.expected[section], f"раздел {section} отличается от v1.3")

    # --- явные инварианты базы -------------------------------------------
    def test_kb_counts(self) -> None:
        docs = self.snap["kb_docs"]
        self.assertEqual(sum(d["kind"] == "company" for d in docs), 8)
        self.assertEqual(sum(d["kind"] == "law" for d in docs), 13)
        self.assertEqual(len(self.snap["kb_chunks"]), 336)

    def test_kb_statuses(self) -> None:
        status = {d["label"]: d["status"] for d in self.snap["kb_docs"] if d["kind"] == "company"}
        old = next(v for k, v in status.items() if k.startswith("Регламент") and "2023" in k)
        new = next(v for k, v in status.items() if k.startswith("Регламент") and "2025" in k)
        self.assertIn("УТРАТИЛ СИЛУ с 01.03.2025", old)
        self.assertIn("действующая редакция, действует с 01.03.2025", new)
        self.assertEqual(sum(bool(v) for v in status.values()), 2, status)

    def test_injection_removed(self) -> None:
        removed = {d["label"]: d["removed"] for d in self.snap["kb_docs"] if d["removed"]}
        self.assertEqual(len(removed), 1, removed)
        label, lines = next(iter(removed.items()))
        self.assertIn("Служебная записка", label)
        self.assertTrue(any("90" in ln for ln in lines))

    # --- сравнение с эталоном v1.3 ----------------------------------------
    def test_kb_docs(self) -> None:
        self.check("kb_docs")

    def test_kb_chunks(self) -> None:
        self.check("kb_chunks")

    def test_related_docs(self) -> None:
        self.check("related_docs")

    def test_date_note(self) -> None:
        self.check("date_note")

    def test_event_periods(self) -> None:
        self.check("event_periods")

    def test_tokenize(self) -> None:
        self.check("tokenize")

    def test_expand_query(self) -> None:
        self.check("expand_query")

    def test_bm25_top10(self) -> None:
        self.check("bm25_top10")

    def test_hybrid_law_top10(self) -> None:
        self.check("hybrid_law_top10")

    def test_select_law(self) -> None:
        self.check("select_law")

    def test_company_context(self) -> None:
        self.check("company_context")

    def test_fallback_answer(self) -> None:
        self.check("fallback_answer")

    def test_parse_dialog(self) -> None:
        self.check("parse_dialog")

    def test_number_checks(self) -> None:
        self.check("number_checks")

    def test_context(self) -> None:
        self.check("context_sha")

    def test_prompts(self) -> None:
        self.check("prompts")

    def test_sanitize(self) -> None:
        self.check("sanitize")

    # --- смысловые проверки числового контроля ----------------------------
    def test_number_check_semantics(self) -> None:
        by_answer = {c["answer"]: c for c in self.snap["number_checks"]}
        spans = lambda ans: [s for s, _ in by_answer[ans]["unverified"]]  # noqa: E731
        ok = characterization.ANSWERS[0][1]
        self.assertEqual(spans(ok), [], "верные числа не должны считаться выдуманными")
        derived = characterization.ANSWERS[2][1]
        self.assertNotIn("481 500", spans(derived), "расчётное число (450 000 × 1,07) подтверждается")
        fabricated = characterization.ANSWERS[7][1]
        self.assertTrue(any(s.startswith("7") for s in spans(fabricated)), "выдуманный срок 7 рабочих дней")
        refs = characterization.ANSWERS[11][1]
        self.assertFalse(any("99" in s or "2099" in s for s in spans(refs)), "номера пунктов и документов не проверяются")


if __name__ == "__main__":
    unittest.main()
