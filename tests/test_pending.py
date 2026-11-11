"""Доработки, ждущие валидации на golden set (eval/experiments.md, «Отложенная валидация»): 7.6, B3, 7.1.

По умолчанию все флаги выключены — поведение v1.3 проверяет golden master. Здесь — что включённые флаги делают то,
что задумано, и не ломают прогон. Только для CORE_TARGET=root.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import unittest

import golden_runner
import harness


def run_py(code: str, env: dict) -> str:
    out = subprocess.run([sys.executable, "-c", code], cwd=harness.ROOT, capture_output=True, text=True,
                         env=dict(os.environ, **env), timeout=120)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    return out.stdout.strip()


@unittest.skipIf(harness.target() == "baseline", "доработок нет в v1.3")
class PendingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.core = harness.load_core()
        with contextlib.redirect_stderr(io.StringIO()):
            cls.kb = cls.core.build_kb()
        cls.kb_text = "\n".join(d.text for d in cls.kb.docs)

    # --- 7.6 few-shot -------------------------------------------------------
    def test_fictional_fewshot_disjoint_from_kb(self) -> None:
        fs = self.core.FEW_SHOT_FICTIONAL
        self.assertEqual(len(fs), len(self.core.FEW_SHOT_DEMO))
        text = "\n".join(q + "\n" + a for q, a in fs)
        for name in set(re.findall(r"«([^»]+)»", text)):
            self.assertNotIn(name.lower(), self.kb_text.lower(), f"название из базы: {name}")
        for code in set(re.findall(r"[А-ЯЁ]{1,3}-\d+(?:/\d+)?", text)):
            self.assertNotIn(code, self.kb_text, f"код из базы: {code}")
        items, dates = self.core.extract_numbers(text)
        kitems, kdates = self.core.extract_numbers(self.kb_text)
        kv = {(round(v, 6), u) for v, _, u in kitems}
        clash = [(s, u) for v, s, u in items
                 if u in ("rub", "pct", "dom", "cday", "wday", "month") and (round(v, 6), u) in kv]
        self.assertEqual(clash, [], "суммы, проценты и сроки примеров не должны совпадать с базой")
        self.assertFalse(dates & kdates)

    def test_fewshot_flag(self) -> None:
        code = ("import prompts; t = prompts.SYSTEM_TEXT + prompts.SYSTEM_TEXT_SHORT + prompts.SYSTEM_TEXT_MINI; "
                "print('Кедр' in t, any(x in t for x in ('Северный Склад', 'А-08', 'ст. 70', 'Альфа-Трейд', 'ЮО-7')))")
        self.assertEqual(run_py(code, {"SOLUTION_FEWSHOT": "fictional"}), "True False")
        self.assertEqual(run_py(code, {"SOLUTION_FEWSHOT": "demo"}), "False True")

    def test_concise_flag(self) -> None:
        code = ("import prompts; t = prompts.SYSTEM_TEXT; print('20–60 слов' in t, '30–120 слов' in t, "
                "'Отвечай только на заданный вопрос' in t, 'Кедр' in t)")
        self.assertEqual(run_py(code, {"SOLUTION_CONCISE": "1"}), "True False True False")
        self.assertEqual(run_py(code, {"SOLUTION_CONCISE": "0"}), "False True False False")
        self.assertEqual(run_py(code, {"SOLUTION_CONCISE": "1", "SOLUTION_FEWSHOT": "fictional"}), "True False True True")

    # --- B3 -----------------------------------------------------------------
    def test_b3_flag(self) -> None:
        code = ("import checks; print([v for v, _, _ in checks.extract_numbers("
                "'По договору ПТ-05/2023 — 45 календарных дней.', answer=True)[0]])")
        self.assertEqual(run_py(code, {"SOLUTION_FIX_DOCNUM": "1"}), "[45.0]")
        self.assertIn("5.0", run_py(code, {"SOLUTION_FIX_DOCNUM": "0"}))

    # --- 7.1 проверка ссылок --------------------------------------------------
    def refs(self, answer: str, q: str = "вопрос") -> list[str]:
        return [s for s, _ in self.core.unverified_refs(self.kb, answer, q, list(self.kb.docs))]

    def test_refs_correct(self) -> None:
        ok = [
            "Арендная плата по договору аренды № А-08/2024 — 450 000 рублей (п. 3.1), не позднее 5-го числа (п. 3.2).",
            "Срок испытания — не более шести месяцев (ТК РФ, ст. 70).",
            "Оператор вправе продолжить обработку при основаниях из п. 2–11 ч. 1 ст. 6 (152-ФЗ, ст. 9, ч. 2).",
            "По письму № К-12 от 15.04.2025 нужна санкционная проверка; см. п. 5.4 Регламента ред. 2025.",
            "Новый адрес — по дополнительному соглашению № 1 к договору № П-17/2024.",
            "Ставка 10% — НК РФ, ст. 164, п. 2, пп. 1.",
        ]
        for a in ok:
            self.assertEqual(self.refs(a), [], a)

    def test_refs_fabricated(self) -> None:
        cases = {
            "Арендная плата — 450 000 рублей по договору № А-08/2024 (п. 3.9).": "п. 3.9",
            "Порядок установлен приказом № 77-П от 01.01.2024.": "77-П",
            "Срок установлен ТК РФ, ст. 470.": "ТК РФ, ст. 470",
            "См. договор № А-09/2024, п. 3.1.": "А-09/2024",
            "Ставка по НК РФ, ст. 164, п. 2, пп. 77.": "пп. 77",
        }
        for a, span in cases.items():
            got = self.refs(a)
            self.assertTrue(any(span in g for g in got), f"{a} → {got}")

    def test_drop_ref_sentences(self) -> None:
        a = ("Арендная плата составляет 450 000 рублей в месяц по договору аренды (п. 3.1). "
             "Пени начисляются по п. 9.9 договора.")
        self.assertEqual(self.core.drop_ref_sentences(a, ["п. 9.9"]),
                         "Арендная плата составляет 450 000 рублей в месяц по договору аренды (п. 3.1).")

    # --- прогон с включёнными флагами ------------------------------------------
    def test_run_with_all_flags(self) -> None:
        env = {"SOLUTION_FEWSHOT": "fictional", "SOLUTION_FIX_DOCNUM": "1", "SOLUTION_CHECK_REFS": "1"}
        for sc in ("normal", "fabricated"):
            res = golden_runner.run_subprocess("root", sc, env)
            self.assertEqual(len(res["answers"]), 18, sc)
            self.assertTrue(all(a.strip() for _, a in res["answers"]), sc)
            self.assertEqual(res["stats"]["level4"], 18, sc)
            system = res["texts"][res["chat"][1]["messages"][0][1]]
            self.assertIn("Кедр-Логистик", system)


@unittest.skipIf(harness.target() == "baseline", "нет в v1.3")
class SoftDegradeTest(unittest.TestCase):
    """B5: разовые 4xx не понижают уровень на весь прогон; настоящий «слишком велик» — понижает сразу."""

    def run_sc(self, sc: str, soft: bool, **extra) -> dict:
        env = {"SOLUTION_SOFT_DEGRADE": "1" if soft else "0", **{k: str(v) for k, v in extra.items()}}
        return golden_runner.run_subprocess("root", sc, env)

    def test_normal_unchanged(self) -> None:
        self.assertEqual(self.run_sc("normal", True)["answers"], self.run_sc("normal", False)["answers"])

    def test_flaky_keeps_full_level(self) -> None:
        old = self.run_sc("flaky4xx", False, FAULT_P=0.1, FAULT_SEED=1)["stats"].get("level4", 0)
        new = self.run_sc("flaky4xx", True, FAULT_P=0.1, FAULT_SEED=1)["stats"].get("level4", 0)
        self.assertGreaterEqual(new, 15)
        self.assertGreater(new, old)

    def test_size_error_degrades_at_once(self) -> None:
        res = self.run_sc("size400", True)
        self.assertTrue(res["flags"]["compact"])
        self.assertEqual(res["stats"].get("level4", 0), 0)


@unittest.skipIf(harness.target() == "baseline", "нет в v1.3")
class RerankRecheckTest(unittest.TestCase):
    """B6: реранкер после тайм-аутов отключается временно; после неустранимых ошибок — навсегда."""

    def run_sc(self, sc: str, on: bool) -> dict:
        return golden_runner.run_subprocess("root", sc, {"SIM": "1", "SIM_N": "30", "SIM_WORKERS": "2",
                                                         "SOLUTION_RERANK_RECHECK": "1" if on else "0"})

    def test_timeouts_temporary(self) -> None:
        old = self.run_sc("rerank_timeout", False)["stats"].get("rerank_ok", 0)
        new = self.run_sc("rerank_timeout", True)["stats"].get("rerank_ok", 0)
        self.assertGreater(new, old)

    def test_permanent_errors_stay_off(self) -> None:
        res = golden_runner.run_subprocess("root", "no_dense", {"SOLUTION_RERANK_RECHECK": "1"})
        self.assertTrue(res["flags"]["rerank_off"])
        self.assertEqual(len(res["rerank"]), 2)   # как в v1.3: две ошибки 404 — и больше не вызывается


@unittest.skipIf(harness.target() == "baseline", "нет в v1.3")
class TimeBudgetTest(unittest.TestCase):
    """B7 (уровень по прогнозу) и B8 (финальный проход): больше ответов на полном уровне в пределах бюджета."""

    def sim(self, n: int, w: int, **flags) -> dict:
        env = {"SIM": "1", "SIM_N": str(n), "SIM_WORKERS": str(w), **{k: str(v) for k, v in flags.items()}}
        res = golden_runner.run_subprocess("root", "normal", env)
        return res["sim"]

    def test_forecast_uses_time(self) -> None:
        old = self.sim(100, 1)
        new = self.sim(100, 1, SOLUTION_LEVEL_FORECAST=1)
        self.assertGreater(new["levels"]["4"], old["levels"]["4"])
        self.assertLess(new["elapsed"], 990)

    def test_final_pass_upgrades(self) -> None:
        old = self.sim(200, 2, SOLUTION_LEVEL_FORECAST=1)
        new = self.sim(200, 2, SOLUTION_LEVEL_FORECAST=1, SOLUTION_FINAL_PASS=1)
        self.assertGreaterEqual(new["levels"]["4"], old["levels"]["4"])
        self.assertLessEqual(new["levels"]["fallback"], old["levels"]["fallback"])
        self.assertLess(new["elapsed"], 1100)   # лимит платформы — 1200 c

    def test_run_stats_levels(self) -> None:
        import re
        res = golden_runner.run_subprocess("root", "normal", {})
        m = re.search(r"=== RUN STATS ===\n(.*?)\n=== /RUN STATS ===", res["_stderr"], re.S)
        stats = json.loads(m.group(1))
        self.assertEqual(stats["answer_levels"]["4"], 18)
        self.assertIn("soft4xx", stats["triggers"])


if __name__ == "__main__":
    unittest.main()
