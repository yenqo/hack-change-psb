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
from unittest import mock

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


@unittest.skipIf(harness.target() == "baseline", "нет в v1.3")
class CrossRefsTest(unittest.TestCase):
    """R6: отсылки внутри законов (SOLUTION_CROSS_REFS)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.core = harness.load_core()
        with contextlib.redirect_stderr(io.StringIO()):
            cls.kb = cls.core.build_kb()
        cls.xr = cls.core.cross_refs(cls.kb)

    def chunk(self, label: str, contains: str = ""):
        return next(c for c in self.kb.chunks if c.label == label and contains in c.text)

    def targets(self, label: str, contains: str = "") -> set[str]:
        c = self.chunk(label, contains)
        return {self.kb.chunks[t].label for r in self.xr.get(c.idx, []) for t in r["targets"]}

    def absent(self, label: str, contains: str = "") -> set[str]:
        c = self.chunk(label, contains)
        return {art for r in self.xr.get(c.idx, []) for _, art in r["absent"]}

    def test_numbers(self) -> None:
        nums = self.core._xr_nums
        self.assertEqual(nums("2 - 11"), [str(i) for i in range(2, 12)])
        self.assertEqual(nums("1.4, 1.6, 1.17-2"), ["1.4", "1.6", "1.17-2"])
        self.assertEqual(nums("2, 3 или 8"), ["2", "3", "8"])

    def test_same_article_refs(self) -> None:
        self.assertIn("КоАП РФ, ст. 13.11, ч. 2", self.targets("КоАП РФ, ст. 13.11, ч. 2.1"))
        self.assertIn("НК РФ, ст. 224, п. 3.1", self.targets("НК РФ, ст. 224, п. 3", "3.1 настоящей статьи"))
        self.assertIn("НК РФ, ст. 284, п. 3, пп. 1",
                      self.targets("НК РФ, ст. 284, п. 3, пп. 3", "Для подтверждения права"))
        self.assertIn("НК РФ, ст. 164, п. 2, пп. 1.1–1.2", self.targets("НК РФ, ст. 164, п. 10"))

    def test_absent_articles(self) -> None:
        label = next(c.label for c in self.kb.chunks if "152-ФЗ" in c.label and c.label.endswith("ч. 2"))
        self.assertEqual(self.absent(label), {"6", "10", "11"})
        self.assertIn("165", self.absent("НК РФ, ст. 164, п. 1, пп. 15–16"))

    def test_company_docs_not_parsed(self) -> None:
        self.assertTrue(all(self.kb.docs[self.kb.chunks[ci].doc].kind == "law" for ci in self.xr))

    def test_expansion_marks_and_context(self) -> None:
        src = self.chunk("КоАП РФ, ст. 13.11, ч. 2.1")
        items = {src.doc: [src]}
        reasons, xref, absent = {src.idx: "search"}, {}, []
        left = self.core._add_cross_refs(self.kb, [src.idx], items, reasons, 12000, xref, absent)
        added = [c for c in items[src.doc] if c is not src]
        self.assertTrue(1 <= len(added) <= 3)
        self.assertEqual(left, 12000 - sum(len(c.text) for c in added))
        self.assertTrue(all(reasons[c.idx] == "cross_ref" and xref[c.idx] == "ч. 2.1" for c in added))
        block = self.core.law_block(self.kb, list(items.items()), xref, [("НК РФ", "165")])
        self.assertIn("<КоАП РФ, ст. 13.11, ч. 2> [упомянут в ч. 2.1]", block)
        self.assertIn("НК РФ — ст. 165]", block)
        # без пометок — блок как раньше
        self.assertNotIn("упомянут", self.core.law_block(self.kb, list(items.items())))

    def test_budget_and_limit(self) -> None:
        src = self.chunk("НК РФ, ст. 224, п. 1")   # ссылается на пп. 1.1, 1.2, 2, 5, 6
        items, xref = {src.doc: [src]}, {}
        self.core._add_cross_refs(self.kb, [src.idx], items, {}, 12000, xref, [])
        self.assertLessEqual(len(xref), 3)
        items, xref = {src.doc: [src]}, {}
        self.core._add_cross_refs(self.kb, [src.idx], items, {}, 50, xref, [])
        self.assertEqual(xref, {})

    def test_run_with_flag(self) -> None:
        res = golden_runner.run_subprocess("root", "normal", {"SOLUTION_CROSS_REFS": "1"})
        self.assertEqual(len(res["answers"]), 18)
        self.assertEqual(res["stats"]["level4"], 18)


@unittest.skipIf(harness.target() == "baseline", "нет в v1.3")
class RewriteTest(unittest.TestCase):
    """R7: переформулирование запроса (SOLUTION_REWRITE)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.core = harness.load_core()
        with contextlib.redirect_stderr(io.StringIO()):
            cls.kb = cls.core.build_kb()
            cls.index = cls.core.Index(cls.kb)

    def test_parse(self) -> None:
        p = self.core._parse_rewrite
        self.assertEqual(p('```json\n{"question": "Срок испытания главбуха?", "queries": ["срок испытания", 5]}\n```'),
                         ("Срок испытания главбуха?", ["срок испытания"]))
        self.assertIsNone(p("не JSON"))
        self.assertIsNone(p('{"queries": ["x"]}'))

    def call(self, current: str, history: list, level: int = 4, reply: str | None = None):
        calls = []

        def fake_chat(messages, max_tokens=0, timeout=0.0):
            calls.append(messages)
            return reply if reply is not None else json.dumps(
                {"question": "Какой максимальный срок испытания главного бухгалтера?",
                 "queries": ["срок испытания главного бухгалтера", "ТК РФ статья 70 испытание"]}, ensure_ascii=False)

        self.core.TIMING.add("chat_full", 5.0)   # переформулирование — только после первых замеров времени
        with mock.patch.object(sys.modules["retrieval"], "chat", fake_chat):
            prep = self.core.Prepared(current, history)
            info: dict = {}
            self.core.maybe_rewrite(self.kb, self.index, prep, level, info)
        return prep, info, calls

    def test_dialog_fires_and_feeds_search(self) -> None:
        prep, info, calls = self.call("Испыталку ему скок максимум?", [("user", "У нас выходит новый главбух."),
                                                                      ("assistant", "Чем помочь?")])
        self.assertEqual(len(calls), 1)
        sys_msg, user_msg = calls[0][0]["content"], calls[0][1]["content"]
        self.assertTrue(sys_msg.startswith("Ты помогаешь поиску"))
        self.assertNotIn("ДОКУМЕНТЫ", user_msg)          # без документов
        self.assertIn("У нас выходит новый главбух.", user_msg)
        self.assertEqual(prep.standalone, "Какой максимальный срок испытания главного бухгалтера?")
        self.assertEqual([q for q, _ in prep.extra][0], prep.standalone)
        self.assertEqual(info["rewrite"]["queries"], ["срок испытания главного бухгалтера", "ТК РФ статья 70 испытание"])
        ranked = self.core._search(self.index, prep.current, prep, {"law"})
        self.assertEqual(self.kb.docs[self.kb.chunks[ranked[0]].doc].label, "ТК РФ, ст. 70")
        msg = self.core.build_user_message(self.kb, [], [], "", prep.current, prep.history, standalone=prep.standalone)
        self.assertLess(msg.index("ВОПРОС СВОИМИ СЛОВАМИ"), msg.index("=== ВОПРОС ПОЛЬЗОВАТЕЛЯ"))
        # второй раз (финальный проход) — без нового вызова
        n = len(calls)
        self.core.maybe_rewrite(self.kb, self.index, prep, 4, {})
        self.assertEqual(len(calls), n)

    def test_waits_for_timing(self) -> None:
        self.core.TIMING.drop("chat_full")
        calls = []
        harness.patch_everywhere(self.core, "chat", lambda *a, **k: calls.append(1) or "{}")
        try:
            self.core.maybe_rewrite(self.kb, self.index, self.core.Prepared("Испыталку скок?", [("user", "главбух")]), 4)
        finally:
            harness.patch_everywhere(self.core, "chat", self.core._modules[[m.__name__ for m in self.core._modules]
                                                                            .index("llm")].chat)
        self.assertEqual(calls, [])

    def test_skips(self) -> None:
        _, _, calls = self.call("Испыталку скок?", [("user", "главбух")], level=3)   # пониженный уровень
        self.assertEqual(calls, [])
        _, _, calls = self.call("Какой гарантийный срок по договору поставки № П-17/2024 с ООО «Альфа-Трейд»?", [])
        self.assertEqual(calls, [])                                                   # поиск сильный, истории нет
        prep, _, calls = self.call("Испыталку скок?", [("user", "главбух")], reply="не JSON")
        self.assertEqual(len(calls), 1)
        self.assertIsNone(prep.standalone)                                            # ответ не разобран — как без R7
        self.assertIsNone(prep.extra)

    def test_run_with_flag(self) -> None:
        res = golden_runner.run_subprocess("root", "normal", {"SOLUTION_REWRITE": "1"})
        self.assertEqual(len(res["answers"]), 18)
        self.assertEqual(res["stats"]["level4"], 18)


@unittest.skipIf(harness.target() == "baseline", "нет в v1.3")
class GeneralPointsTest(unittest.TestCase):
    """B12: общие пункты больших статей вне урезанного бюджета (SOLUTION_GENERAL_POINTS)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.core = harness.load_core()
        with contextlib.redirect_stderr(io.StringIO()):
            cls.kb = cls.core.build_kb()

    def doc(self, label: str):
        return next(d for d in self.kb.docs if d.label == label)

    def test_topic(self) -> None:
        hit = self.core._topic_hit
        self.assertTrue(hit(self.doc("НК РФ, ст. 164"), "Какую ставку НДС применять к поставкам по договору с «Альфа-Трейд»?"))
        self.assertFalse(hit(self.doc("НК РФ, ст. 284"), "Какую ставку НДС применять к поставкам по договору с «Альфа-Трейд»?"))
        self.assertFalse(hit(self.doc("КоАП РФ, ст. 13.11"), "Куда должна быть доставлена партия мебели по спецификации?"))
        self.assertTrue(hit(self.doc("КоАП РФ, ст. 13.11"), "Какой штраф за обработку персональных данных без согласия?"))

    def test_selection(self) -> None:
        code = ("import contextlib, io, kb, index, dialog, retrieval\n"
                "with contextlib.redirect_stderr(io.StringIO()):\n"
                "    b = kb.build_kb(); i = index.Index(b)\n"
                "p = dialog.Prepared('Какая ставка НДС применялась к поставкам мебели по договору с «Альфа-Трейд» в 2024 году?', [])\n"
                "items, _ = retrieval.select_law(b, i, p, use_rerank=False)\n"
                "print(any(c.label == 'НК РФ, ст. 164, п. 3' for _, cs in items if cs for c in cs))")
        base = {k: v for k, v in harness.TEST_ENV.items()}
        self.assertEqual(run_py(code, {**base, "SOLUTION_GENERAL_POINTS": "1"}), "True")

    def test_run_with_flag(self) -> None:
        res = golden_runner.run_subprocess("root", "normal", {"SOLUTION_GENERAL_POINTS": "1"})
        self.assertEqual(len(res["answers"]), 18)
        self.assertEqual(res["stats"]["level4"], 18)


@unittest.skipIf(harness.target() == "baseline", "нет в v1.3")
class AnswerTemplateTest(unittest.TestCase):
    """B14: формат «… Источник: …», после источника ничего (SOLUTION_ANSWER_TEMPLATE)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.core = harness.load_core()

    def test_trim(self) -> None:
        t = self.core.trim_after_source
        a = "Арендная плата — 450 000 рублей. Источник: договор аренды № А-08/2024, п. 3.1."
        self.assertEqual(t(a + " Коммунальные платежи оплачиваются отдельно (п. 3.3)."), a)
        self.assertEqual(t(a), a)                                     # хвоста нет — без изменений
        self.assertEqual(t("Да, нужно (п. 2.4)."), "Да, нужно (п. 2.4).")  # нет «Источник:» — без изменений
        b = "Срок — 3 рабочих дня. Источник: Регламент в редакции от 01.03.2025, п. 3.2."
        self.assertEqual(t(b + " Оригинал хранится 5 лет."), b)       # даты и «п. 3.2.» внутри источника не режут его
        c = "Ставка — 22%. Источник: НК РФ, ст. 164, п. 3; договор № П-17/2024, п. 2.1."
        self.assertEqual(t(c), c)

    def test_prompt(self) -> None:
        code = ("import prompts\n"
                "print('После предложения «Источник» ничего не пиши' in prompts.SYSTEM_TEXT, "
                "all(a.rstrip().endswith('.') and ('Источник:' in a or 'нет документов' in a) for _, a in prompts.FEW_SHOT), "
                "'Коммунальные' in prompts.SYSTEM_TEXT)")
        on = run_py(code, {**harness.TEST_ENV, "SOLUTION_CONCISE": "1", "SOLUTION_ANSWER_TEMPLATE": "1"})
        self.assertEqual(on, "True True False")
        off = run_py(code, {**harness.TEST_ENV, "SOLUTION_CONCISE": "1", "SOLUTION_ANSWER_TEMPLATE": "0"})
        self.assertTrue(off.startswith("False"))
        fic = run_py(code, {**harness.TEST_ENV, "SOLUTION_CONCISE": "1", "SOLUTION_ANSWER_TEMPLATE": "1",
                            "SOLUTION_FEWSHOT": "fictional"})
        self.assertEqual(fic, "True True False")

    def test_run_with_flag(self) -> None:
        res = golden_runner.run_subprocess("root", "normal", {"SOLUTION_CONCISE": "1", "SOLUTION_ANSWER_TEMPLATE": "1"})
        self.assertEqual(len(res["answers"]), 18)
        self.assertEqual(res["stats"]["level4"], 18)


@unittest.skipIf(harness.target() == "baseline", "нет в v1.3")
class RewritePromptTest(unittest.TestCase):
    """R7b: SOLUTION_REWRITE_PROMPT=0 — переформулировка только для поиска, в сообщение модели не попадает."""

    def run_case(self, show: str) -> str:
        code = ("import json, contextlib, io, runtime, kb, index, dialog, retrieval, answering\n"
                "with contextlib.redirect_stderr(io.StringIO()):\n"
                "    b = kb.build_kb(); i = index.Index(b)\n"
                "sent = []\n"
                "def fake(messages, max_tokens=0, timeout=0.0):\n"
                "    sent.append(messages)\n"
                "    if messages[0]['content'].startswith('Ты помогаешь поиску'):\n"
                "        return json.dumps({'question': 'Какой максимальный срок испытания главного бухгалтера?', "
                "'queries': ['срок испытания главного бухгалтера']}, ensure_ascii=False)\n"
                "    return 'Не более шести месяцев. Источник: ТК РФ, ст. 70.'\n"
                "retrieval.chat = answering.chat = fake\n"
                "runtime.TIMING.add('chat_full', 5.0)\n"
                "p = dialog.Prepared('Испыталку ему скок?', [('user', 'У нас новый главбух.'), ('assistant', 'Чем помочь?')])\n"
                "answering.answer_one(b, i, 'вопрос', p, 4)\n"
                "user = sent[-1][1]['content']\n"
                "print(len(sent), p.standalone is not None, 'ВОПРОС СВОИМИ СЛОВАМИ' in user)")
        return run_py(code, {**harness.TEST_ENV, "SOLUTION_REWRITE": "1", "SOLUTION_REWRITE_PROMPT": show})

    def test_shown_by_default(self) -> None:
        self.assertEqual(self.run_case("1"), "2 True True")

    def test_search_only(self) -> None:
        self.assertEqual(self.run_case("0"), "2 True False")


@unittest.skipIf(harness.target() == "baseline", "нет в v1.3")
class B14FixesTest(unittest.TestCase):
    """Пакет 1: обрезка источника, защита удаления предложений, связка B9/B14, B14b, счётчик B5."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.core = harness.load_core()

    def test_trim_keeps_source(self) -> None:
        t = self.core.trim_after_source
        for a in ("Плата — 450 000 рублей. Источник: договор аренды № А-08/2024 с ООО «Северный Склад» (п. 3.1).",
                  "Срок — 3 дня. Источник: письмо № К-12 от 15.04.2025 г. Москва, п. 2.",
                  "Срок — 3 рабочих дня. Источник: Регламент в редакции от 01.03.2025, п. 3.2."):
            self.assertEqual(t(a), a)
        b = "Плата — 450 000 рублей. Источник: договор № А-08/2024, п. 3.1."
        self.assertEqual(t(b + " Коммунальные платежи отдельно."), b)

    def test_trim_leaves_multi_and_leading_source(self) -> None:
        t = self.core.trim_after_source
        multi = ("По П-17 — 30 дней. Источник: договор № П-17/2024, п. 4.2. По ПТ-05 — 45 дней. "
                 "Источник: договор № ПТ-05/2023, п. 4.1. Уточните контрагента.")
        self.assertEqual(t(multi), multi)
        lead = "Источник: договор № А-08/2024, п. 3.1. Плата — 450 000 рублей."
        self.assertEqual(t(lead), lead)

    def test_drop_never_leaves_only_source(self) -> None:
        a = ("Неустойка — 0,1% за каждый день, срок претензии 15 дней. "
             "Источник: договор поставки № П-17/2024 с ООО «Альфа-Трейд», п. 7.1.")
        self.assertEqual(self.core.drop_sentences(a, ["15 дней"]), a)
        self.assertEqual(self.core.drop_ref_sentences(a, ["15 дней"]), a)

    def test_template_trim_needs_concise(self) -> None:
        src = (harness.ROOT / "answering.py").read_text(encoding="utf-8")
        self.assertIn("if ANSWER_TEMPLATE and CONCISE:", src)

    def test_source_rule_prompt(self) -> None:
        code = ("import prompts\n"
                "print('Не бери источник из примеров' in prompts.SYSTEM_TEXT, "
                "any('ЮО-7 от' in a for _, a in prompts.FEW_SHOT))")
        env = {**harness.TEST_ENV, "SOLUTION_CONCISE": "1", "SOLUTION_ANSWER_TEMPLATE": "1"}
        self.assertEqual(run_py(code, {**env, "SOLUTION_SOURCE_RULE": "1"}), "True False")
        self.assertEqual(run_py(code, {**env, "SOLUTION_SOURCE_RULE": "0"}), "False True")

    def test_soft4xx_counts_question_once(self) -> None:
        code = ("import contextlib, io, runtime, kb, index, dialog, answering, llm\n"
                "with contextlib.redirect_stderr(io.StringIO()):\n"
                "    b = kb.build_kb(); i = index.Index(b)\n"
                "def bad(messages, max_tokens=0, timeout=0.0):\n"
                "    raise llm.APIError('400', status=400, detail='bad request')\n"
                "answering.chat = bad\n"
                "answering.time.sleep = lambda s: None\n"
                "p = dialog.Prepared('Какая арендная плата по договору с «Северным Складом»?', [])\n"
                "with contextlib.redirect_stderr(io.StringIO()):\n"
                "    try:\n"
                "        answering.answer_one(b, i, p.current, p, 4)\n"
                "    except Exception:\n"
                "        pass\n"
                "f = runtime.FLAGS\n"
                "print(f.soft4xx_streak, f.compact, f.minimal, f.tiny)")
        out = run_py(code, {**harness.TEST_ENV, "SOLUTION_SOFT_DEGRADE": "1"})
        self.assertEqual(out, "1 False False False")   # один вопрос — один шаг серии, глобальный режим не понижен


@unittest.skipIf(harness.target() == "baseline", "нет в v1.3")
class CheckFixesTest(unittest.TestCase):
    """Пакет 2 (SOLUTION_CHECK_FIXES): ложные срабатывания проверки чисел, B1, синонимы, даты только пользователя."""

    CODE = ("import json, checks, textproc, dialog\n"
            "ctx = 'Цена 2 000 000 рублей. Неустойка 0,1% за каждый день просрочки.'\n"
            "r = [checks.unverified(a, ctx, 'вопрос') for a in ('Стоимость — 2 050 рублей.', "
            "'Неустойка: 2 000 000 × 0,1% = 2 000 рублей за день.', 'Отвечать нужно за один день.')]\n"
            "a = 'Срок поставки — 10 дней. Договор действует, он подписан 31.12.2020. Источник: договор № П-17/2024, п. 3.1.'\n"
            "p = dialog.Prepared('А по договору 2024 года?', [('user', 'Вопрос'), ('assistant', 'Редакция от 01.03.2025.')])\n"
            "print(json.dumps([[k for _, k in x] for x in r] + [checks.drop_sentences(a, ['31.12.2020']) != a, "
            "'срочный' in textproc.expand_query('Нужно своевременно оплатить счёт?'), "
            "'01.03.2025' in dialog.event_text('Пользователь: Вопрос\\nАссистент: Редакция от 01.03.2025.\\nПользователь: А по договору 2024 года?')], "
            "ensure_ascii=False))")

    def test_on(self) -> None:
        out = run_py(self.CODE, {**harness.TEST_ENV, "SOLUTION_CHECK_FIXES": "1"})
        self.assertEqual(json.loads(out), [["amount"], [], [], True, False, False])

    def test_off_is_v13(self) -> None:
        out = run_py(self.CODE, {**harness.TEST_ENV, "SOLUTION_CHECK_FIXES": "0"})
        self.assertEqual(json.loads(out), [["date"], ["amount"], ["term"], False, True, True])

    def test_embed_bad_payload_is_api_error(self) -> None:
        code = ("import llm\n"
                "llm.post_json = lambda *a, **k: {'data': [{'index': 0}]}\n"
                "try:\n    llm.embed(['x'])\nexcept llm.APIError as e:\n    print('APIError')\n")
        self.assertEqual(run_py(code, dict(harness.TEST_ENV)), "APIError")


@unittest.skipIf(harness.target() == "baseline", "нет в v1.3")
class ChunkFixesTest(unittest.TestCase):
    """B19–B20 (SOLUTION_CHUNK_FIXES): нет фрагментов из одного заголовка; продолжения — с подпунктом и вводной фразой."""

    CODE = ("import contextlib, io, json, kb\n"
            "with contextlib.redirect_stderr(io.StringIO()):\n"
            "    b = kb.build_kb()\n"
            "law = [c for c in b.chunks if b.docs[c.doc].kind == 'law']\n"
            "heads = sum(1 for c in law if kb.HEADING_RE.match(c.text.strip()) and '\\n' not in c.text.strip())\n"
            "bare = sum(1 for c in law if c.text.startswith('(продолжение') and ':' not in c.text.split('\\n', 1)[0])\n"
            "head_intro = sum(1 for c in law if c.text.startswith('(продолжение') and '«Статья' in c.text.split('\\n', 1)[0])\n"
            "src = [d.text for d in b.docs]\n"
            "offs = all(b.docs[c.doc].text[c.start:c.end].strip() for c in b.chunks if c.start >= 0)\n"
            "print(json.dumps([len(b.chunks), heads, bare, head_intro, offs]))")

    def test_on(self) -> None:
        n, heads, bare, head_intro, offs = json.loads(run_py(self.CODE, {**harness.TEST_ENV, "SOLUTION_CHUNK_FIXES": "1"}))
        self.assertEqual((heads, bare, head_intro, offs), (0, 0, 0, True))
        self.assertEqual(n, 328)

    def test_off_is_v13(self) -> None:
        n, heads, bare, _, _ = json.loads(run_py(self.CODE, {**harness.TEST_ENV, "SOLUTION_CHUNK_FIXES": "0"}))
        self.assertEqual((n, heads), (336, 8))
        self.assertGreater(bare, 0)
