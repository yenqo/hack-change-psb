"""Контракт фасада assistant.py для веб-части (docs/integration.md).

Проверяет: JSON-структуру по контракту, ответы фасада = пакетный режим (golden master v1.3), стабильность doc_id,
согласованность смещений (text[start:end] == quote), отсутствие исключений наружу.
Только для CORE_TARGET=root: в v1.3 фасада нет.
"""

from __future__ import annotations

import gzip
import json
import unittest

import contract_runner
import harness

STATUSES = {"answer", "partial", "no_info", "premise_error", "clarify", "out_of_scope", "error"}
RESULT_KEYS = {"contract_version", "status", "text", "citations", "warnings", "timing_ms", "debug"}
CITATION_KEYS = {"doc_id", "doc_title", "doc_kind", "clause", "quote", "start", "end", "found",
                 "applies_to_event_date"}
DOC_KEYS = {"doc_id", "title", "label", "kind", "type", "status", "valid_from", "valid_to", "export_date", "related",
            "has_removed_instructions"}


def skip_baseline(cls):
    if harness.target() == "baseline":
        return unittest.skip("фасада нет в v1.3")(cls)
    return cls


@skip_baseline
class ContractTest(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.res = contract_runner.run_subprocess("normal", "dialogs")
        with gzip.open(harness.FIXTURES / "golden_normal.json.gz", "rt", encoding="utf-8") as f:
            cls.batch = dict(map(tuple, json.load(f)["answers"]))

    def test_same_as_batch(self) -> None:
        """Тот же диалог → тот же текст, что в submission.csv пакетного режима (и v1.3)."""
        for qid, r in self.res["answers"]:
            self.assertEqual(r["text"], self.batch[qid], f"вопрос {qid}")

    def test_same_as_batch_text_mode(self) -> None:
        res = contract_runner.run_subprocess("normal", "text")
        self.assertEqual([r["text"] for _, r in res["answers"]], [r["text"] for _, r in self.res["answers"]])

    def test_round_trip_platform_format(self) -> None:
        self.assertTrue(all(self.res["extra"]["to_platform"]))

    def test_result_schema(self) -> None:
        for qid, r in self.res["answers"]:
            self.assertEqual(set(r), RESULT_KEYS, qid)
            self.assertEqual(r["contract_version"], "1.0")
            self.assertIn(r["status"], STATUSES)
            self.assertTrue(r["text"].strip())
            self.assertIn("total", r["timing_ms"])
            self.assertEqual(r["debug"], {})  # без отладки — пусто
            for c in r["citations"]:
                self.assertEqual(set(c), CITATION_KEYS, qid)
                self.assertIn(c["doc_kind"], ("company", "law"))
            for w in r["warnings"]:
                self.assertIn("code", w)
                self.assertIn("message", w)

    def test_stages(self) -> None:
        for qid, seen in self.res["stages"].items():
            self.assertEqual(seen[:3], ["search", "generate", "verify"], qid)

    def test_citation_offsets(self) -> None:
        full = self.res["full"]
        n = 0
        for qid, r in self.res["answers"]:
            for c in r["citations"]:
                text = full[c["doc_id"]]["text"]
                self.assertEqual(text[c["start"]:c["end"]], c["quote"], f"{qid}: {c['doc_id']} {c['clause']}")
                n += 1
        self.assertGreater(n, 5)

    def test_documents(self) -> None:
        docs = self.res["documents"]
        self.assertEqual(len(docs), 21)
        ids = [d["doc_id"] for d in docs]
        self.assertEqual(len(set(ids)), len(ids))
        for d in docs:
            self.assertEqual(set(d), DOC_KEYS)
            for rel in d["related"]:
                self.assertIn(rel, ids)
        self.assertEqual(sum(d["has_removed_instructions"] for d in docs), 1)

    def test_doc_ids_stable(self) -> None:
        """doc_id не меняются между сборками: эталон — tests/fixtures/doc_ids.json."""
        path = harness.FIXTURES / "doc_ids.json"
        ids = sorted(d["doc_id"] for d in self.res["documents"])
        if not path.is_file():  # первый запуск фиксирует эталон
            path.write_text(json.dumps(ids, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        self.assertEqual(ids, json.loads(path.read_text(encoding="utf-8")))

    def test_get_document_markup(self) -> None:
        for doc_id, d in self.res["full"].items():
            text = d["text"]
            for c in d["clauses"]:
                self.assertTrue(0 <= c["start"] < c["end"] <= len(text), doc_id)
                self.assertTrue(text[c["start"]:c["end"]].startswith(c["title"][:20]), f"{doc_id} {c['clause']}")
            for n in d["edition_notes"]:
                self.assertEqual(text[n["start"]:n["end"]], n["text"])
            for r in d["removed_instructions"]:
                self.assertTrue(text[r["start"]:r["end"]].startswith("[Фрагмент удалён"))
                self.assertTrue(r["original"])
        self.assertIsNone(self.res["extra"]["unknown_doc"])

    def test_empty_question(self) -> None:
        e = self.res["extra"]["empty"]
        self.assertEqual(e["status"], "error")
        self.assertEqual(e["warnings"][0]["code"], "empty_question")

    def test_no_secrets(self) -> None:
        blob = json.dumps(self.res, ensure_ascii=False)
        self.assertNotIn("test-key", blob)
        self.assertNotIn("stub.invalid", blob)


@skip_baseline
class CitationsOnRealAnswersTest(unittest.TestCase):
    """Цитаты и статусы на реальных ответах модели (смоук-прогон, сохранён в eval/runs) — без сети."""

    def test_offsets_and_resolution(self) -> None:
        import contextlib
        import csv
        import io
        core = harness.load_core()
        run = json.loads((harness.ROOT / "eval" / "runs" / "smoke_dense_current.json").read_text(encoding="utf-8"))
        with open(harness.ROOT / "eval" / "smoke_demo.csv", encoding="utf-8", newline="") as f:
            qs = {r["id"]: r["question"] for r in csv.DictReader(f)}
        with contextlib.redirect_stderr(io.StringIO()):
            kb = core.build_kb()
        ids = core.doc_ids(kb)
        by_id = {v: kb.docs[k] for k, v in ids.items()}
        total = 0
        for qid, ans in run["answers"].items():
            cites = core.extract_citations(kb, ans, qs[qid], ids)
            for c in cites:
                self.assertEqual(by_id[c["doc_id"]].text[c["start"]:c["end"]], c["quote"])
                total += 1
            self.assertIn(core.classify_status(ans, cites), STATUSES)
        self.assertGreater(total, 25)
        # известные случаи: п. 2.4 в ответе про август 2024 — редакция 2023; «ст. 9, ч. 2 … № 152-ФЗ»
        c5 = core.extract_citations(kb, run["answers"]["5"], qs["5"], ids)
        self.assertIn(("reglament-soglasovaniya-dogovorov-red-2023-utratil-silu", "2.4"),
                      {(c["doc_id"], c["clause"]) for c in c5})
        c13 = core.extract_citations(kb, run["answers"]["13"], qs["13"], ids)
        self.assertIn("ч. 2", {c["clause"] for c in c13 if c["doc_id"].startswith("152fz")})


@skip_baseline
class ContractFailureTest(unittest.TestCase):
    def test_broken_transport(self) -> None:
        res = contract_runner.run_subprocess("broken")
        for qid, r in res["answers"]:
            self.assertEqual(r["status"], "error", qid)
            self.assertTrue(r["text"].strip())
            self.assertTrue(r["warnings"])

    def test_timeout(self) -> None:
        res = contract_runner.run_subprocess("slow")
        (qid, r), = res["answers"]
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["warnings"][0]["code"], "timeout")
        self.assertLess(r["timing_ms"]["total"], 3000)


if __name__ == "__main__":
    unittest.main()
