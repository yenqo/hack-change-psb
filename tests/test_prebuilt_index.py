"""7.2: готовый индекс index_vectors.json. Ответы не меняются (golden master), эмбеддинги базы при старте не считаются."""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest

import golden_runner
import harness

BUILD_STUB_INDEX = r"""
import sys, contextlib, io
from pathlib import Path
sys.path.insert(0, "tests"); sys.path.insert(0, ".")
import harness
core = harness.load_core("root")
with contextlib.redirect_stderr(io.StringIO()):
    kb = core.build_kb()
idx = core.Index(kb)
idx.vecs = [harness.fake_vector(c.search[:1200]) for c in kb.chunks]   # те же векторы, что даёт заглушка
idx.save_vectors(Path(sys.argv[1]), core.EMB_MODEL)
"""


@unittest.skipIf(harness.target() == "baseline", "нет в v1.3")
class PrebuiltIndexTest(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        with gzip.open(harness.FIXTURES / "golden_normal.json.gz", "rt", encoding="utf-8") as f:
            cls.fixture = json.load(f)
        cls.tmp = tempfile.mkdtemp()
        cls.stub_index = os.path.join(cls.tmp, "index_vectors.json")
        subprocess.run([sys.executable, "-c", BUILD_STUB_INDEX, cls.stub_index], cwd=harness.ROOT, check=True,
                       env=dict(os.environ, **harness.TEST_ENV), capture_output=True)

    def probe_block(self, err: str) -> dict:
        import re
        return json.loads(re.search(r"=== PLATFORM PROBE ===\n(.*?)\n=== /PLATFORM PROBE ===", err, re.S).group(1))

    def test_used_and_same_answers(self) -> None:
        res = golden_runner.run_subprocess("root", "normal", {"SOLUTION_PREBUILT_INDEX": "1",
                                                              "SOLUTION_INDEX_FILE": self.stub_index})
        self.assertEqual(res["answers"], self.fixture["answers"])
        self.assertEqual(res["chat"], self.fixture["chat"])
        self.assertEqual(res["rerank"], self.fixture["rerank"])
        # эмбеддинги базы (336 текстов батчами по 16) не запрашивались: только контроль + вопросы
        self.assertLess(len(res["embed"]), len(self.fixture["embed"]))
        info = self.probe_block(res["_stderr"])["kb"]["prebuilt_index"]
        self.assertTrue(info["used"], info)
        self.assertGreaterEqual(info["min_cos"], 0.99)

    def test_mismatch_falls_back(self) -> None:
        """Векторы из другой модели (настоящая bge-m3 против заглушки) — контроль не проходит, расчёт как в v1.3."""
        real = harness.ROOT / "index_vectors.json"
        if not real.is_file():
            self.skipTest("нет index_vectors.json")
        res = golden_runner.run_subprocess("root", "normal", {"SOLUTION_PREBUILT_INDEX": "1",
                                                              "SOLUTION_INDEX_FILE": str(real)})
        self.assertEqual(res["answers"], self.fixture["answers"])
        self.assertEqual(res["chat"], self.fixture["chat"])
        info = self.probe_block(res["_stderr"])["kb"]["prebuilt_index"]
        self.assertFalse(info["used"])
        self.assertRegex(info["reason"], "отличаются|размерность")

    def test_wrong_kb_ignored(self) -> None:
        bad = os.path.join(self.tmp, "bad.json")
        with open(self.stub_index, encoding="utf-8") as f:
            data = json.load(f)
        data["kb_sha256"] = "0" * 64
        with open(bad, "w", encoding="utf-8") as f:
            json.dump(data, f)
        res = golden_runner.run_subprocess("root", "normal", {"SOLUTION_PREBUILT_INDEX": "1",
                                                              "SOLUTION_INDEX_FILE": bad})
        self.assertEqual(res["answers"], self.fixture["answers"])
        self.assertEqual(res["embed"], self.fixture["embed"])   # контроль не вызывался — сразу v1.3
        self.assertEqual(self.probe_block(res["_stderr"])["kb"]["prebuilt_index"]["reason"], "не совпал kb_sha256")


@unittest.skipIf(harness.target() == "baseline", "нет в v1.3")
class PrebuiltIndexRobustTest(unittest.TestCase):
    def test_unexpected_error_falls_back(self) -> None:
        """Любая неожиданная ошибка при загрузке (например, странный ответ /embeddings) — не исключение, а откат."""
        import contextlib
        import io
        core = harness.load_core()
        with contextlib.redirect_stderr(io.StringIO()):
            kb = core.build_kb()
        idx = core.Index(kb)
        mod = next(m for m in core._modules if m.__name__ == "index")
        old_base, old_embed = mod.API_BASE, mod.embed

        def boom(texts):
            raise ZeroDivisionError("неожиданно")
        mod.API_BASE, mod.embed = "http://x", boom
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                info = idx.load_prebuilt(path=harness.ROOT / "index_vectors.json")
        finally:
            mod.API_BASE, mod.embed = old_base, old_embed
        self.assertFalse(info["used"])
        self.assertIn("ZeroDivisionError", info["reason"])
        self.assertIsNone(idx.vecs)


if __name__ == "__main__":
    unittest.main()
