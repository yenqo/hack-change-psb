"""Служебная папка eval/chunks совпадает с текущей нарезкой kb.py (иначе — пересоздать: tools/show_chunks.py --dump)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import harness


@unittest.skipIf(harness.target() == "baseline", "служебная папка — для текущего ядра")
class ChunksDumpTest(unittest.TestCase):
    def test_up_to_date(self) -> None:
        saved = harness.ROOT / "eval" / "chunks"
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(harness.ROOT / "tools" / "show_chunks.py"), "--dump", tmp],
                           check=True, capture_output=True, cwd=harness.ROOT)
            fresh = {p.name: p.read_text(encoding="utf-8") for p in Path(tmp).glob("*.md")}
        old = {p.name: p.read_text(encoding="utf-8") for p in saved.glob("*.md")}
        self.assertEqual(sorted(old), sorted(fresh), "состав файлов eval/chunks устарел — tools/show_chunks.py --dump eval/chunks")
        stale = [n for n in fresh if fresh[n] != old.get(n)]
        self.assertEqual(stale, [], "eval/chunks устарела — tools/show_chunks.py --dump eval/chunks")


if __name__ == "__main__":
    unittest.main()
