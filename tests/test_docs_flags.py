"""Каждая переменная SOLUTION_* ядра описана в README и есть в .env.example (иначе документация расходится с кодом)."""

from __future__ import annotations

import re
import unittest

import harness


@unittest.skipIf(harness.target() == "baseline", "для текущего ядра")
class DocsFlagsTest(unittest.TestCase):
    def test_flags_documented(self) -> None:
        names = set()
        for p in harness.ROOT.glob("*.py"):
            names |= set(re.findall(r'env(?:_int|_float)?\("(SOLUTION_[A-Z_]+)"', p.read_text(encoding="utf-8")))
        self.assertTrue(names)
        readme = (harness.ROOT / "README.md").read_text(encoding="utf-8")
        example = (harness.ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertEqual(sorted(n for n in names if f"`{n}`" not in readme), [], "нет в README")
        self.assertEqual(sorted(n for n in names if not re.search(rf"^{n}=", example, re.M)), [], "нет в .env.example")


if __name__ == "__main__":
    unittest.main()
