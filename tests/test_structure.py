"""Согласованность структуры ядра: списки модулей в инструментах и флаги в окружении тестов.

Ядро — плоские модули в корне, и их перечисляют четыре места: фасад (принудительный веб-режим), статическая
проверка, сборка архива и каркас тестов. Модуль, забытый в сборке архива, — ImportError на платформе; забытый в
фасаде — пакетный режим в веб-процессе.
"""

from __future__ import annotations

import ast
import re
import unittest

import harness

ROOT = harness.ROOT
# не меняют ответы: блоки диагностики, режим процесса, показ переформулировки (действует только вместе с R7)
NOT_PINNED = {"SOLUTION_PROBE", "SOLUTION_MODE", "SOLUTION_REWRITE_PROMPT"}


def list_literal(path: str, name: str) -> list[str]:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise AssertionError(f"{name} не найден в {path}")


class ModuleListsTest(unittest.TestCase):
    def test_lists_agree(self) -> None:
        tests_list = harness.CORE_MODULES
        self.assertEqual(tests_list[-2:], ["probe", "solution"])
        self.assertEqual(list_literal("tools/make_submission.py", "CORE"), tests_list)
        self.assertEqual(list_literal("tools/check_core.py", "CORE"), tests_list + ["assistant"])
        self.assertEqual(list_literal("assistant.py", "_CORE_MODULES"), tests_list[:-2])

    def test_every_root_module_listed(self) -> None:
        root = sorted(p.stem for p in ROOT.glob("*.py"))
        self.assertEqual(root, sorted(list_literal("tools/check_core.py", "CORE")))


class FlagsPinnedTest(unittest.TestCase):
    def test_default_on_flags_pinned(self) -> None:
        """Флаг доработки, включённый по умолчанию, в тестах зафиксирован на v1.3 (иначе golden master сравнивает не то)."""
        src = (ROOT / "config.py").read_text(encoding="utf-8")
        on = {m.group(1) for m in re.finditer(r'env\("(SOLUTION_\w+)", "([^"]+)"\)', src)}
        missing = sorted(on - set(harness.TEST_ENV) - NOT_PINNED)
        self.assertEqual(missing, [])


class FingerprintDefaultsTest(unittest.TestCase):
    def test_prod_defaults_match_config(self) -> None:
        """«prod» в tools/fingerprint.py — копия значений по умолчанию из config.py; при смене умолчания — обновить."""
        import json
        import os
        import subprocess
        import sys
        sys.path.insert(0, str(ROOT / "tools"))
        import fingerprint
        keys = sorted(fingerprint.PROD_DEFAULTS)
        code = ("import json, config\n"
                f"keys = {keys!r}\n"
                "out = {}\n"
                "for k in keys:\n"
                "    v = getattr(config, k[len('SOLUTION_'):])\n"
                "    out[k] = v if isinstance(v, str) else ('1' if v else '0')\n"
                "print(json.dumps(out))")
        env = {k: v for k, v in os.environ.items() if not k.startswith("SOLUTION_")}
        out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env, capture_output=True, text=True,
                             check=True).stdout
        self.assertEqual(json.loads(out), fingerprint.PROD_DEFAULTS)


if __name__ == "__main__":
    unittest.main()
