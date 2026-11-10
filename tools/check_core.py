"""Статическая проверка модулей ядра перед коммитом.

    python tools/check_core.py

1. Импорты корневых модулей — только стандартная библиотека, httpx и сами модули ядра.
2. Нет неопределённых имён (включая выражения внутри f-строк).
3. Нет циклических импортов между модулями ядра (порядок config → … → solution).
"""

from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ["config", "textproc", "dates", "kb", "runtime", "llm", "index", "prompts", "checks", "pipeline", "probe",
        "solution", "assistant"]
ALLOWED_THIRD_PARTY = {"httpx"}
STDLIB = set(getattr(sys, "stdlib_module_names", ())) or {
    "__future__", "argparse", "array", "base64", "builtins", "collections", "concurrent", "contextlib", "copy", "csv",
    "dataclasses", "datetime", "enum", "functools", "gzip", "hashlib", "heapq", "io", "itertools", "json", "logging",
    "math", "operator", "os", "pathlib", "platform", "sqlite3", "http", "urllib", "socket", "ssl", "glob", "signal", "fnmatch", "bisect", "decimal", "fractions", "pickle", "shlex", "inspect", "random", "re", "shutil", "statistics", "string", "struct", "subprocess",
    "sys", "tempfile", "textwrap", "threading", "time", "traceback", "types", "typing", "unicodedata", "uuid",
    "warnings", "xml", "zipfile", "zlib",
}


def undefined_names(tree: ast.Module) -> list[tuple[int, str]]:
    defined = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            defined.add(n.id)
        elif isinstance(n, ast.arg):
            defined.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                defined.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            defined.add(n.name)
    in_annotation = set()
    for n in ast.walk(tree):
        for attr in ("annotation", "returns"):
            x = getattr(n, attr, None)
            if x is not None:
                in_annotation.update(id(y) for y in ast.walk(x))
    return [(n.lineno, n.id) for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in defined
            and id(n) not in in_annotation]


def main() -> int:
    problems = []
    graph: dict[str, set[str]] = {}
    for name in CORE:
        path = ROOT / f"{name}.py"
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        deps = set()
        type_only = {id(x) for n in ast.walk(tree) if isinstance(n, ast.If) and "TYPE_CHECKING" in ast.unparse(n.test)
                     for x in ast.walk(n)}
        for n in ast.walk(tree):
            if id(n) in type_only:  # импорты только для аннотаций не создают зависимости при запуске
                continue
            mods = []
            if isinstance(n, ast.Import):
                mods = [a.name.split(".")[0] for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                mods = [n.module.split(".")[0]]
            for m in mods:
                if m in CORE:
                    deps.add(m)
                elif m not in STDLIB and m not in ALLOWED_THIRD_PARTY:
                    problems.append(f"{name}.py: недопустимая зависимость {m}")
        graph[name] = deps
        for line, ident in undefined_names(tree):
            problems.append(f"{name}.py:{line}: неопределённое имя {ident}")
    order = {m: i for i, m in enumerate(CORE)}
    for m, deps in graph.items():
        for d in deps:
            if order[d] >= order[m]:
                problems.append(f"{m}.py импортирует {d}.py: нарушен порядок зависимостей (возможен цикл)")
    for p in problems:
        print(p)
    print(f"Модулей ядра: {len(graph)}; проблем: {len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
