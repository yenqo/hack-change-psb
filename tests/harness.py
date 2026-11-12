"""Общий каркас тестов: загрузка ядра и детерминированные заглушки HTTP-слоя.

Ядро загружается в одном из двух видов (переменная CORE_TARGET):
  baseline — tools/baseline/solution_v1_3.py, один файл v1.3;
  root     — модули ядра в корне репозитория (по умолчанию).
Тесты обращаются к ядру через `load_core()`: объект с атрибутами всех функций и классов ядра,
независимо от того, в каком модуле они лежат.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import math
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
BASELINE_FILE = ROOT / "tools" / "baseline" / "solution_v1_3.py"
TODAY = "2026-11-10"

# Порядок зависимостей модулей ядра (сверху вниз); отсутствующие пропускаются.
CORE_MODULES = ["config", "textproc", "dates", "kb", "runtime", "llm", "index", "prompts", "checks", "pipeline",
                "probe", "solution"]

# Окружение ядра в тестах: без сети, дата и бюджет времени зафиксированы.
TEST_ENV = {
    "DATA_DIR": str(DATA),
    "SOLUTION_TIME_BUDGET": "3600",
    "SOLUTION_TODAY": TODAY,
    "MAX_WORKERS": "1",
    # golden master сравнивает с v1.3 — без готового индекса (его путь проверяет tests/test_prebuilt_index.py)
    "SOLUTION_PREBUILT_INDEX": "0",
    # и без исправлений устойчивости (B5 и далее): их поведение проверяют tests/test_pending.py и tools/fault_eval.py
    "SOLUTION_SOFT_DEGRADE": "0",
    "SOLUTION_RERANK_RECHECK": "0",
    "SOLUTION_LEVEL_FORECAST": "0",
    "SOLUTION_FINAL_PASS": "0",
    "SOLUTION_CONCISE": "0",      # golden master — промпт v1.3; B9 проверяет tests/test_pending.py
    "SOLUTION_DATE_CONDITIONS": "0",
    "SOLUTION_CROSS_REFS": "0",
    "SOLUTION_REWRITE": "0",
    "SOLUTION_GENERAL_POINTS": "0",
    "SOLUTION_ANSWER_TEMPLATE": "0",
    "SOLUTION_SOURCE_RULE": "0",
    "SOLUTION_CHECK_FIXES": "0",
    "SOLUTION_CHUNK_FIXES": "0",
}
PLATFORM_VARS = ["LLM_BASE_URL", "BASE_URL", "OPENAI_BASE_URL", "LLM_API_KEY", "OPENAI_API_KEY", "EMBEDDER_API_KEY",
                 "RERANKER_API_KEY", "LLM_MODEL", "GEN_MODEL", "OPENAI_MODEL", "EMB_MODEL", "RERANKER_MODEL",
                 "TEMPERATURE", "MAX_TOKENS", "TOP_K", "RERANKER_TOP_N", "CHUNKER", "SOLUTION_DEBUG"]


def target() -> str:
    t = os.environ.get("CORE_TARGET", "root")
    if t not in ("root", "baseline"):
        raise ValueError(f"CORE_TARGET={t}: ожидается root или baseline")
    return t


# Значения TEST_ENV, которые тест может переопределить через окружение подпроцесса
OVERRIDABLE = {"SOLUTION_PREBUILT_INDEX", "SOLUTION_SOFT_DEGRADE", "SOLUTION_RERANK_RECHECK",
               "SOLUTION_LEVEL_FORECAST", "SOLUTION_FINAL_PASS", "SOLUTION_CONCISE",
               "SOLUTION_DATE_CONDITIONS", "SOLUTION_CROSS_REFS",
               "SOLUTION_REWRITE", "SOLUTION_GENERAL_POINTS",
               "SOLUTION_ANSWER_TEMPLATE", "SOLUTION_SOURCE_RULE",
               "SOLUTION_CHECK_FIXES", "SOLUTION_CHUNK_FIXES"}


def apply_env(extra: dict | None = None) -> None:
    for k in PLATFORM_VARS:
        os.environ.pop(k, None)
    for k, v in TEST_ENV.items():
        if k in OVERRIDABLE:
            os.environ.setdefault(k, v)
        else:
            os.environ[k] = v
    if extra:
        os.environ.update(extra)


_CACHE: dict[str, SimpleNamespace] = {}


def load_core(which: str | None = None, extra_env: dict | None = None) -> SimpleNamespace:
    """Ядро как пространство имён. Загружается один раз на процесс и цель; окружение читается при импорте."""
    which = which or target()
    if which in _CACHE:
        return _CACHE[which]
    apply_env(extra_env)
    ns = SimpleNamespace()
    if which == "baseline":
        spec = importlib.util.spec_from_file_location("solution_v1_3", BASELINE_FILE)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["solution_v1_3"] = mod
        spec.loader.exec_module(mod)
        ns.__dict__.update({k: v for k, v in vars(mod).items() if not k.startswith("__")})
        ns._modules = [mod]
    else:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        mods = []
        for name in CORE_MODULES:
            if (ROOT / f"{name}.py").is_file():
                mods.append(importlib.import_module(name))
        for mod in mods:
            ns.__dict__.update({k: v for k, v in vars(mod).items() if not k.startswith("__")})
        ns._modules = mods
    _CACHE[which] = ns
    return ns


def patch_everywhere(core: SimpleNamespace, name: str, value) -> None:
    """Подменяет атрибут во всех модулях ядра, где он есть (в т. ч. импортированные копии)."""
    for mod in core._modules:
        if hasattr(mod, name):
            setattr(mod, name, value)
    setattr(core, name, value)


# ---------------------------------------------------------------------------
# Детерминированные заглушки прокси
# ---------------------------------------------------------------------------

def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fake_vector(text: str, dim: int = 64) -> list[float]:
    raw = b""
    i = 0
    while len(raw) < dim * 4:
        raw += hashlib.sha256(f"{i}:{text}".encode("utf-8")).digest()
        i += 1
    ints = struct.unpack(f"<{dim}I", raw[: dim * 4])
    v = [x / 2 ** 32 - 0.5 for x in ints]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def fake_rerank_score(query: str, doc: str) -> float:
    return int(sha(query + "\x00" + doc)[:8], 16) / 2 ** 32
