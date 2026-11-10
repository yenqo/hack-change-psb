"""Сборка и проверка ZIP-сабмита для платформы.

    python tools/make_submission.py                 # собрать dist/psb_submission_<commit>.zip и проверить
    python tools/make_submission.py --no-check      # только собрать
    python tools/make_submission.py --python python3.11   # интерпретатор для чистого venv
    python tools/make_submission.py --stub-proxy    # прогон через локальную заглушку прокси (tools/stub_proxy.py)

В архив попадают только: модули ядра, от которых зависит solution.py (плоско в корне), requirements.txt
и готовый индекс index_*.json, если он есть. Ни одной вложенной папки.

Проверка: распаковка во временную папку → venv только с зависимостями из requirements.txt → запуск
solution.py на data/ (demo_questions.csv как questions.csv) с ограничением времени → проверка формата
submission.csv. Переменные LLM_* из текущего окружения передаются в прогон, если заданы (иначе — запасные ответы).
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
CORE = ["config", "textproc", "dates", "kb", "runtime", "llm", "index", "prompts", "checks", "pipeline", "probe",
        "solution"]
PASS_ENV = ["LLM_BASE_URL", "OPENAI_BASE_URL", "BASE_URL", "LLM_API_KEY", "OPENAI_API_KEY", "EMBEDDER_API_KEY",
            "RERANKER_API_KEY", "LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL", "EMB_MODEL", "RERANKER_MODEL", "TEMPERATURE",
            "MAX_TOKENS", "MAX_WORKERS", "TOP_K", "RERANKER_TOP_N", "CHUNKER", "DEV_EXTRA_BODY"]


def core_closure() -> list[str]:
    """Модули ядра, которые импортирует solution.py (транзитивно)."""
    need, todo = set(), ["solution"]
    while todo:
        m = todo.pop()
        if m in need:
            continue
        need.add(m)
        tree = ast.parse((ROOT / f"{m}.py").read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            names = []
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                names = [n.module]
            todo += [x for x in names if x in CORE and (ROOT / f"{x}.py").is_file()]
    return [m for m in CORE if m in need]


def git_rev() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "*.py", "requirements.txt", "index_*.json"],
                               cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return time.strftime("%Y%m%d%H%M")


def build(out: Path | None = None) -> Path:
    files = [f"{m}.py" for m in core_closure()] + ["requirements.txt"] + sorted(p.name for p in ROOT.glob("index_*.json"))
    DIST.mkdir(exist_ok=True)
    out = out or DIST / f"psb_submission_{git_rev()}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in files:
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, (ROOT / name).read_bytes())
    size = out.stat().st_size
    print(f"Собран {out.relative_to(ROOT)}: {len(files)} файлов, {size / 1024:.0f} КБ")
    for name in files:
        print(f"  {name}")
    if size > 128 * 1024 * 1024:
        raise SystemExit("ZIP больше 128 МБ")
    return out


def check_archive(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    allowed_extra = {"requirements.txt"}
    bad = [n for n in names if "/" in n or "\\" in n
           or not (n.endswith(".py") or n in allowed_extra or (n.startswith("index_") and n.endswith(".json")))]
    if "solution.py" not in names:
        bad.append("нет solution.py")
    if "requirements.txt" not in names:
        bad.append("нет requirements.txt")
    return bad


def check_submission(path: Path, ids: list[str]) -> list[str]:
    problems = []
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] != ["id", "answer"]:
        problems.append(f"заголовок {rows[0] if rows else None}, ожидается id,answer")
        return problems
    got = [r[0] for r in rows[1:]]
    if any(len(r) != 2 for r in rows):
        problems.append("строки не из двух колонок")
    if len(set(got)) != len(got):
        problems.append("повторяющиеся id")
    if sorted(got) != sorted(ids):
        problems.append(f"id не совпадают с вопросами: нет {sorted(set(ids) - set(got))[:5]}, лишние {sorted(set(got) - set(ids))[:5]}")
    for r in rows[1:]:
        if len(r) == 2 and not r[1].strip():
            problems.append(f"пустой ответ id={r[0]}")
        if len(r) == 2 and len(r[1]) > 10000:
            problems.append(f"ответ длиннее 10 000 символов id={r[0]}")
    return problems


def run_clean(zip_path: Path, python: str, questions: Path, time_limit: int, stub: bool = False) -> int:
    work = Path(tempfile.mkdtemp(prefix="psb_check_"))
    try:
        app = work / "app"
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(app)
        print(f"Чистый venv ({python}) только с зависимостями из requirements.txt…")
        subprocess.run([python, "-m", "venv", str(work / "venv")], check=True)
        vpy = work / "venv" / "bin" / "python"
        subprocess.run([str(vpy), "-m", "pip", "install", "-q", "--disable-pip-version-check", "-r",
                        str(app / "requirements.txt")], check=True)
        frozen = subprocess.run([str(vpy), "-m", "pip", "freeze"], capture_output=True, text=True).stdout.split()
        print(f"  установлено: {', '.join(frozen)}")
        data = work / "data"
        data.mkdir()
        shutil.copy(questions, data / "questions.csv")
        shutil.copy(ROOT / "data" / "knowledge-base.zip", data / "knowledge-base.zip")
        run_dir = work / "run"
        run_dir.mkdir()
        env = {k: v for k, v in os.environ.items() if k not in PASS_ENV and not k.startswith("SOLUTION_")}
        env.update({k: os.environ[k] for k in PASS_ENV if os.environ.get(k)})
        env["DATA_DIR"] = str(data)
        srv = None
        if stub:
            sys.path.insert(0, str(ROOT / "tools"))
            import stub_proxy
            srv, url = stub_proxy.start()
            env.update({"LLM_BASE_URL": url, "LLM_API_KEY": "stub-key", "LLM_MODEL": "stub-model"})
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        print(f"Запуск solution.py, лимит {time_limit} с, модель: {'есть' if env.get('LLM_BASE_URL') else 'нет (запасные ответы)'}")
        t0 = time.monotonic()
        try:
            proc = subprocess.run([str(vpy), str(app / "solution.py")], cwd=run_dir, env=env, capture_output=True,
                                  text=True, timeout=time_limit)
        except subprocess.TimeoutExpired:
            print(f"ОШИБКА: превышен лимит {time_limit} с")
            return 1
        finally:
            if srv is not None:
                srv.shutdown()
                print(f"  вызовы заглушки: {stub_proxy.Handler.calls}")
        elapsed = time.monotonic() - t0
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        print(f"  код возврата {proc.returncode}, {elapsed:.1f} с; конец лога:\n{tail}")
        if proc.returncode != 0:
            return 1
        with open(questions, encoding="utf-8-sig", newline="") as f:
            ids = [r["id"] for r in csv.DictReader(f)]
        sub = run_dir / "submission.csv"
        if not sub.is_file():
            print("ОШИБКА: submission.csv не создан")
            return 1
        problems = check_submission(sub, ids)
        stray = [p.name for p in run_dir.iterdir() if p.name != "submission.csv"]
        if stray:
            print(f"  в рабочем каталоге также: {stray}")
        for p in problems:
            print(f"ОШИБКА: {p}")
        if not problems:
            print(f"submission.csv в порядке: {len(ids)} ответов")
        return 1 if problems else 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--python", default=sys.executable, help="интерпретатор, от которого создаётся чистый venv")
    ap.add_argument("--questions", default=str(ROOT / "data" / "demo_questions.csv"))
    ap.add_argument("--time-limit", type=int, default=1200)
    ap.add_argument("--out")
    ap.add_argument("--stub-proxy", action="store_true", help="прогон через локальную заглушку прокси")
    args = ap.parse_args()
    z = build(Path(args.out) if args.out else None)
    bad = check_archive(z)
    for b in bad:
        print(f"ОШИБКА в архиве: {b}")
    if bad:
        return 1
    if args.no_check:
        return 0
    return run_clean(z, args.python, Path(args.questions), args.time_limit, args.stub_proxy)


if __name__ == "__main__":
    sys.exit(main())
