"""Сборка готового индекса index_vectors.json (доработка 7.2): эмбеддинги всех фрагментов базы заранее.

    python tools/build_index.py              # через tools/dev_proxy.py: локальная bge-m3 (ollama)
    python tools/build_index.py --env        # эмбеддер из окружения (LLM_BASE_URL/EMBEDDER_API_KEY/EMB_MODEL)

Файл кладётся в корень репозитория и попадает в ZIP-сабмит. На старте ядро (SOLUTION_PREBUILT_INDEX=1) берёт его,
только если совпали отпечаток базы, нарезка, модель и размерность, а контрольные векторы API платформы совпадают
с сохранёнными (косинус ≥ 0,99); иначе считает эмбеддинги как раньше.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", action="store_true", help="эмбеддер из окружения, а не dev-прокси")
    ap.add_argument("--out", default=str(ROOT / "index_vectors.json"))
    args = ap.parse_args()
    proxy = None
    if not args.env:
        import dev_proxy
        proxy = dev_proxy.DevProxy(stage="7.2_index", purpose="сборка готового индекса").start()
        os.environ.update(proxy.core_env())
    os.environ.setdefault("DATA_DIR", str(ROOT / "data"))
    os.environ["SOLUTION_TIME_BUDGET"] = "3600"
    try:
        import config
        import index as indexmod
        import kb as kbmod
        with contextlib.redirect_stderr(io.StringIO()):
            base = kbmod.build_kb()
        idx = indexmod.Index(base)
        t0 = time.monotonic()
        idx.build_dense(cap=1800.0)
        if idx.vecs is None:
            print("Эмбеддинги не получены — индекс не собран (проверьте ollama / эмбеддер)")
            return 1
        meta = idx.save_vectors(Path(args.out), config.EMB_MODEL)
        size = Path(args.out).stat().st_size
        print(f"{Path(args.out).name}: {meta['n']} векторов × {meta['dim']}, модель {meta['emb_model']}, "
              f"{size / 1024:.0f} КБ, расчёт {time.monotonic() - t0:.1f} c")
        print(f"  kb_sha256 {meta['kb_sha256'][:16]}…  texts_sha256 {meta['texts_sha256'][:16]}…")
    finally:
        if proxy is not None:
            proxy.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
