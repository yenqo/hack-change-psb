"""Снимает эталоны поведения с tools/baseline/solution_v1_3.py в tests/fixtures/.

    .venv/bin/python tests/make_fixtures.py

Эталоны фиксируют v1.3 и не перегенерируются при рефакторинге. Перегенерация допустима только при
осознанном изменении поведения (принятая доработка) — с отдельным коммитом и пояснением.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import characterization  # noqa: E402
import golden_runner  # noqa: E402
import harness  # noqa: E402


def write_json(path: Path, data) -> None:
    raw = json.dumps(data, ensure_ascii=False, indent=1, sort_keys=False)
    if path.suffix == ".gz":
        with gzip.GzipFile(path, "wb", mtime=0) as f:
            f.write(raw.encode("utf-8"))
    else:
        path.write_text(raw + "\n", encoding="utf-8")
    print(f"{path.relative_to(harness.ROOT)}: {path.stat().st_size // 1024} КБ")


def main() -> None:
    harness.FIXTURES.mkdir(exist_ok=True)
    for sc in golden_runner.SCENARIOS:
        res = golden_runner.run_subprocess("baseline", sc)
        for k in ("_elapsed", "_stderr", "probe_rerank_calls"):
            res.pop(k, None)
        write_json(harness.FIXTURES / f"golden_{sc}.json.gz", res)
    core = harness.load_core("baseline")
    write_json(harness.FIXTURES / "characterization.json", characterization.snapshot(core))


if __name__ == "__main__":
    main()
