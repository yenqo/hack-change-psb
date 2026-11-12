"""Нарезка базы знаний на фрагменты — как её видит ядро (kb.build_kb), без вызовов модели.

    python tools/show_chunks.py "Договор аренды"            # один документ: метки, длины, смещения, начало текста
    python tools/show_chunks.py "НК РФ, ст. 164" --full     # с полным текстом фрагментов и поисковым текстом
    python tools/show_chunks.py --list                      # все документы: вид, длина, число фрагментов
    python tools/show_chunks.py --dump eval/chunks          # служебная папка: по файлу на документ + README

Документ ищется по подстроке метки или названия (без учёта регистра). Фрагменты документов компании нужны только для
поиска, подсветки и запасного ответа — в модель документы компании идут целиком; фрагменты законов идут в модель.
После изменения нарезки (kb.py) папку нужно пересоздать: --dump.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load():
    import kb as kbmod
    with contextlib.redirect_stderr(io.StringIO()):
        return kbmod, kbmod.build_kb()


def find(base, query: str):
    q = query.lower()
    hits = [d for d in base.docs if q in d.label.lower() or q in (d.title or "").lower()]
    exact = [d for d in hits if d.label.lower() == q]
    return exact or hits


def doc_header(d, chunks) -> list[str]:
    kind = "документ компании" if d.kind == "company" else "закон"
    lines = [f"# {d.label}", "",
             f"- Вид: {kind}; длина {len(d.text)} символов; фрагментов {len(chunks)}"]
    if d.kind == "company":
        lines.append("- В модель документ идёт **целиком**; фрагменты — для поиска, подсветки в вебе и запасного ответа")
        if d.status:
            lines.append(f"- Статус: {d.status}")
    else:
        lines.append(f"- Заголовок для модели: {d.title}")
        if d.export_date:
            lines.append(f"- Дата выгрузки: {d.export_date}")
        for n in d.art_notes:
            lines.append(f"- Пометка статьи: {n}")
        lines.append(f"- Целиком (если в топ-4 поиска): {'да' if len(d.body) <= 7000 else 'нет'} — длина текста статьи {len(d.body)}")
    if d.removed:
        lines.append(f"- Вырезано встроенных инструкций: {len(d.removed)}")
    return lines


def chunk_block(c, full: bool, first: int = 200) -> list[str]:
    head = f"## [{c.idx}] {c.label}  ·  {len(c.text)} симв.  ·  смещения {c.start}–{c.end}"
    if full:
        return [head, "", "```", c.text, "```", "", "<details><summary>Поисковый текст (BM25 и эмбеддинги)</summary>", "",
                "```", c.search, "```", "</details>", ""]
    t = c.text.replace("\n", " ⏎ ")
    return [head, "", t[:first] + ("…" if len(t) > first else ""), ""]


def render(d, chunks, full: bool) -> str:
    return "\n".join(doc_header(d, chunks) + [""] + [ln for c in chunks for ln in chunk_block(c, full)])


def dump(kbmod, base, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.md"):
        old.unlink()
    rows = []
    for d in base.docs:
        chunks = [c for c in base.chunks if c.doc == d.idx]
        name = f"{'law' if d.kind == 'law' else 'company'}_{kbmod.slug(d.label)}.md"
        (out / name).write_text(render(d, chunks, full=True) + "\n", encoding="utf-8")
        rows.append((d, chunks, name))
    lines = ["# Нарезка базы знаний (служебная папка)", "",
             "Сгенерировано `python tools/show_chunks.py --dump eval/chunks` из текущего `kb.py` — руками не править; "
             "после изменения нарезки пересоздать. Полный текст фрагментов и поисковый текст — в файлах документов.", "",
             f"Документов: {len(base.docs)}; фрагментов: {len(base.chunks)}.", "",
             "| Документ | Вид | Символов | Фрагментов | Файл |", "|---|---|---|---|---|"]
    for d, chunks, name in rows:
        lines.append(f"| {d.label} | {'закон' if d.kind == 'law' else 'компания'} | {len(d.text)} | {len(chunks)} | [{name}]({name}) |")
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Записано {len(rows)} документов в {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Нарезка базы знаний на фрагменты")
    ap.add_argument("query", nargs="?", help="подстрока метки или названия документа")
    ap.add_argument("--full", action="store_true", help="полный текст фрагментов и поисковый текст")
    ap.add_argument("--list", action="store_true", help="список документов")
    ap.add_argument("--dump", metavar="DIR", help="записать нарезку всех документов в папку")
    args = ap.parse_args()
    kbmod, base = load()
    if args.dump:
        dump(kbmod, base, (ROOT / args.dump) if not Path(args.dump).is_absolute() else Path(args.dump))
        return 0
    if args.list or not args.query:
        for d in base.docs:
            n = sum(1 for c in base.chunks if c.doc == d.idx)
            print(f"{'закон   ' if d.kind == 'law' else 'компания'}  {len(d.text):6d} симв.  {n:3d} фрагм.  {d.label}")
        return 0
    docs = find(base, args.query)
    if not docs:
        print(f"Документ «{args.query}» не найден; список — --list")
        return 1
    for d in docs:
        print(render(d, [c for c in base.chunks if c.doc == d.idx], args.full))
    return 0


if __name__ == "__main__":
    sys.exit(main())
