"""Нарезка базы знаний на фрагменты — как её видит ядро (kb.build_kb), без вызовов модели.

    python tools/show_chunks.py "Договор аренды"            # один документ: метки, длины, смещения, начало текста
    python tools/show_chunks.py "НК РФ, ст. 164" --full     # с полным текстом фрагментов и поисковым текстом
    python tools/show_chunks.py --list                      # все документы: вид, длина, число фрагментов
    python tools/show_chunks.py --dump eval/chunks          # служебная папка: по файлу на документ + README
    python tools/show_chunks.py --html eval/chunks/index.html  # одна страница: все документы, границы фрагментов,
                                                            # переключатель «текущая нарезка / с B19–B20»

Документ ищется по подстроке метки или названия (без учёта регистра). Фрагменты документов компании нужны только для
поиска, подсветки и запасного ответа — в модель документы компании идут целиком; фрагменты законов идут в модель.
После изменения нарезки (kb.py) папку нужно пересоздать: --dump.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
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
             "Страница со всеми документами, границами фрагментов и переключателем «текущая нарезка / с B19–B20» — "
             "[index.html](index.html) (открыть в браузере; пересоздаётся `--html eval/chunks/index.html`).", "",
             "| Документ | Вид | Символов | Фрагментов | Файл |", "|---|---|---|---|---|"]
    for d, chunks, name in rows:
        lines.append(f"| {d.label} | {'закон' if d.kind == 'law' else 'компания'} | {len(d.text)} | {len(chunks)} | [{name}]({name}) |")
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Записано {len(rows)} документов в {out}")


def collect(base) -> dict:
    """Нарезка в виде данных для страницы: документы, тексты, фрагменты со смещениями."""
    docs = []
    for d in base.docs:
        chunks = [c for c in base.chunks if c.doc == d.idx]
        docs.append({"label": d.label, "kind": d.kind, "title": d.title or d.label, "status": d.status or "",
                     "export_date": d.export_date or "", "art_notes": list(d.art_notes), "removed": len(d.removed),
                     "whole": d.kind == "company" or len(d.body) <= 7000, "text": d.text,
                     "chunks": [{"label": c.label, "text": c.text, "search": c.search, "start": c.start, "end": c.end}
                                for c in chunks]})
    return {"docs": docs, "n_chunks": len(base.chunks)}


def variant(flag: str) -> dict:
    """Нарезка при SOLUTION_CHUNK_FIXES=flag — в отдельном процессе (флаги ядра читаются при импорте)."""
    env = dict(os.environ, SOLUTION_CHUNK_FIXES=flag)
    out = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--json"], env=env, cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def write_html(path: Path) -> None:
    data = {"current": variant("0"), "fixes": variant("1")}
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HTML_TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    print(f"Записано {path} ({path.stat().st_size // 1024} КБ)")


HTML_TEMPLATE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Нарезка базы знаний</title>
<style>
:root { --bg:#fbfaf7; --panel:#ffffff; --ink:#1f2328; --muted:#6b7280; --line:#e5e2da; --accent:#2f5d8a;
  --gap:#eceae4; --c0:#fde7c7; --c1:#d7ecf8; --c2:#e3f2d9; --c3:#f6dcea; --c4:#e8e1f7; --c5:#fbf3c4; --sel:#2f5d8a; }
@media (prefers-color-scheme: dark) { :root { --bg:#16181c; --panel:#1e2127; --ink:#e6e6e6; --muted:#9aa0a6; --line:#2e323a;
  --accent:#8db8e6; --gap:#2a2d33; --c0:#4a3a22; --c1:#21384a; --c2:#2c3f27; --c3:#46293a; --c4:#352d4a; --c5:#454020; --sel:#8db8e6; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 -apple-system,"Segoe UI",Roboto,sans-serif; }
header { padding:12px 16px; border-bottom:1px solid var(--line); display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
header h1 { font-size:16px; margin:0; }
header .stat { color:var(--muted); }
.seg { display:inline-flex; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
.seg button { border:0; background:transparent; color:var(--ink); padding:6px 10px; cursor:pointer; font:inherit; }
.seg button.on { background:var(--accent); color:var(--panel); }
main { display:grid; grid-template-columns:300px 1fr; min-height:calc(100vh - 56px); }
nav { border-right:1px solid var(--line); padding:12px; overflow:auto; max-height:calc(100vh - 56px); position:sticky; top:0; }
nav input { width:100%; padding:7px 9px; border:1px solid var(--line); border-radius:8px; background:var(--panel); color:var(--ink); font:inherit; }
nav h3 { font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin:14px 0 6px; }
nav a { display:flex; justify-content:space-between; gap:8px; padding:6px 8px; border-radius:6px; color:var(--ink); text-decoration:none; cursor:pointer; }
nav a:hover { background:var(--gap); } nav a.on { background:var(--accent); color:var(--panel); }
nav a .n { color:var(--muted); font-variant-numeric:tabular-nums; } nav a.on .n { color:inherit; }
section { padding:16px 20px; min-width:0; }
.meta { color:var(--muted); margin:4px 0 12px; }
.meta span { margin-right:14px; }
.tabs { margin:8px 0 12px; }
.doc { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px; white-space:pre-wrap; word-wrap:break-word; }
.doc .ch { border-radius:3px; cursor:pointer; } .doc .gap { background:var(--gap); color:var(--muted); }
.doc .ch.sel { outline:2px solid var(--sel); }
.c0{background:var(--c0)}.c1{background:var(--c1)}.c2{background:var(--c2)}.c3{background:var(--c3)}.c4{background:var(--c4)}.c5{background:var(--c5)}
.card { background:var(--panel); border:1px solid var(--line); border-left:6px solid transparent; border-radius:10px; padding:10px 14px; margin:0 0 10px; }
.card h4 { margin:0 0 4px; font-size:14px; } .card .sub { color:var(--muted); font-size:12px; margin-bottom:6px; }
.card pre { white-space:pre-wrap; word-wrap:break-word; margin:0; font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
details { margin-top:6px; } summary { color:var(--muted); cursor:pointer; font-size:12px; }
.info { position:sticky; bottom:0; background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px 14px; margin-top:12px; }
.legend { color:var(--muted); font-size:12px; margin-bottom:8px; }
mark { background:var(--c5); color:inherit; }
@media (max-width:760px) { main { grid-template-columns:1fr; } nav { position:static; max-height:none; border-right:0; border-bottom:1px solid var(--line); } }
</style>
</head>
<body>
<header>
  <h1>Нарезка базы знаний</h1>
  <div class="seg" id="variant"><button data-v="current" class="on">Текущая нарезка</button><button data-v="fixes">С B19–B20</button></div>
  <span class="stat" id="stat"></span>
</header>
<main>
  <nav><input id="q" placeholder="Поиск по документам и фрагментам"><div id="list"></div></nav>
  <section id="view"></section>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
let V = "current", DOC = 0, TAB = "doc", SEL = -1;
const esc = s => s.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function hl(text, q) { const t = esc(text); if (!q) return t; const i = t.toLowerCase().indexOf(esc(q).toLowerCase());
  return i < 0 ? t : t.slice(0, i) + "<mark>" + t.slice(i, i + esc(q).length) + "</mark>" + t.slice(i + esc(q).length); }
function list() {
  const q = document.getElementById("q").value.trim().toLowerCase(), docs = DATA[V].docs;
  document.getElementById("stat").textContent = docs.length + " документов · " + DATA[V].n_chunks + " фрагментов";
  let html = "";
  for (const kind of ["company", "law"]) {
    html += "<h3>" + (kind === "company" ? "Документы компании" : "Законы") + "</h3>";
    docs.forEach((d, i) => { if (d.kind !== kind) return;
      if (q && !(d.label.toLowerCase().includes(q) || d.chunks.some(c => c.text.toLowerCase().includes(q)))) return;
      html += `<a class="${i === DOC ? "on" : ""}" data-i="${i}"><span>${esc(d.label)}</span><span class="n">${d.chunks.length}</span></a>`; });
  }
  const el = document.getElementById("list"); el.innerHTML = html;
  el.querySelectorAll("a").forEach(a => a.onclick = () => { DOC = +a.dataset.i; SEL = -1; render(); });
}
function docView(d) {
  const spans = d.chunks.map((c, i) => ({s: c.start, e: c.end, i})).filter(x => x.s >= 0).sort((a, b) => a.s - b.s);
  let pos = 0, out = "";
  for (const x of spans) {
    if (x.s > pos) out += `<span class="gap">${esc(d.text.slice(pos, x.s))}</span>`;
    const s = Math.max(x.s, pos);
    if (x.e > s) out += `<span class="ch c${x.i % 6}${x.i === SEL ? " sel" : ""}" data-i="${x.i}">${esc(d.text.slice(s, x.e))}</span>`;
    pos = Math.max(pos, x.e);
  }
  if (pos < d.text.length) out += `<span class="gap">${esc(d.text.slice(pos))}</span>`;
  const lost = d.chunks.filter(c => c.start < 0).length;
  return `<div class="legend">Цвет — фрагмент (щёлкните для подробностей); серым — текст вне фрагментов (служебная шапка выгрузки, пустые строки)${lost ? "; фрагментов без смещения: " + lost : ""}.</div><div class="doc">${out}</div>` + info(d);
}
function info(d) {
  if (SEL < 0) return "";
  const c = d.chunks[SEL];
  return `<div class="info"><b>[${SEL + 1}/${d.chunks.length}] ${esc(c.label)}</b> · ${c.text.length} симв. · смещения ${c.start}–${c.end}
    <details open><summary>Текст фрагмента (как в поиске и в модели)</summary><pre>${esc(c.text)}</pre></details>
    <details><summary>Поисковый текст (BM25 и эмбеддинги)</summary><pre>${esc(c.search)}</pre></details></div>`;
}
function cardsView(d) {
  const q = document.getElementById("q").value.trim();
  return d.chunks.map((c, i) => `<div class="card" style="border-left-color:var(--c${i % 6})"><h4>[${i + 1}] ${esc(c.label)}</h4>
    <div class="sub">${c.text.length} симв. · смещения ${c.start}–${c.end}</div><pre>${hl(c.text, q)}</pre>
    <details><summary>Поисковый текст</summary><pre>${esc(c.search)}</pre></details></div>`).join("");
}
function render() {
  list();
  const d = DATA[V].docs[DOC], lens = d.chunks.map(c => c.text.length);
  const meta = [d.kind === "company" ? "документ компании — в модель идёт целиком" : (d.whole ? "закон — короткая статья: из топ-4 поиска идёт целиком" : "закон — в модель идут найденные фрагменты"),
    `${d.text.length} символов`, `${d.chunks.length} фрагментов`, lens.length ? `длина фрагмента ${Math.min(...lens)}–${Math.max(...lens)}` : "",
    d.status, d.export_date ? "выгрузка " + d.export_date : "", d.removed ? "вырезано инструкций: " + d.removed : ""].filter(Boolean);
  document.getElementById("view").innerHTML = `<h2 style="margin:0">${esc(d.title)}</h2>
    <div class="meta">${meta.map(m => "<span>" + esc(m) + "</span>").join("")}</div>
    ${d.art_notes.length ? `<div class="meta">Пометки статьи: ${d.art_notes.map(esc).join(" · ")}</div>` : ""}
    <div class="seg tabs" id="tabs"><button data-t="doc" class="${TAB === "doc" ? "on" : ""}">Документ с границами</button><button data-t="cards" class="${TAB === "cards" ? "on" : ""}">Фрагменты</button></div>
    ${TAB === "doc" ? docView(d) : cardsView(d)}`;
  document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => { TAB = b.dataset.t; render(); });
  document.querySelectorAll(".doc .ch").forEach(s => s.onclick = () => { SEL = +s.dataset.i; render(); });
}
document.querySelectorAll("#variant button").forEach(b => b.onclick = () => {
  V = b.dataset.v; document.querySelectorAll("#variant button").forEach(x => x.classList.toggle("on", x === b));
  DOC = Math.min(DOC, DATA[V].docs.length - 1); SEL = -1; render(); });
document.getElementById("q").oninput = render;
render();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Нарезка базы знаний на фрагменты")
    ap.add_argument("query", nargs="?", help="подстрока метки или названия документа")
    ap.add_argument("--full", action="store_true", help="полный текст фрагментов и поисковый текст")
    ap.add_argument("--list", action="store_true", help="список документов")
    ap.add_argument("--dump", metavar="DIR", help="записать нарезку всех документов в папку")
    ap.add_argument("--html", metavar="FILE", help="одна HTML-страница со всеми документами и границами фрагментов")
    ap.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.html:
        write_html((ROOT / args.html) if not Path(args.html).is_absolute() else Path(args.html))
        return 0
    kbmod, base = load()
    if args.json:
        print(json.dumps(collect(base), ensure_ascii=False))
        return 0
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
