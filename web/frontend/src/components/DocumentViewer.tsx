import { useEffect, useRef, useState } from "react";
import { fetchDocument, type DocFull, type Span } from "../api";

// Документ целиком внутри правой панели.
// highlight — основная подсветка (цитата ответа), marks — дополнительная (что нашёл поиск).
// Пометки редакций Гаранта — серым курсивом, вырезанные инструкции для ИИ — красным.
export function DocumentViewer(props: { docId: string; highlight?: Span | null; marks?: Span[] }) {
  const [doc, setDoc] = useState<DocFull | null>(null);
  const [error, setError] = useState<string | null>(null);
  const anchor = useRef<HTMLElement>(null);

  useEffect(() => {
    setDoc(null);
    setError(null);
    fetchDocument(props.docId)
      .then(setDoc)
      .catch((e) => setError(e.message));
  }, [props.docId]);

  // После загрузки — прокрутка к первой подсветке.
  useEffect(() => {
    anchor.current?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [doc, props.highlight, props.marks]);

  if (error) return <p className="p-5 text-sm text-red-700">{error}</p>;
  if (!doc) return <p className="p-5 text-sm text-slate-500">Загрузка документа…</p>;

  // Подсвечиваем пункт целиком: цитата в ответе может быть обрезана до ~600 символов.
  let hl = props.highlight ?? null;
  if (hl) {
    const clause = doc.clauses.find((c) => c.start === hl!.start);
    if (clause && clause.end > hl.end) hl = { start: hl.start, end: clause.end };
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="border-b border-slate-200/80 px-5 pb-3">
        <h3 className="text-base leading-snug font-semibold text-psb-navy">{doc.label}</h3>
        <div className="mt-1 flex flex-wrap items-center gap-2 text-xs">
          <span className="rounded-full bg-slate-200 px-2 py-0.5 text-slate-600">{doc.type}</span>
          {doc.status && (
            <span
              className={`rounded-full px-2 py-0.5 ${
                /утратил/i.test(doc.status) ? "bg-rose-100 text-rose-800" : "bg-emerald-100 text-emerald-800"
              }`}
            >
              {doc.status}
            </span>
          )}
        </div>
      </div>
      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-5 py-4">
        <pre className="font-sans text-[14px] leading-relaxed whitespace-pre-wrap text-slate-800">
          {renderText(doc, hl, props.marks ?? [], anchor)}
        </pre>
      </div>
    </div>
  );
}

type Kind = "hl" | "mark" | "note" | "removed";

// Режем текст по границам всех размеченных диапазонов и оформляем каждый кусок
// по тому, в какие диапазоны он попадает.
function renderText(doc: DocFull, hl: Span | null, marks: Span[], anchorRef: React.RefObject<HTMLElement | null>) {
  const ranges: (Span & { kind: Kind; original?: string })[] = [
    ...(hl ? [{ ...hl, kind: "hl" as const }] : []),
    ...marks.map((m) => ({ ...m, kind: "mark" as const })),
    ...doc.edition_notes.map((n) => ({ ...n, kind: "note" as const })),
    ...doc.removed_instructions.map((r) => ({ ...r, kind: "removed" as const })),
  ];
  const cuts = new Set([0, doc.text.length]);
  ranges.forEach((r) => {
    cuts.add(r.start);
    cuts.add(r.end);
  });
  const points = [...cuts].sort((a, b) => a - b);
  let anchorPlaced = false;
  const wantAnchor: Kind = hl ? "hl" : "mark";

  return points.slice(0, -1).map((start, i) => {
    const end = points[i + 1];
    const piece = doc.text.slice(start, end);
    const inside = ranges.filter((r) => r.start <= start && end <= r.end);
    if (inside.length === 0) return <span key={i}>{piece}</span>;

    const has = (k: Kind) => inside.some((r) => r.kind === k);
    const removed = inside.find((r) => r.kind === "removed");
    const cls = has("hl")
      ? "rounded bg-amber-200/90 ring-2 ring-amber-200/90"
      : removed
        ? "bg-red-50 text-red-700"
        : has("mark")
          ? "rounded bg-sky-100"
          : "bg-transparent";
    const extra = has("note") ? " text-slate-400 italic" : "";
    const ref = !anchorPlaced && has(wantAnchor) ? ((anchorPlaced = true), anchorRef) : undefined;
    return (
      <mark key={i} ref={ref} className={cls + extra} title={removed ? `Удалено: ${removed.original}` : undefined}>
        {piece}
      </mark>
    );
  });
}
