import type { Citation, ContextDocument, ContextReason, Result, Span } from "../api";
import { citationLabel } from "./AnswerCard";
import { DocumentViewer } from "./DocumentViewer";

export type DocsTab = "citations" | "context" | "document";

export interface OpenDoc {
  docId: string;
  highlight: Span | null;
  marks: Span[];
}

const REASON: Record<ContextReason, string> = {
  company_full: "целиком",
  search: "найдено поиском",
  related: "связанный документ",
  short_article: "статья целиком",
  general_point: "общий пункт",
  cross_ref: "отсылка из закона",
  date_applicable: "по дате",
};

// Правая колонка: на чём основан выбранный ответ и просмотр документов.
export function DocsPanel(props: {
  tab: DocsTab;
  onTab: (t: DocsTab) => void;
  result?: Result;
  open: OpenDoc | null;
  onOpen: (d: OpenDoc) => void;
  docLabels: Record<string, string>;
  onClose: () => void; // свернуть панель (или закрыть шторку на узком экране)
}) {
  const r = props.result;
  const tabs: { value: DocsTab; label: string; count?: number }[] = [
    { value: "citations", label: "Цитаты", count: r?.citations.length },
    { value: "document", label: "Документ" },
    { value: "context", label: "Контекст", count: r?.context?.documents?.length },
  ];

  function openCitation(c: Citation) {
    props.onOpen({ docId: c.doc_id, highlight: c.found ? { start: c.start, end: c.end } : null, marks: [] });
  }

  function openContextDoc(d: ContextDocument) {
    const marks = d.included === "fragments" ? d.fragments : d.search_hits;
    props.onOpen({ docId: d.doc_id, highlight: null, marks });
  }

  return (
    <aside className="flex min-h-0 flex-1 flex-col bg-white">
      <div className="px-5 pt-5 pb-3">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-psb-navy">Документы</h2>
          <button onClick={props.onClose} aria-label="Скрыть документы" title="Скрыть документы" className="rounded-full px-2 py-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700">
            ✕
          </button>
        </div>
        <div className="mt-3 flex gap-1.5">
          {tabs.map((t) => (
            <button
              key={t.value}
              onClick={() => props.onTab(t.value)}
              className={`rounded-full px-3.5 py-1.5 text-sm font-medium transition ${
                props.tab === t.value ? "bg-psb-blue text-white shadow-sm" : "bg-slate-200/70 text-slate-600 hover:bg-slate-200"
              }`}
            >
              {t.label}
              {t.count ? <span className="ml-1.5 opacity-70">{t.count}</span> : null}
            </button>
          ))}
        </div>
      </div>

      {props.tab === "document" ? (
        props.open ? (
          <DocumentViewer docId={props.open.docId} highlight={props.open.highlight} marks={props.open.marks} />
        ) : (
          <Empty>Откройте цитату из ответа или документ из базы знаний — он появится здесь.</Empty>
        )
      ) : !r ? (
        <Empty>Выберите ответ в чате, чтобы увидеть, на чём он основан.</Empty>
      ) : props.tab === "citations" ? (
        <div className="scroll-thin min-h-0 flex-1 space-y-3 overflow-y-auto px-5 pb-5">
          {r.citations.length === 0 && <p className="text-sm text-slate-500">Ответ не ссылается на документы базы.</p>}
          {r.citations.map((c, i) => (
            <button
              key={i}
              onClick={() => openCitation(c)}
              className="block w-full rounded-2xl border border-slate-200 bg-white p-4 text-left shadow-sm transition hover:border-psb-sky"
            >
              <div className="flex items-start justify-between gap-2">
                <span className="text-[15px] leading-snug font-medium text-psb-navy">{citationLabel(c, props.docLabels)}</span>
                <span className="shrink-0 rounded-full bg-slate-100 px-2 py-0.5 text-[11px] text-slate-500">
                  {c.doc_kind === "law" ? "закон" : "документ"}
                </span>
              </div>
              {c.found ? (
                <p className="mt-2 line-clamp-6 text-sm leading-relaxed text-slate-600">{c.quote}</p>
              ) : (
                <p className="mt-2 text-sm text-red-700">Такого пункта в документе нет — ссылка не подтверждена.</p>
              )}
              {c.applies_to_event_date === true && <p className="mt-2 text-xs text-violet-700">Действовал на дату события</p>}
              {c.applies_to_event_date === false && <p className="mt-2 text-xs text-rose-700">Не действовал на дату события</p>}
            </button>
          ))}
        </div>
      ) : (
        <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-5 pb-5">
          {/* При ошибке ядро присылает пустой context: {} — documents там нет */}
          {!r.context?.documents?.length ? (
            <p className="text-sm text-slate-500">Ядро не передало контекст для этого ответа.</p>
          ) : (
            <>
              <p className="mb-3 text-sm text-slate-500">Что получила модель для ответа. Подсвечено то, что нашёл поиск.</p>
              {r.context.date_note && (
                <div className="mb-3 rounded-2xl bg-violet-50 p-3 text-xs leading-relaxed text-violet-900">{r.context.date_note}</div>
              )}
              {(["company", "law"] as const).map((kind) => {
                const docs = (r.context?.documents ?? []).filter((d) => d.doc_kind === kind);
                if (!docs.length) return null;
                return (
                  <div key={kind} className="mb-4">
                    <h3 className="mb-2 text-xs font-semibold tracking-wide text-slate-500 uppercase">
                      {kind === "company" ? "Документы компании" : "Законы"}
                    </h3>
                    <div className="space-y-1.5">
                      {docs.map((d) => (
                        <button
                          key={d.doc_id}
                          onClick={() => openContextDoc(d)}
                          className={`block w-full rounded-xl border px-3 py-2.5 text-left transition hover:border-psb-sky ${
                            d.cited ? "border-psb-orange/60 bg-orange-50/70" : "border-slate-200 bg-white"
                          }`}
                        >
                          <div className="text-sm leading-snug font-medium text-slate-800">{props.docLabels[d.doc_id] ?? d.doc_title}</div>
                          <div className="mt-1 flex flex-wrap gap-1.5 text-[11px]">
                            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-slate-600">{REASON[d.reason] ?? d.reason}</span>
                            {d.cited && <span className="rounded-full bg-psb-orange px-2 py-0.5 text-white">в ответе</span>}
                            {d.applies_to_event_date === false && (
                              <span className="rounded-full bg-rose-100 px-2 py-0.5 text-rose-800">не действовал на дату</span>
                            )}
                            {d.included === "fragments" && (
                              <span className="rounded-full bg-sky-100 px-2 py-0.5 text-sky-800">фрагментов: {d.fragments.length}</span>
                            )}
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
                );
              })}
            </>
          )}
        </div>
      )}
    </aside>
  );
}

function Empty(props: { children: React.ReactNode }) {
  return <p className="px-5 text-sm leading-relaxed text-slate-500">{props.children}</p>;
}
