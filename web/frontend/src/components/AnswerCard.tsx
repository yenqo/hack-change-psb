import type { Citation, Result, Status, Warning } from "../api";

// Подписи и цвета статусов ответа (status из контракта ядра).
const STATUS: Record<Status, { label: string; cls: string }> = {
  answer: { label: "Ответ по документам", cls: "bg-emerald-50 text-emerald-700 ring-emerald-200" },
  partial: { label: "Неполные данные", cls: "bg-amber-50 text-amber-800 ring-amber-200" },
  no_info: { label: "Нет данных в базе", cls: "bg-slate-100 text-slate-700 ring-slate-200" },
  premise_error: { label: "Исправлена ошибка в вопросе", cls: "bg-orange-50 text-orange-800 ring-orange-200" },
  clarify: { label: "Нужно уточнение", cls: "bg-sky-50 text-sky-800 ring-sky-200" },
  out_of_scope: { label: "Вне компетенции", cls: "bg-slate-100 text-slate-700 ring-slate-200" },
  error: { label: "Ошибка", cls: "bg-red-50 text-red-700 ring-red-200" },
};

// Заголовки предупреждений (warnings[].code). edition_note показываем отдельно, свёрнутым блоком.
const WARNING_TITLE: Record<string, string> = {
  law_current_edition_only: "В базе только действующая редакция закона",
  removed_instructions: "Из документа удалена встроенная инструкция для ИИ",
  unverified_numbers: "Есть числа, которых нет в документах (возможно, расчёт)",
  clause_not_found: "Ответ ссылается на пункт, которого нет в документе",
  timeout: "Ответ не уложился во время",
  llm_unavailable: "Модель недоступна",
};

export function citationLabel(c: Citation, docLabels: Record<string, string>): string {
  const doc = docLabels[c.doc_id] ?? c.doc_title;
  if (!c.clause) return doc;
  const clause = /^\d/.test(c.clause) ? `п. ${c.clause}` : c.clause;
  return `${doc}, ${clause}`;
}

export function AnswerCard(props: {
  result: Result;
  selected: boolean;
  docLabels: Record<string, string>;
  onSelect: () => void;
  onCitation: (c: Citation) => void;
  onRetry?: () => void;
}) {
  const { result: r } = props;
  const status = STATUS[r.status] ?? STATUS.answer;
  const editionNote = r.warnings.find((w) => w.code === "edition_note");
  const others = r.warnings.filter((w) => w.code !== "edition_note" && r.status !== "error");
  const total = r.timing_ms.total;

  return (
    <div
      onClick={props.onSelect}
      className={`max-w-[88%] cursor-pointer rounded-3xl rounded-bl-md border bg-white px-5 py-4 text-[15px] shadow-sm transition ${
        props.selected ? "border-psb-sky ring-4 ring-psb-sky/20" : "border-slate-200 hover:border-slate-300"
      }`}
    >
      <div className="mb-2.5 flex items-center gap-2">
        <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ${status.cls}`}>{status.label}</span>
        {total !== undefined && <span className="text-xs text-slate-400">{(total / 1000).toFixed(1)} с</span>}
      </div>

      <p className="leading-relaxed whitespace-pre-wrap text-slate-800">{r.text}</p>

      {r.citations.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {r.citations.map((c, i) => (
            <button
              key={i}
              onClick={(e) => {
                e.stopPropagation();
                props.onCitation(c);
              }}
              title={c.found ? c.quote : "Такого пункта в документе нет"}
              className={`rounded-full px-3 py-1 text-left text-xs ring-1 ${
                c.found
                  ? "bg-psb-blue/5 text-psb-blue ring-psb-blue/25 hover:bg-psb-blue/10"
                  : "bg-red-50 text-red-700 line-through ring-red-200"
              }`}
            >
              {citationLabel(c, props.docLabels)}
            </button>
          ))}
        </div>
      )}

      {/* Справка ядра о редакциях на дату из вопроса: зелёная пометка, по нажатию раскрывается текст справки */}
      {editionNote && (
        <details className="group/ed mt-3 rounded-xl bg-emerald-50 px-3 py-2 text-xs text-emerald-900 ring-1 ring-emerald-200" onClick={(e) => e.stopPropagation()}>
          <summary className="flex cursor-pointer list-none items-center gap-1.5 font-semibold [&::-webkit-details-marker]:hidden">
            <svg viewBox="0 0 24 24" className="h-4 w-4 shrink-0 text-emerald-600" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M5 12.5l4.5 4.5L19 7.5" />
            </svg>
            Учтена редакция на дату
            <svg viewBox="0 0 24 24" className="ml-auto h-3.5 w-3.5 text-emerald-700 transition-transform group-open/ed:rotate-180" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
              <path d="M6 9l6 6 6-6" />
            </svg>
          </summary>
          <p className="mt-1.5 leading-relaxed">{editionNote.message}</p>
        </details>
      )}

      {/* Ответ сослался на документ, который на дату события не действовал — это предупреждение, а не «учтено» */}
      {r.citations.some((c) => c.applies_to_event_date === false) && (
        <div className="mt-2 rounded-xl bg-amber-50 px-3 py-2 text-xs text-amber-900 ring-1 ring-amber-200">
          <span className="font-semibold">Ответ ссылается на документ, не действовавший на дату события:</span>{" "}
          {[...new Set(r.citations.filter((c) => c.applies_to_event_date === false).map((c) => props.docLabels[c.doc_id] ?? c.doc_title))].join(", ")}
        </div>
      )}

      {others.map((w: Warning, i) => (
        <div key={i} className="mt-2 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-900">
          <span className="font-medium">{WARNING_TITLE[w.code] ?? w.code}.</span> {w.message}
        </div>
      ))}

      {/* Причина сбоя от ядра: тайм-аут, модель недоступна и т. п. */}
      {r.status === "error" && r.warnings[0]?.message && (
        <p className="mt-2 text-xs text-slate-500">Причина: {r.warnings[0].message}</p>
      )}

      {r.status === "error" && props.onRetry && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            props.onRetry?.();
          }}
          className="mt-3 rounded-md border border-slate-300 px-3 py-1.5 text-xs hover:bg-slate-50"
        >
          Повторить
        </button>
      )}
    </div>
  );
}
