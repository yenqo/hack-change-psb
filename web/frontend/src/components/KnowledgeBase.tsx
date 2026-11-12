import type { DocInfo } from "../api";

// Вкладка «База знаний»: все документы, по которым отвечает ассистент.
export function KnowledgeBase(props: { docs: DocInfo[]; activeId?: string; onOpen: (id: string) => void }) {
  const groups: { kind: DocInfo["kind"]; title: string }[] = [
    { kind: "company", title: "Документы компании" },
    { kind: "law", title: "Законы (действующая редакция)" },
  ];
  return (
    <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-6 py-6">
      <p className="mb-5 max-w-2xl text-[15px] text-slate-600">
        Ассистент отвечает только по этим документам. Откройте любой — он появится в панели справа.
      </p>
      {props.docs.length === 0 && <p className="text-sm text-slate-500">Загрузка списка документов…</p>}
      {groups.map((g) => {
        const docs = props.docs.filter((d) => d.kind === g.kind);
        if (!docs.length) return null;
        return (
          <section key={g.kind} className="mb-7">
            <h2 className="mb-3 text-sm font-semibold tracking-wide text-slate-500 uppercase">
              {g.title} · {docs.length}
            </h2>
            <div className="grid gap-3 md:grid-cols-2">
              {docs.map((d) => (
                <button
                  key={d.doc_id}
                  onClick={() => props.onOpen(d.doc_id)}
                  className={`flex flex-col items-start rounded-2xl border bg-white p-4 text-left shadow-sm transition hover:border-psb-sky ${
                    props.activeId === d.doc_id ? "border-psb-blue ring-2 ring-psb-sky/30" : "border-slate-200"
                  }`}
                >
                  <div className="text-[15px] leading-snug font-medium text-psb-navy">{d.label}</div>
                  <div className="mt-2 flex flex-wrap gap-1.5 text-xs">
                    <span className="rounded-full bg-slate-100 px-2 py-0.5 text-slate-600">{d.type}</span>
                    {d.status && (
                      <span
                        className={`rounded-full px-2 py-0.5 ${
                          /утратил/i.test(d.status) ? "bg-rose-100 text-rose-800" : "bg-emerald-100 text-emerald-800"
                        }`}
                      >
                        {d.status}
                      </span>
                    )}
                    {d.related.length > 0 && (
                      <span className="rounded-full bg-sky-50 px-2 py-0.5 text-sky-800">связанных: {d.related.length}</span>
                    )}
                    {d.has_removed_instructions && (
                      <span className="rounded-full bg-red-50 px-2 py-0.5 text-red-700">вырезана инструкция для ИИ</span>
                    )}
                  </div>
                </button>
              ))}
            </div>
          </section>
        );
      })}
    </div>
  );
}
