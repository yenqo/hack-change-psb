import { useEffect, useRef, useState } from "react";
import type { Health, ModelLamp } from "../api";

export type Tab = "chats" | "kb";

const LAMP_COLOR: Record<ModelLamp["state"], string> = {
  green: "bg-emerald-500 shadow-emerald-500/60",
  yellow: "bg-amber-400 shadow-amber-400/60",
  red: "bg-red-500 shadow-red-500/60",
};

// Шапка карточки чата: вкладки, состояние моделей, кнопка панели документов.
export function TopBar(props: {
  tab: Tab;
  onTab: (t: Tab) => void;
  health: Health | null;
  onChats?: () => void; // задан на узком экране: открыть шторку со списком чатов
  onDocs: () => void; // показать / скрыть панель документов
  docsOpen: boolean;
  docsCount?: number;
}) {
  const lamps: ModelLamp[] = props.health?.models ?? [
    { key: "llm", name: "LLM", state: "red", detail: "Нет связи с сервером" },
    { key: "emb", name: "Emb", state: "red", detail: "Нет связи с сервером" },
    { key: "rerank", name: "Reranker", state: "red", detail: "Нет связи с сервером" },
  ];

  return (
    // Всегда одна строка. relative z-40: подсказки и список статуса лежат поверх ленты и шторки документов (z-30).
    // На узком экране три лампочки сворачиваются в одну кнопку статуса, «Документы» — в иконку со счётчиком.
    <div className="relative z-40 flex items-center justify-between gap-2 border-b border-slate-200/80 px-2.5 py-2 sm:gap-3 sm:px-5 sm:py-3">
      <div className="flex min-w-0 items-center gap-1.5 sm:gap-2">
        {props.onChats && (
          <IconButton onClick={props.onChats} label="Список чатов">
            <path d="M4 6h16M4 12h16M4 18h16" />
          </IconButton>
        )}
        <Segmented
          value={props.tab}
          onChange={props.onTab}
          options={[
            { value: "chats", label: "Чаты" },
            { value: "kb", label: "База знаний" },
          ]}
        />
      </div>

      <div className="flex shrink-0 items-center gap-1.5 sm:gap-4">
        {/* Переключатель «Кратко / Подробно» вернём, когда ядро начнёт принимать параметр style (согласовать с MLE) */}
        <StatusMenu lamps={lamps} />

        <div className="hidden items-center gap-1 md:flex">
          {lamps.map((l) => (
            <div
              key={l.key}
              tabIndex={0}
              title={`${l.name}: ${l.detail}`}
              className="group relative flex cursor-default items-center gap-1.5 rounded-full px-2 py-1.5 outline-none hover:bg-slate-200/60 focus:bg-slate-200/60 sm:px-2.5"
            >
              <span className={`h-2.5 w-2.5 rounded-full shadow-[0_0_8px] ${LAMP_COLOR[l.state]}`} />
              <span className="text-xs font-medium text-slate-700 sm:text-sm">{l.name}</span>
              <Tooltip>{l.detail}</Tooltip>
            </div>
          ))}
        </div>

        <button
          onClick={props.onDocs}
          aria-pressed={props.docsOpen}
          title={props.docsOpen ? "Скрыть документы" : "Показать документы"}
          aria-label="Документы"
          className={`flex h-9 items-center gap-1.5 rounded-full px-2.5 text-sm font-medium transition sm:px-3 ${
            props.docsOpen ? "bg-psb-blue text-white shadow-sm" : "bg-slate-200/70 text-slate-700 hover:bg-slate-200"
          }`}
        >
          <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" />
            <path d="M14 3v6h6M8 13h8M8 17h5" />
          </svg>
          <span className="hidden sm:inline">Документы</span>
          {props.docsCount ? (
            <span className={`rounded-full px-1.5 text-xs ${props.docsOpen ? "bg-white/25" : "bg-white text-slate-600"}`}>{props.docsCount}</span>
          ) : null}
        </button>
      </div>
    </div>
  );
}

// Узкий экран: одна кнопка с тремя точками-лампочками; по нажатию — список моделей с пояснениями.
// Список выровнен по правому краю шапки и не шире экрана, поэтому не уходит за край.
function StatusMenu(props: { lamps: ModelLamp[] }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const close = (e: PointerEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("pointerdown", close);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("pointerdown", close);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);

  return (
    <div ref={ref} className="md:hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-label="Состояние моделей"
        className={`flex h-9 items-center gap-1 rounded-full px-3 transition ${open ? "bg-slate-200" : "bg-slate-200/70 hover:bg-slate-200"}`}
      >
        {props.lamps.map((l) => (
          <span key={l.key} className={`h-2.5 w-2.5 rounded-full shadow-[0_0_6px] ${LAMP_COLOR[l.state]}`} />
        ))}
      </button>
      {open && (
        <div className="absolute top-full right-2.5 left-2.5 mt-2 rounded-2xl border border-slate-200 bg-white p-2 shadow-xl sm:left-auto sm:w-80">
          <div className="px-2 pt-1 pb-2 text-xs font-semibold tracking-wide text-slate-500 uppercase">Состояние моделей</div>
          {props.lamps.map((l) => (
            <div key={l.key} className="flex gap-3 rounded-xl px-2 py-2">
              <span className={`mt-1 h-2.5 w-2.5 shrink-0 rounded-full shadow-[0_0_6px] ${LAMP_COLOR[l.state]}`} />
              <div>
                <div className="text-sm font-semibold text-psb-navy">{l.name}</div>
                <div className="text-xs leading-snug text-slate-600">{l.detail}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function IconButton(props: { onClick: () => void; label: string; children: React.ReactNode }) {
  return (
    <button
      onClick={props.onClick}
      aria-label={props.label}
      title={props.label}
      className="grid h-9 w-9 place-items-center rounded-full bg-slate-200/70 text-psb-navy hover:bg-slate-200"
    >
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
        {props.children}
      </svg>
    </button>
  );
}

function Segmented<T extends string>(props: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: string }[];
  disabled?: boolean;
}) {
  return (
    <div className={`flex rounded-full bg-slate-200/70 p-1 ${props.disabled ? "opacity-50" : ""}`}>
      {props.options.map((o) => (
        <button
          key={o.value}
          disabled={props.disabled}
          onClick={() => props.onChange(o.value)}
          className={`rounded-full px-3 py-1.5 text-sm font-medium whitespace-nowrap transition sm:px-4 ${
            props.value === o.value ? "bg-white text-psb-navy shadow-sm" : "text-slate-600 hover:text-psb-navy"
          } ${props.disabled ? "cursor-not-allowed" : ""}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

// Всплывающая подсказка при наведении на родителя с классом group.
export function Tooltip(props: { children: React.ReactNode }) {
  return (
    <div className="pointer-events-none absolute top-full right-0 z-30 mt-2 hidden w-64 rounded-xl bg-psb-navy px-3 py-2 text-xs leading-snug text-white shadow-lg group-hover:block group-focus:block">
      {props.children}
    </div>
  );
}
