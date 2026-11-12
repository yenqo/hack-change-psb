import type { Chat, Pending } from "../chats";
import { PsbLogo } from "./BrandBlock";

// Перечень чатов — лежит прямо на тёмном стекле оболочки, без своей карточки.
// activeId = null — открыт черновик нового чата (в списке его нет).
export function ChatSidebar(props: {
  chats: Chat[];
  activeId: string | null;
  pending: Record<string, Pending>;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}) {
  const sorted = [...props.chats].sort((a, b) => b.updatedAt - a.updatedAt);
  return (
    <aside className="flex h-full min-h-0 flex-col text-white">
      <div className="px-6 pt-6 pb-5">
        <PsbLogo theme="dark" className="h-8 w-auto" />
        <div className="mt-2 text-sm font-semibold text-white/85">Юридический ассистент</div>
      </div>

      <div className="px-4">
        <button
          onClick={props.onNew}
          disabled={props.activeId === null}
          title={props.activeId === null ? "Новый чат уже открыт" : "Начать новый чат"}
          className="flex w-full items-center justify-center gap-2 rounded-2xl bg-psb-orange px-4 py-2.5 text-sm font-semibold text-white shadow-lg shadow-psb-orange/25 transition hover:brightness-110 disabled:opacity-40 disabled:shadow-none"
        >
          <span className="text-lg leading-none">+</span> Новый чат
        </button>
      </div>

      <div className="px-6 pt-6 pb-2 text-[11px] font-bold tracking-[0.12em] text-white/60 uppercase">Чаты</div>
      <ul className="scroll-thin min-h-0 flex-1 space-y-0.5 overflow-y-auto px-3 pb-4">
        {sorted.length === 0 && <li className="px-3 py-2 text-sm font-medium text-white/70">Чатов пока нет — задайте первый вопрос.</li>}
        {sorted.map((c) => {
          const p = props.pending[c.id];
          const active = c.id === props.activeId;
          return (
            <li key={c.id} className="group relative">
              <button
                onClick={() => props.onSelect(c.id)}
                className={`w-full rounded-xl px-3 py-2.5 pr-8 text-left transition ${
                  active ? "bg-white/18 text-white" : "text-white/90 hover:bg-white/10 hover:text-white"
                }`}
              >
                {active && <span className="absolute top-3 bottom-3 left-0 w-1 rounded-full bg-psb-orange" />}
                <div className="line-clamp-2 text-[14px] leading-snug font-semibold">{c.title}</div>
                <div className="mt-1 flex items-center gap-1.5 text-xs font-medium text-white/65">
                  {p ? (
                    <>
                      <span className={`h-2 w-2 rounded-full ${p === "queued" ? "bg-amber-300" : "animate-pulse bg-psb-orange"}`} />
                      <span className={p === "queued" ? "text-amber-200" : "text-orange-200"}>
                        {p === "queued" ? "в очереди" : "отвечает…"}
                      </span>
                    </>
                  ) : (
                    <>
                      {c.items.filter((m) => m.role === "user").length} вопр. · {formatTime(c.updatedAt)}
                    </>
                  )}
                </div>
              </button>
              <button
                onClick={() => props.onDelete(c.id)}
                title="Удалить чат"
                className="absolute top-2.5 right-2 hidden rounded-full px-1.5 text-white/50 group-hover:block hover:bg-white/15 hover:text-white"
              >
                ✕
              </button>
            </li>
          );
        })}
      </ul>
    </aside>
  );
}

function formatTime(ts: number): string {
  const d = new Date(ts);
  const today = new Date().toDateString() === d.toDateString();
  return today
    ? d.toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" })
    : d.toLocaleDateString("ru-RU", { day: "numeric", month: "short" });
}
