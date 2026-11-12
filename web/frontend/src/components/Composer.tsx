import { useState } from "react";

// Строка ввода под чатом: Enter — отправить, Shift+Enter — перенос строки, кнопка справа.
export function Composer(props: { disabled: boolean; placeholder: string; onSend: (text: string) => void }) {
  const [text, setText] = useState("");

  function send() {
    const t = text.trim();
    if (!t || props.disabled) return;
    setText("");
    props.onSend(t);
  }

  return (
    <div className="flex items-end gap-2 rounded-2xl border border-slate-200 bg-white p-2 pl-4 shadow-sm transition focus-within:border-psb-sky focus-within:ring-4 focus-within:ring-psb-sky/15">

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            send();
          }
        }}
        rows={1}
        placeholder={props.placeholder}
        className="max-h-40 min-h-[44px] flex-1 resize-none bg-transparent py-2.5 text-[15px] outline-none placeholder:text-slate-400 field-sizing-content"
      />
      <button
        onClick={send}
        disabled={props.disabled || !text.trim()}
        aria-label="Отправить"
        className="grid h-11 w-11 shrink-0 place-items-center rounded-full bg-gradient-to-br from-psb-blue to-psb-orange text-white shadow-md transition hover:brightness-110 disabled:opacity-40"
      >
        <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M5 12h14M13 6l6 6-6 6" />
        </svg>
      </button>
    </div>
  );
}
