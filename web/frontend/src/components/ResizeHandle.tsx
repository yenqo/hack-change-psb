import { useEffect, useRef, useState } from "react";

// Ширина панели документов: безопасные пределы и сохранение в браузере.
export const DOCS_DEFAULT = 420;
const DOCS_MIN = 320;
const DOCS_MAX = 720;
const CHAT_MIN = 400; // чату всегда остаётся не меньше этой ширины
const KEY = "psb-docs-width";

// Ограничивает ширину панели: не уже DOCS_MIN, не шире DOCS_MAX и не шире, чем оставляет чату CHAT_MIN.
export function clampDocsWidth(width: number, containerWidth: number): number {
  const max = Math.max(DOCS_MIN, Math.min(DOCS_MAX, containerWidth - CHAT_MIN));
  return Math.round(Math.min(Math.max(width, DOCS_MIN), max));
}

export function loadDocsWidth(): number {
  try {
    const w = Number(localStorage.getItem(KEY));
    return Number.isFinite(w) && w > 0 ? w : DOCS_DEFAULT;
  } catch {
    return DOCS_DEFAULT;
  }
}

export function saveDocsWidth(width: number) {
  try {
    localStorage.setItem(KEY, String(width));
  } catch {
    /* не сохраняем — не страшно */
  }
}

// Ширина элемента; обновляется при изменении размера окна и раскладки.
export function useElementWidth(ref: React.RefObject<HTMLElement | null>): number {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, [ref]);
  return width;
}

// Вертикальная граница, которую тянут мышью или пальцем. Двойной клик — стандартная ширина,
// стрелки влево/вправо — шаг 20px. onWidth получает ширину панели, отсчитанную от правого края контейнера.
export function ResizeHandle(props: {
  containerRef: React.RefObject<HTMLElement | null>;
  width: number;
  onWidth: (w: number) => void;
}) {
  const dragging = useRef(false);
  const [active, setActive] = useState(false);

  function onPointerDown(e: React.PointerEvent) {
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    dragging.current = true;
    setActive(true);
    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";
  }

  function onPointerMove(e: React.PointerEvent) {
    if (!dragging.current || !props.containerRef.current) return;
    const rect = props.containerRef.current.getBoundingClientRect();
    props.onWidth(rect.right - e.clientX);
  }

  function onPointerUp() {
    if (!dragging.current) return;
    dragging.current = false;
    setActive(false);
    document.body.style.userSelect = "";
    document.body.style.cursor = "";
  }

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label="Ширина панели документов"
      aria-valuenow={props.width}
      tabIndex={0}
      title="Потяните, чтобы изменить ширину. Двойной клик — стандартная ширина"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onDoubleClick={() => {
        props.onWidth(DOCS_DEFAULT);
      }}
      onKeyDown={(e) => {
        if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
          e.preventDefault();
          props.onWidth(props.width + (e.key === "ArrowLeft" ? 20 : -20));
        }
      }}
      className="group absolute inset-y-0 -left-1.5 z-10 flex w-3 cursor-col-resize justify-center outline-none"
    >
      <span
        className={`h-full w-px transition-colors ${
          active ? "w-0.5 bg-psb-blue" : "bg-slate-200 group-hover:w-0.5 group-hover:bg-psb-sky group-focus:bg-psb-sky"
        }`}
      />
    </div>
  );
}
