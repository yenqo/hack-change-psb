import { useEffect, useState } from "react";

// true, если ширина окна подходит под медиазапрос; обновляется при изменении размера окна.
export function useMedia(query: string): boolean {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const onChange = () => setMatches(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [query]);
  return matches;
}

// Выезжающая шторка. По умолчанию — поверх всей страницы; contained — внутри ближайшего
// родителя с position: relative (документы выезжают внутри карточки чата).
// Закрывается по клику на затемнение и по Esc.
export function Drawer(props: {
  side: "left" | "right";
  open: boolean;
  onClose: () => void;
  contained?: boolean;
  children: React.ReactNode;
}) {
  useEffect(() => {
    if (!props.open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && props.onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [props.open, props.onClose]);

  const hidden = props.side === "left" ? "-translate-x-full" : "translate-x-full";
  const position = props.contained ? "absolute z-30" : "fixed z-40";
  const panel = props.contained
    ? "w-[92%] max-w-md shadow-2xl"
    : `w-[88%] p-3 ${props.side === "left" ? "max-w-sm" : "max-w-lg"}`;
  return (
    <div className={`${position} inset-0 overflow-hidden ${props.open ? "" : "pointer-events-none"}`} aria-hidden={!props.open}>
      <div
        onClick={props.onClose}
        className={`absolute inset-0 bg-psb-navy/35 backdrop-blur-[2px] transition-opacity duration-300 ${props.open ? "opacity-100" : "opacity-0"}`}
      />
      <div
        className={`absolute inset-y-0 ${props.side === "left" ? "left-0" : "right-0"} flex flex-col transition-transform duration-300 ${panel} ${
          props.open ? "translate-x-0" : hidden
        }`}
      >
        {props.children}
      </div>
    </div>
  );
}
