// Список чатов хранится в браузере (localStorage). Сервер историю не хранит: каждый запрос несёт её целиком.
import type { Message, Result } from "./api";

export interface ChatItem extends Message {
  result?: Result; // у ответов ассистента — полный результат ядра
}

export interface Chat {
  id: string;
  title: string;
  items: ChatItem[];
  updatedAt: number;
}

const KEY = "psb-chats-v2";

export function newChat(items: ChatItem[] = []): Chat {
  return { id: Math.random().toString(36).slice(2, 10), title: titleFrom(items), items, updatedAt: Date.now() };
}

// Состояние ответа в чате: этап работы ядра или "queued" — модель занята ответом в другом чате.
export type Pending = "queued" | "search" | "generate" | "verify" | "done";

// В списке только чаты с сообщениями: новый чат — это черновик, он попадает в список при первой отправке.
export function loadChats(): Chat[] {
  try {
    const chats = JSON.parse(localStorage.getItem(KEY) || "[]") as Chat[];
    return chats.filter((c) => c.items.length > 0);
  } catch {
    return [];
  }
}

export function saveChats(chats: Chat[]) {
  try {
    localStorage.setItem(KEY, JSON.stringify(chats));
  } catch {
    /* приватный режим или переполнение — просто не сохраняем */
  }
}

export function titleFrom(items: ChatItem[]): string {
  const first = items.find((m) => m.role === "user")?.content ?? "Новый чат";
  return first.length > 60 ? first.slice(0, 57) + "…" : first;
}
