// Типы и вызовы API бэкенда (web/server.py). Ответ ассистента — контракт ядра 1.0 (docs/integration.md).

export type Role = "user" | "assistant";

export interface Message {
  role: Role;
  content: string;
}

export type Status = "answer" | "partial" | "no_info" | "premise_error" | "clarify" | "out_of_scope" | "error";

export interface Citation {
  doc_id: string;
  doc_title: string;
  doc_kind: "company" | "law";
  clause: string | null;
  quote: string;
  start: number;
  end: number;
  found: boolean;
  applies_to_event_date: boolean | null;
}

export interface Warning {
  code: string;
  message: string;
  doc_id?: string;
}

// Контекст модели (контракт 1.1): какие документы и фрагменты ушли в запрос и почему. Текстов нет —
// только doc_id и смещения в тексте документа из /api/documents/{doc_id}.
export type ContextReason =
  | "company_full" // документы компании идут целиком
  | "search" // найден поиском
  | "related" // связан с найденным (договор ↔ доп. соглашение, регламент ↔ письмо)
  | "short_article" // короткая статья закона — целиком
  | "general_point" // общий пункт большой статьи
  | "cross_ref" // пункт, на который ссылается найденный (флаг R6)
  | "date_applicable"; // зарезервировано

export interface SearchHit extends Span {
  clause: string | null;
  rank: number;
}

export interface ContextFragment extends Span {
  clause: string | null;
  rank: number | null;
  reason: ContextReason;
  edition_note: boolean;
}

export interface ContextDocument {
  doc_id: string;
  doc_title: string;
  doc_kind: "company" | "law";
  included: "whole" | "fragments";
  reason: ContextReason;
  search_hits: SearchHit[]; // для included = "whole"
  fragments: ContextFragment[]; // для included = "fragments"
  applies_to_event_date: boolean | null;
  cited: boolean;
}

export interface AnswerContext {
  documents: ContextDocument[];
  date_note: string;
  removed_from_context: (Span & { doc_id: string; note: string })[];
}

export interface Result {
  contract_version: string;
  status: Status;
  text: string;
  citations: Citation[];
  warnings: Warning[];
  context?: Partial<AnswerContext>; // с версии 1.1; при ошибке приходит пустой объект {}
  timing_ms: Record<string, number>;
  debug?: Record<string, unknown>;
}

export type Stage = "search" | "generate" | "verify" | "done";

export interface DocInfo {
  doc_id: string;
  title: string;
  label: string;
  kind: "company" | "law";
  type: string;
  status: string;
  valid_from: string | null;
  valid_to: string | null;
  related: string[];
  has_removed_instructions: boolean;
}

export interface Span {
  start: number;
  end: number;
}

export interface DocFull extends DocInfo {
  text: string;
  clauses: (Span & { clause: string; title: string })[];
  edition_notes: (Span & { text: string })[];
  removed_instructions: (Span & { original: string; note: string })[];
}

export interface Example {
  id: string;
  title: string;
  messages: Message[];
}

export interface ModelLamp {
  key: "llm" | "emb" | "rerank";
  name: string;
  state: "green" | "yellow" | "red";
  detail: string;
}

export interface Health {
  state: "warming" | "ready" | "error";
  warm_seconds: number | null;
  local_embeddings: boolean;
  core?: { dense_index: boolean; model: string; documents: number };
  models: ModelLamp[];
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

async function errorText(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data.detail === "string") return data.detail;
  } catch {
    /* тело не JSON */
  }
  return `Ошибка сервера ${res.status}`;
}

export const fetchHealth = () => getJson<Health>("/api/health");
export const fetchExamples = () => getJson<Example[]>("/api/examples");
export const fetchDocuments = () => getJson<DocInfo[]>("/api/documents");
export const fetchDocument = (id: string) => getJson<DocFull>(`/api/documents/${encodeURIComponent(id)}`);

// POST /api/chat отвечает потоком SSE: несколько событий stage, затем result.
// EventSource умеет только GET, поэтому читаем поток вручную через fetch.
export async function streamChat(messages: Message[], onStage: (s: Stage) => void): Promise<Result> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });
  if (!res.ok || !res.body) throw new Error(await errorText(res));

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // События разделены пустой строкой: "event: …\ndata: …\n\n"
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const event = /^event: (.*)$/m.exec(raw)?.[1];
      const data = /^data: (.*)$/m.exec(raw)?.[1];
      if (!event || data === undefined) continue;
      const parsed = JSON.parse(data);
      if (event === "stage") onStage(parsed as Stage);
      if (event === "result") return parsed as Result;
    }
  }
  throw new Error("Соединение прервалось до ответа");
}
