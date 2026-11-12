import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  fetchDocuments,
  fetchExamples,
  fetchHealth,
  streamChat,
  type Citation,
  type DocInfo,
  type Example,
  type Health,
} from "./api";
import { loadChats, newChat, saveChats, titleFrom, type Chat, type ChatItem, type Pending } from "./chats";
import { AnswerCard } from "./components/AnswerCard";
import { ChatSidebar } from "./components/ChatSidebar";
import { Composer } from "./components/Composer";
import { DocsPanel, type DocsTab, type OpenDoc } from "./components/DocsPanel";
import { Drawer, useMedia } from "./components/Drawer";
import { clampDocsWidth, loadDocsWidth, ResizeHandle, saveDocsWidth, useElementWidth } from "./components/ResizeHandle";
import { KnowledgeBase } from "./components/KnowledgeBase";
import { TopBar, type Tab } from "./components/TopBar";

const STAGE_TEXT: Record<Pending, string> = {
  queued: "Модель занята ответом в другом чате — вопрос в очереди…",
  search: "Ищу в документах…",
  generate: "Формулирую ответ…",
  verify: "Проверяю числа и ссылки…",
  done: "Готово",
};

export default function App() {
  const [chats, setChats] = useState<Chat[]>(loadChats);
  // null — открыт черновик нового чата: в списке его нет, он появится при первой отправке
  const [activeId, setActiveId] = useState<string | null>(() => loadChats()[0]?.id ?? null);
  const [tab, setTab] = useState<Tab>("chats");
  const [health, setHealth] = useState<Health | null>(null);
  const [docs, setDocs] = useState<DocInfo[]>([]);
  const [examples, setExamples] = useState<Example[]>([]);
  const [pending, setPending] = useState<Record<string, Pending>>({}); // чаты, ждущие ответа, и их этап
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [selected, setSelected] = useState<number | null>(null); // индекс выбранного ответа в активном чате
  const [docsTab, setDocsTab] = useState<DocsTab>("citations");
  const [openDoc, setOpenDoc] = useState<OpenDoc | null>(null);
  // Ширина окна: от 1024px (lg) перечень чатов стоит в оболочке, от 1280px (xl) документы стоят
  // панелью внутри карточки чата. Иначе — шторки.
  const medium = useMedia("(min-width: 1024px)");
  const wide = useMedia("(min-width: 1280px)");
  const [leftOpen, setLeftOpen] = useState(false); // шторка чатов (узкий экран)
  const [rightOpen, setRightOpen] = useState(false); // шторка документов внутри карточки (средний экран)
  const [docsOpen, setDocsOpen] = useState(true); // панель документов развёрнута (широкий экран)
  // Ширина панели документов: тянется мышью, ограничена безопасными пределами (см. ResizeHandle.tsx)
  const bodyRef = useRef<HTMLDivElement>(null);
  const bodyWidth = useElementWidth(bodyRef);
  const [docsWidthRaw, setDocsWidthRaw] = useState(loadDocsWidth);
  const docsWidth = bodyWidth ? clampDocsWidth(docsWidthRaw, bodyWidth) : docsWidthRaw;
  const showDocs = useCallback(() => (wide ? setDocsOpen(true) : setRightOpen(true)), [wide]);
  const closeLeft = useCallback(() => setLeftOpen(false), []);
  const closeRight = useCallback(() => setRightOpen(false), []);

  const chat = activeId ? chats.find((c) => c.id === activeId) : undefined;
  const items = chat?.items ?? [];
  const ready = health?.state === "ready";
  const chatPending = activeId ? pending[activeId] : undefined;
  const docLabels = useMemo(() => Object.fromEntries(docs.map((d) => [d.doc_id, d.label])), [docs]);

  useEffect(() => saveChats(chats), [chats]);

  useEffect(() => {
    fetchExamples().then(setExamples).catch(() => setExamples([]));
  }, []);

  // Состояние ядра и моделей: часто, пока идёт прогрев, потом раз в 20 секунд (для лампочек).
  useEffect(() => {
    const tick = () => fetchHealth().then(setHealth).catch(() => setHealth(null));
    tick();
    const id = setInterval(tick, ready ? 20000 : 2000);
    return () => clearInterval(id);
  }, [ready]);

  useEffect(() => {
    if (ready) fetchDocuments().then(setDocs).catch(() => undefined);
  }, [ready]);

  // Активный чат может смениться, пока ждём ответ; ref всегда хранит актуальный id.
  const activeRef = useRef(activeId);
  activeRef.current = activeId;

  function updateChat(id: string, items: ChatItem[]) {
    setChats((cs) => cs.map((c) => (c.id === id ? { ...c, items, title: titleFrom(items), updatedAt: Date.now() } : c)));
  }

  function setChatPending(id: string, p: Pending | null) {
    setPending((ps) => {
      const next = { ...ps };
      if (p) next[id] = p;
      else delete next[id];
      return next;
    });
  }

  // Очередь вопросов: модель отвечает на один вопрос за раз. Вопрос из другого чата ждёт со статусом «в очереди».
  const queue = useRef<{ id: string; history: ChatItem[] }[]>([]);
  const busy = useRef(false);

  function enqueue(id: string, history: ChatItem[]) {
    setErrors((es) => {
      const { [id]: _drop, ...rest } = es;
      return rest;
    });
    queue.current.push({ id, history });
    setChatPending(id, busy.current ? "queued" : "search");
    void pump();
  }

  async function pump() {
    if (busy.current) return;
    const job = queue.current.shift();
    if (!job) return;
    busy.current = true;
    const { id, history } = job;
    setChatPending(id, "search");
    try {
      const result = await streamChat(
        history.map(({ role, content }) => ({ role, content })),
        (stage) => setChatPending(id, stage),
      );
      const next = [...history, { role: "assistant" as const, content: result.text, result }];
      updateChat(id, next);
      if (activeRef.current === id) {
        setSelected(next.length - 1);
        setDocsTab("citations");
      }
    } catch (e) {
      setErrors((es) => ({ ...es, [id]: e instanceof Error ? e.message : "Не удалось получить ответ" }));
    } finally {
      setChatPending(id, null);
      busy.current = false;
      void pump();
    }
  }

  // Отправка из черновика создаёт чат и только тогда добавляет его в список.
  function startChat(history: ChatItem[]) {
    const c = newChat(history);
    setChats((cs) => [c, ...cs]);
    setActiveId(c.id);
    activeRef.current = c.id;
    setSelected(null);
    setTab("chats");
    enqueue(c.id, history);
  }

  function send(text: string) {
    if (!ready || chatPending) return;
    const msg: ChatItem = { role: "user", content: text };
    if (!chat) return startChat([msg]);
    setTab("chats");
    const history = [...chat.items, msg];
    updateChat(chat.id, history);
    enqueue(chat.id, history);
  }

  function runExample(ex: Example) {
    if (!ready) return;
    startChat(ex.messages); // пример — всегда отдельный новый чат
  }

  function newChatClick() {
    setActiveId(null);
    setSelected(null);
    setTab("chats");
  }

  function selectChat(id: string) {
    setActiveId(id);
    const c = chats.find((x) => x.id === id);
    const last = c ? c.items.map((m, i) => (m.result ? i : -1)).filter((i) => i >= 0).pop() : undefined;
    setSelected(last ?? null);
    setTab("chats");
  }

  function deleteChat(id: string) {
    queue.current = queue.current.filter((j) => j.id !== id); // вопрос удалённого чата из очереди снимаем
    if (pending[id] === "queued") setChatPending(id, null);
    setChats((cs) => cs.filter((c) => c.id !== id));
    if (id === activeId) {
      setActiveId(null);
      setSelected(null);
    }
  }

  function retry(i: number) {
    if (!chat || chatPending) return;
    const history = chat.items.slice(0, i);
    updateChat(chat.id, history);
    enqueue(chat.id, history);
  }

  function openCitation(c: Citation) {
    openDocument({ docId: c.doc_id, highlight: c.found ? { start: c.start, end: c.end } : null, marks: [] });
  }

  function openDocument(d: OpenDoc) {
    setOpenDoc(d);
    setDocsTab("document");
    showDocs();
  }

  const selectedResult = selected !== null ? items[selected]?.result : undefined;
  const error = activeId ? errors[activeId] : undefined;
  const docsVisible = wide ? docsOpen : rightOpen;

  const chatSidebar = (
      <ChatSidebar
        chats={chats}
        activeId={activeId}
        pending={pending}
        onSelect={(id) => {
          selectChat(id);
          setLeftOpen(false);
        }}
        onNew={() => {
          newChatClick();
          setLeftOpen(false);
        }}
        onDelete={deleteChat}
      />
  );
  const docsPanel = (
    <DocsPanel
      tab={docsTab}
      onTab={setDocsTab}
      result={selectedResult}
      open={openDoc}
      onOpen={openDocument}
      docLabels={docLabels}
      onClose={wide ? () => setDocsOpen(false) : closeRight}
    />
  );

  // Вложенная композиция, три слоя по плотности:
  //   оболочка (тёмное стекло, перечень чатов + логотип) →
  //   карточка открытого чата (светлая: шапка, лента, строка ввода) →
  //   панель документов внутри карточки (белая, сворачивается).
  // На узком экране перечень чатов — шторка поверх страницы, документы — шторка внутри карточки.
  return (
    <div className="h-full sm:p-4">
      {/* На телефоне оболочки нет: карточка чата занимает весь экран */}
      <div className="flex h-full overflow-hidden sm:rounded-[36px] sm:border sm:border-white/15 sm:bg-[#0a0f45]/55 sm:shadow-2xl sm:shadow-psb-navy/40 sm:backdrop-blur-2xl">
        {medium && <div className="w-[280px] shrink-0">{chatSidebar}</div>}

        <section className="relative flex min-w-0 flex-1 flex-col overflow-hidden bg-[#f6f6fb] sm:m-2.5 sm:rounded-[28px] sm:shadow-xl sm:shadow-black/20 lg:ml-0">
          <TopBar
            tab={tab}
            onTab={setTab}
            health={health}
            onChats={medium ? undefined : () => setLeftOpen(true)}
            onDocs={() => (docsVisible ? (wide ? setDocsOpen(false) : setRightOpen(false)) : showDocs())}
            docsOpen={docsVisible}
            docsCount={selectedResult?.citations.length}
          />

          <div ref={bodyRef} className="relative flex min-h-0 flex-1">
            <div className="flex min-w-0 flex-1 flex-col">
          {tab === "kb" ? (
            <KnowledgeBase docs={docs} activeId={openDoc?.docId} onOpen={(id) => openDocument({ docId: id, highlight: null, marks: [] })} />
          ) : (
            <ChatView
              items={items}
              stage={chatPending ?? null}
              selected={selected}
              docLabels={docLabels}
              examples={examples}
              ready={ready}
              onSelect={(i) => {
                setSelected(i);
                if (docsTab === "document") setDocsTab("citations");
                showDocs(); // по клику на ответ показываем, на чём он основан
              }}
              onExample={runExample}
              onCitation={openCitation}
              onRetry={retry}
              error={error}
            />
          )}
              <div className="px-3 pt-1 pb-3 sm:px-5 sm:pb-4">
                <Composer
                  disabled={!ready || !!chatPending}
                  placeholder={ready ? "Задайте вопрос по договорам, регламентам или законам…" : "Ассистент загружает базу знаний…"}
                  onSend={send}
                />
              </div>
            </div>

            {wide ? (
              docsOpen && (
                <div className="relative flex shrink-0" style={{ width: docsWidth }}>
                  <ResizeHandle
                    containerRef={bodyRef}
                    width={docsWidth}
                    onWidth={(w) => {
                      const v = clampDocsWidth(w, bodyWidth);
                      setDocsWidthRaw(v);
                      saveDocsWidth(v);
                    }}
                  />
                  {docsPanel}
                </div>
              )
            ) : (
              <Drawer side="right" contained open={rightOpen} onClose={closeRight}>
                {docsPanel}
              </Drawer>
            )}
          </div>
        </section>
      </div>

      {!medium && (
        <Drawer side="left" open={leftOpen} onClose={closeLeft}>
          <div className="h-full overflow-hidden rounded-[28px] border border-white/15 bg-[#0a0f45]/85 shadow-2xl backdrop-blur-2xl">
            {chatSidebar}
          </div>
        </Drawer>
      )}
    </div>
  );
}

// Лента сообщений активного чата.
function ChatView(props: {
  items: ChatItem[];
  stage: Pending | null;
  selected: number | null;
  docLabels: Record<string, string>;
  examples: Example[];
  ready: boolean;
  onSelect: (i: number) => void;
  onExample: (ex: Example) => void;
  onCitation: (c: Citation) => void;
  onRetry: (i: number) => void;
  error?: string;
}) {
  const bottom = useRef<HTMLDivElement>(null);
  // Фигурные скобки важны: эффект не должен ничего возвращать.
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [props.items, props.stage]);

  if (props.items.length === 0 && !props.stage) {
    return (
      <div className="scroll-thin flex flex-1 items-center justify-center overflow-y-auto p-8">
        <div className="max-w-2xl text-center">
          <h1 className="text-2xl font-semibold text-psb-navy">Юридический ассистент</h1>
          <p className="mx-auto mt-2 max-w-lg text-[15px] leading-relaxed text-slate-600">
            Отвечаю только по документам базы знаний, ссылаюсь на пункты и прямо говорю, если данных недостаточно.
          </p>
          <div className="mt-6 grid gap-3 text-left sm:grid-cols-2">
            {props.examples.slice(0, 6).map((ex) => (
              <button
                key={ex.id}
                onClick={() => props.onExample(ex)}
                disabled={!props.ready}
                className="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-left text-sm leading-snug text-slate-700 shadow-sm transition hover:border-psb-sky disabled:opacity-50"
              >
                {ex.messages.length > 1 && (
                  <span className="mr-1.5 rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-medium text-slate-500">диалог</span>
                )}
                {ex.title}
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-3 py-4 sm:px-6 sm:py-6">
      <div className="mx-auto max-w-4xl space-y-5">
        {props.items.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="flex justify-end">
              <div className="max-w-[78%] rounded-3xl rounded-br-md bg-psb-blue px-5 py-3 text-[15px] leading-relaxed whitespace-pre-wrap text-white shadow-md">
                {m.content}
              </div>
            </div>
          ) : m.result ? (
            <div key={i} className="flex">
              <AnswerCard
                result={m.result}
                selected={props.selected === i}
                docLabels={props.docLabels}
                onSelect={() => props.onSelect(i)}
                onCitation={props.onCitation}
                onRetry={() => props.onRetry(i)}
              />
            </div>
          ) : (
            // Реплика ассистента из истории примера (без результата ядра)
            <div key={i} className="flex">
              <div className="max-w-[88%] rounded-3xl rounded-bl-md border border-slate-200 bg-white px-5 py-3 text-[15px] leading-relaxed whitespace-pre-wrap text-slate-700">
                {m.content}
              </div>
            </div>
          ),
        )}
        {/* Последнее сообщение — вопрос без ответа, и ответ не ожидается: связь оборвалась или страницу
            перезагрузили во время ответа. Даём повторить тот же вопрос, не набирая его заново. */}
        {!props.stage && props.items.length > 0 && props.items[props.items.length - 1].role === "user" && (
          <div className="flex">
            <div className="max-w-[88%] rounded-3xl rounded-bl-md border border-red-200 bg-white px-5 py-4 text-[15px] shadow-sm">
              <span className="rounded-full bg-red-50 px-2.5 py-0.5 text-xs font-medium text-red-700 ring-1 ring-red-200">Ответ не получен</span>
              <p className="mt-2.5 leading-relaxed text-slate-700">
                {props.error ?? "Ответ на этот вопрос не пришёл — возможно, прервалось соединение."}
              </p>
              <button
                onClick={() => props.onRetry(props.items.length)}
                disabled={!props.ready}
                className="mt-3 rounded-full bg-psb-blue px-4 py-1.5 text-sm font-medium text-white hover:brightness-110 disabled:opacity-40"
              >
                Повторить
              </button>
            </div>
          </div>
        )}
        {props.stage && (
          <div className="flex">
            <div className="flex items-center gap-2.5 rounded-3xl rounded-bl-md border border-slate-200 bg-white px-5 py-3 text-[15px] text-slate-500">
              <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-psb-orange" />
              {STAGE_TEXT[props.stage]}
            </div>
          </div>
        )}
        <div ref={bottom} />
      </div>
    </div>
  );
}
