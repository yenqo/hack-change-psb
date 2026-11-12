"""Веб-бэкенд юридического ассистента: FastAPI поверх ядра (assistant.LegalAssistant, контракт 1.0).

Запуск из корня репозитория:
    .venv/bin/uvicorn web.server:app --port 8000

Переменные окружения (читаются также из .env в корне репозитория, окружение важнее):
    LLM_BASE_URL, LLM_API_KEY, LLM_MODEL  — провайдер генерации (DeepSeek V4.1 Flash)
    EMB_BASE_URL     — необязательно: локальный OpenAI-совместимый сервер эмбеддингов bge-m3
                       (например, llama-server на http://127.0.0.1:8081/v1). Без него поиск по законам — BM25.
    EMB_LOCAL_MODEL  — имя модели для этого сервера (по умолчанию bge-m3)
    RERANK_BASE_URL  — необязательно: локальный сервер реранкера с /rerank
    WEB_PORT         — порт, на котором запущен этот сервер (по умолчанию 8000), нужен для маршрутизации
    EMB_WAIT_SECONDS — сколько ждать готовности сервера эмбеддингов при старте, с (по умолчанию 300)
    WEB_ANSWER_TIMEOUT — предельное время одного ответа, с (по умолчанию 150: с переформулированием R7
                       и медленным провайдером диалоговый ответ бывает дольше минуты)

Ядро ходит во все модели по одному адресу LLM_BASE_URL. Если задан EMB_BASE_URL или RERANK_BASE_URL, ядру
выдаётся внутренний адрес этого сервера (/internal/llm/v1), а сервер раскладывает запросы: генерация —
к провайдеру, эмбеддинги и реранкер — к локальным серверам.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import re
import secrets
import socket
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
FRONT_DIST = Path(__file__).resolve().parent / "frontend" / "dist"
sys.path.insert(0, str(ROOT))  # модули ядра лежат плоско в корне репозитория


def load_dotenv(path: Path) -> None:
    """KEY=VALUE из .env; уже заданные переменные окружения не перезаписываются."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if v and not os.environ.get(k):
            os.environ[k] = v


load_dotenv(ROOT / ".env")


class _HideInternalRoutes(logging.Filter):
    """Не пишем в лог доступа внутренние запросы ядра к маршрутизации моделей (/internal/llm/...):
    их десятки на прогрев, это внутренняя кухня. Режим работы описывают строки «[web] …»."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "/internal/llm/" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_HideInternalRoutes())

UPSTREAM_BASE = (os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
UPSTREAM_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
# Пустое значение (например, скопированное из .env.example) означает «по умолчанию»
EMB_BASE = os.environ.get("EMB_BASE_URL", "").rstrip("/")
EMB_LOCAL_MODEL = os.environ.get("EMB_LOCAL_MODEL") or "bge-m3"
# Сколько ждать сервер эмбеддингов при старте (в Docker он сначала скачивает модель ~0,6 ГБ)
EMB_WAIT_SECONDS = float(os.environ.get("EMB_WAIT_SECONDS") or 300)
RERANK_BASE = os.environ.get("RERANK_BASE_URL", "").rstrip("/")
WEB_PORT = int(os.environ.get("WEB_PORT") or 8000)
ANSWER_TIMEOUT = float(os.environ.get("WEB_ANSWER_TIMEOUT") or 150)
USE_ROUTER = bool(EMB_BASE or RERANK_BASE)
ROUTER_TOKEN = secrets.token_urlsafe(24)  # ядро получает этот токен вместо настоящего ключа

TURN_RE = re.compile(r"^\s*(Пользователь|Ассистент)\s*:\s*", re.M)


# ---------------------------------------------------------------------------- схемы

class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=30)


class Example(BaseModel):
    id: str
    title: str
    messages: list[Message]


# ---------------------------------------------------------------------------- ядро

class Core:
    """Один LegalAssistant на процесс + состояние прогрева."""

    def __init__(self) -> None:
        overrides = {}
        if USE_ROUTER:
            base = f"http://127.0.0.1:{WEB_PORT}/internal/llm/v1"
            overrides = {"LLM_BASE_URL": base, "LLM_API_KEY": ROUTER_TOKEN,
                         "EMBEDDER_API_KEY": ROUTER_TOKEN, "RERANKER_API_KEY": ROUTER_TOKEN}
        if not os.environ.get("DATA_DIR") and (ROOT / "data").is_dir():
            overrides["DATA_DIR"] = str(ROOT / "data")
        from assistant import LegalAssistant
        self.bot = LegalAssistant(overrides=overrides, answer_timeout=ANSWER_TIMEOUT)
        self.state = "warming"  # warming | ready | error
        self.warm_seconds: float | None = None

    def warmup(self) -> None:
        t0 = time.monotonic()
        if USE_ROUTER:
            wait_port(WEB_PORT)  # эмбеддинги базы идут через этот же сервер — ждём, пока он начнёт слушать
        if EMB_BASE:
            # Ядро проверяет эмбеддинги один раз, при прогреве. Если сервер эмбеддингов ещё загружает модель,
            # ядро до перезапуска работало бы только на BM25 — поэтому сначала дожидаемся его готовности.
            global emb_available
            # Если эмбеддингов нет, маршрутизация сразу отвечает ядру 404 «модели нет» — ядро без повторов
            # переходит на BM25 (на 502 оно бы долго повторяло запросы).
            emb_available = wait_embeddings(EMB_BASE, EMB_WAIT_SECONDS)
        health = self.bot.warmup()
        self.warm_seconds = round(time.monotonic() - t0, 1)
        self.state = "ready" if health.get("kb_loaded") else "error"
        print(f"[web] ядро: {self.state} за {self.warm_seconds} с; документов {health.get('documents')}, "
              f"dense_index={health.get('dense_index')}, ошибка={health.get('kb_error')}", file=sys.stderr, flush=True)


def wait_embeddings(base: str, timeout: float) -> bool:
    """Ждёт, пока сервер эмбеддингов (llama-server) ответит 200 на /health. Если его хост вообще не существует
    (контейнер эмбеддингов не запущен — лёгкий режим), не ждёт: ядро будет работать на BM25."""
    url = base.removesuffix("/v1") + "/health"
    host = httpx.URL(url).host
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            socket.getaddrinfo(host, None)
        except socket.gaierror:
            print(f"[web] сервер эмбеддингов {host} не найден — поиск только по BM25", file=sys.stderr, flush=True)
            return False
        try:
            if httpx.get(url, timeout=3.0).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(2)
    print(f"[web] сервер эмбеддингов не готов за {timeout:.0f} с — поиск только по BM25", file=sys.stderr, flush=True)
    return False


def wait_port(port: int, timeout: float = 30.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.2)


core: Core | None = None
emb_available = True  # сервер эмбеддингов отвечал при старте (иначе — BM25)
http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global core, http_client
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
    core = Core()
    # Прогрев (база, BM25, эмбеддинги) — в фоне: сервер и фронт доступны сразу.
    threading.Thread(target=core.warmup, name="core-warmup", daemon=True).start()
    yield
    await http_client.aclose()


app = FastAPI(title="Юридический ассистент", lifespan=lifespan)


def require_ready() -> Core:
    if core is None or core.state == "warming":
        raise HTTPException(503, "Ассистент загружает базу знаний, попробуйте через несколько секунд")
    if core.state == "error":
        raise HTTPException(503, "Ядро ассистента не загрузилось, см. /api/health")
    return core


# ---------------------------------------------------------------------------- API

_probe_cache: dict[str, tuple[float, bool, str]] = {}


async def probe(name: str, url: str, key: str = "", ttl: float = 30.0) -> tuple[bool, str]:
    """Дешёвая проверка доступности сервиса (GET без вызова модели), с кэшем на ttl секунд."""
    now = time.monotonic()
    hit = _probe_cache.get(name)
    if hit and now - hit[0] < ttl:
        return hit[1], hit[2]
    assert http_client is not None
    try:
        r = await http_client.get(url, headers={"Authorization": f"Bearer {key}"} if key else {}, timeout=5.0)
        ok, detail = r.status_code < 400, f"HTTP {r.status_code}"
    except httpx.HTTPError as exc:
        ok, detail = False, type(exc).__name__
    _probe_cache[name] = (now, ok, detail)
    return ok, detail


def lamp(key: str, name: str, state: str, detail: str) -> dict:
    return {"key": key, "name": name, "state": state, "detail": detail}


async def model_lamps(core_health: dict | None) -> list[dict]:
    """Состояние моделей для лампочек в интерфейсе: green — подключено и работает; yellow — загружается
    или работает в запасном режиме (эмбеддингов нет — поиск по BM25); red — не подключено или недоступно."""
    if core_health is None:
        msg = "Ассистент загружает базу знаний"
        return [lamp("llm", "LLM", "yellow", msg), lamp("emb", "Emb", "yellow", msg),
                lamp("rerank", "Reranker", "yellow", msg)]
    model = core_health.get("model") or "модель"
    # Генерация
    if not core_health.get("llm_configured"):
        llm = lamp("llm", "LLM", "red", "Не заданы адрес или ключ провайдера (LLM_BASE_URL, LLM_API_KEY)")
    elif core_health.get("llm_last_ok") is False:
        llm = lamp("llm", "LLM", "red", f"{model}: последний запрос к модели завершился ошибкой")
    else:
        ok, detail = await probe("llm", UPSTREAM_BASE + "/models", UPSTREAM_KEY)
        llm = (lamp("llm", "LLM", "green", f"{model}: провайдер доступен") if ok
               else lamp("llm", "LLM", "red", f"{model}: провайдер не отвечает ({detail})"))
    # Эмбеддинги
    dense = core_health.get("dense_index")
    # Без эмбеддингов ассистент работает на BM25 — это рабочий режим, поэтому жёлтый, а не красный.
    if EMB_BASE and emb_available:
        ok, detail = await probe("emb", EMB_BASE.removesuffix("/v1") + "/health")
        if not ok:
            emb = lamp("emb", "Emb", "yellow", f"Сервер эмбеддингов не отвечает ({detail}); поиск только по BM25")
        elif dense:
            emb = lamp("emb", "Emb", "green", "bge-m3 работает: гибридный поиск BM25 + эмбеддинги")
        else:
            emb = lamp("emb", "Emb", "yellow", "Сервер эмбеддингов доступен, но индекс не построен; поиск только по BM25")
    else:
        emb = (lamp("emb", "Emb", "green", "Эмбеддинги через провайдера: гибридный поиск") if dense
               else lamp("emb", "Emb", "yellow", "Эмбеддинги не подключены: поиск только по BM25"))
    # Реранкер — необязательный компонент: без него ассистент отвечает, но найденное не переупорядочивается
    if USE_ROUTER and not RERANK_BASE:
        rr = lamp("rerank", "Reranker", "red",
                  "Реранкер не подключён (необязательный компонент): ответы без переупорядочивания найденного")
    elif core_health.get("degraded", {}).get("rerank_off"):
        rr = lamp("rerank", "Reranker", "red", "Реранкер отключён после ошибок провайдера; ответы без него")
    else:
        rr = lamp("rerank", "Reranker", "green", "Реранкер подключён")
    return [llm, emb, rr]


@app.get("/api/health")
async def health() -> dict:
    data = {"state": core.state if core else "warming", "warm_seconds": core.warm_seconds if core else None,
            "local_embeddings": bool(EMB_BASE), "local_rerank": bool(RERANK_BASE)}
    core_health = None
    if core and core.state != "warming":
        core_health = await run_in_threadpool(core.bot.health)
        data["core"] = core_health
    data["models"] = await model_lamps(core_health)
    return data


@app.get("/api/documents")
async def documents() -> list[dict]:
    c = require_ready()
    return await run_in_threadpool(c.bot.list_documents)


@app.get("/api/documents/{doc_id}")
async def document(doc_id: str) -> dict:
    c = require_ready()
    doc = await run_in_threadpool(c.bot.get_document, doc_id)
    if doc is None:
        raise HTTPException(404, "Документ не найден")
    return doc


@app.get("/api/examples")
def examples() -> list[Example]:
    return EXAMPLES


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """SSE: события stage (search → generate → verify → done), затем result — Result.to_dict() по контракту."""
    c = require_ready()
    if req.messages[-1].role != "user":
        raise HTTPException(422, "Последняя реплика должна быть от пользователя")
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    messages = [m.model_dump() for m in req.messages]

    def on_stage(stage: str) -> None:
        # Вызывается из рабочего потока ядра: передаём событие в event loop потокобезопасно.
        loop.call_soon_threadsafe(queue.put_nowait, ("stage", stage))

    async def run() -> None:
        try:
            res = await run_in_threadpool(c.bot.answer, messages, on_stage)
            payload = res.to_dict()
        except Exception as exc:  # фасад не бросает исключений, но на всякий случай
            payload = {"status": "error", "text": "Не удалось получить ответ", "citations": [],
                       "warnings": [{"code": "internal_error", "message": type(exc).__name__}], "timing_ms": {}}
        await queue.put(("result", payload))

    async def events():
        task = asyncio.create_task(run())
        try:
            while True:
                try:
                    kind, data = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    # Пока модель думает, событий нет до минуты и дольше. Комментарий SSE раз в 15 с
                    # держит соединение живым (браузеры и прокси закрывают «молчащие» соединения).
                    yield ": ping\n\n"
                    continue
                yield sse(kind, data)
                if kind == "result":
                    break
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------- маршрутизация моделей для ядра

def check_token(request: Request) -> None:
    if request.headers.get("authorization", "") != f"Bearer {ROUTER_TOKEN}":
        raise HTTPException(401, "unauthorized")


async def forward(url: str, key: str, payload: dict) -> JSONResponse:
    assert http_client is not None
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        r = await http_client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        return JSONResponse({"error": {"message": f"upstream: {type(exc).__name__}"}}, status_code=502)
    try:
        data = r.json()
    except ValueError:
        data = {"error": {"message": "upstream: bad json"}}
    extra = {k: v for k, v in r.headers.items() if k.lower() == "retry-after"}
    return JSONResponse(data, status_code=r.status_code, headers=extra)


if USE_ROUTER:
    @app.post("/internal/llm/v1/chat/completions", include_in_schema=False)
    async def r_chat(request: Request) -> JSONResponse:
        check_token(request)
        return await forward(UPSTREAM_BASE + "/chat/completions", UPSTREAM_KEY, await request.json())

    @app.post("/internal/llm/v1/embeddings", include_in_schema=False)
    async def r_emb(request: Request) -> JSONResponse:
        check_token(request)
        if not EMB_BASE or not emb_available:
            return JSONResponse({"error": {"message": "no local embeddings"}}, status_code=404)
        payload = await request.json()
        payload["model"] = EMB_LOCAL_MODEL
        return await forward(EMB_BASE + "/embeddings", "", payload)

    @app.post("/internal/llm/v1/rerank", include_in_schema=False)
    async def r_rerank(request: Request) -> JSONResponse:
        check_token(request)
        if not RERANK_BASE:
            return JSONResponse({"error": {"message": "no local reranker"}}, status_code=404)
        return await forward(RERANK_BASE + "/rerank", "", await request.json())


# ---------------------------------------------------------------------------- примеры из демо-набора

def split_dialog(text: str) -> list[Message]:
    """Строка «Пользователь: … / Ассистент: …» → список сообщений."""
    marks = list(TURN_RE.finditer(text))
    if not marks:
        return [Message(role="user", content=text.strip())]
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        role = "user" if m.group(1) == "Пользователь" else "assistant"
        out.append(Message(role=role, content=text[m.end():end].strip()))
    return out


def load_examples() -> list[Example]:
    path = ROOT / "data" / "demo_questions.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        msgs = split_dialog(row["question"])
        out.append(Example(id=row["id"], title=msgs[-1].content, messages=msgs))
    return out


EXAMPLES = load_examples()


# ---------------------------------------------------------------------------- фронт

if FRONT_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONT_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        # index.html не кэшируем: после пересборки фронта браузер сразу берёт новую версию
        # (сами JS/CSS в /assets имеют хеш в имени и кэшируются безопасно).
        return FileResponse(FRONT_DIST / "index.html", headers={"Cache-Control": "no-cache"})
