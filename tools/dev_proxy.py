"""Локальный прокси для РАЗРАБОТКИ: даёт ядру тот же OpenAI-совместимый интерфейс, что платформа.

    python tools/dev_proxy.py --port 8780          # отдельный процесс
    LLM_BASE_URL=http://127.0.0.1:8780/v1 LLM_API_KEY=dev python solution.py

Маршруты:
  /chat/completions → провайдер из .env (LLM_BASE_URL/LLM_API_KEY); DEV_EXTRA_BODY (JSON) добавляется в тело,
                      usage пишется в журнал токенов (eval/token_ledger.csv); кэш ответов по хешу запроса
                      (включается параметром chat_cache / DEV_CHAT_CACHE=1);
  /embeddings       → локальная bge-m3 в ollama (DEV_EMB_URL, по умолчанию http://127.0.0.1:11434/v1), кэш по тексту;
                      если модели нет — 404, ядро работает на BM25, как при сбое на платформе;
  /rerank           → 404: реранкера в разработке нет, ядро отключает его само.

На платформе прокси не используется: ядро ходит прямо в прокси организаторов. Ключ провайдера ядру не передаётся —
ядро получает фиктивный ключ прокси.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import devenv  # noqa: E402

CACHE_DB = devenv.ROOT / "eval" / "cache" / "dev_cache.sqlite"


class Cache:
    """Кэш в SQLite: таблицы chat, emb, judge. Ключ — sha256 запроса."""

    def __init__(self, path: Path | None = None) -> None:
        path = path or CACHE_DB
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            for t in ("chat", "emb", "judge"):
                self.db.execute(f"CREATE TABLE IF NOT EXISTS {t} (k TEXT PRIMARY KEY, v TEXT)")
            self.db.commit()

    def get(self, table: str, key: str):
        with self.lock:
            row = self.db.execute(f"SELECT v FROM {table} WHERE k=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, table: str, key: str, value) -> None:
        with self.lock:
            self.db.execute(f"INSERT OR REPLACE INTO {table} (k, v) VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False)))
            self.db.commit()


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


class Upstream:
    """Провайдер чата из .env. Ключ хранится только здесь и никуда не выводится."""

    def __init__(self) -> None:
        env = devenv.read_dotenv()
        self.base = (os.environ.get("DEV_UPSTREAM_BASE_URL") or env.get("LLM_BASE_URL", "")).rstrip("/")
        self.key = os.environ.get("DEV_UPSTREAM_API_KEY") or env.get("LLM_API_KEY", "")
        extra = os.environ.get("DEV_EXTRA_BODY") or env.get("DEV_EXTRA_BODY", "")
        self.extra = json.loads(extra) if extra else {}
        self.http = httpx.Client(timeout=httpx.Timeout(300.0, connect=20.0))
        if not self.base or not self.key:
            raise SystemExit("В .env нет LLM_BASE_URL / LLM_API_KEY провайдера")

    def chat(self, payload: dict) -> tuple[int, dict, dict]:
        body = {**payload, **self.extra}
        try:
            r = self.http.post(self.base + "/chat/completions", headers={"Authorization": "Bearer " + self.key}, json=body)
        except httpx.TransportError as exc:
            return 502, {"error": {"message": f"upstream {type(exc).__name__}"}}, {}
        try:
            data = r.json()
        except ValueError:
            data = {"error": {"message": "upstream bad json"}}
        headers = {k: v for k, v in r.headers.items() if k.lower() == "retry-after"}
        return r.status_code, data, headers


class DevProxy:
    def __init__(self, stage: str = "dev", purpose: str = "dev", chat_cache: bool = False, port: int = 0) -> None:
        self.upstream = Upstream()
        self.cache = Cache()
        self.chat_cache = chat_cache
        self.emb_url = (os.environ.get("DEV_EMB_URL") or "http://127.0.0.1:11434/v1").rstrip("/")
        self.emb_model = os.environ.get("DEV_EMB_MODEL") or "bge-m3"
        self.usage = devenv.Usage(stage, purpose)
        self.stats = {"chat": 0, "chat_cached": 0, "chat_err": 0, "emb_texts": 0, "emb_cached": 0, "emb_err": 0,
                      "rerank": 0}
        self.latency: list[float] = []
        self.lock = threading.Lock()
        # журнал вызовов чата: {tag, sec, cached, status, prompt_tokens, completion_tokens}; tag задаёт tagger(payload)
        self.tagger = None
        self.calls: list[dict] = []
        self.errors: dict[str, int] = {}   # «статус сообщение» → сколько раз (без ключей и адресов)
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def _send(self, code: int, body: dict, headers: dict | None = None) -> None:
                raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:
                self._send(200, {"ok": True, "stats": proxy.stats})

            def do_POST(self) -> None:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                path = self.path.rstrip("/")
                if path.endswith("/chat/completions"):
                    self._send(*proxy.handle_chat(payload))
                elif path.endswith("/embeddings"):
                    self._send(*proxy.handle_embeddings(payload))
                elif path.endswith("/rerank"):
                    with proxy.lock:
                        proxy.stats["rerank"] += 1
                    self._send(404, {"error": {"message": "Not found", "type": "not_found_error"}})
                else:
                    self._send(404, {"error": {"message": "Not found"}})

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        self._flusher = None
        self._stop = threading.Event()

    # -- чат -------------------------------------------------------------------
    def handle_chat(self, payload: dict) -> tuple[int, dict, dict]:
        key = sha({k: payload.get(k) for k in ("model", "messages", "temperature", "max_tokens")} | {"x": self.upstream.extra})
        tag = self.tagger(payload) if self.tagger else None
        if self.chat_cache:
            hit = self.cache.get("chat", key)
            if hit is not None:
                with self.lock:
                    self.stats["chat_cached"] += 1
                    self.calls.append({"tag": tag, "sec": 0.0, "cached": True, "status": 200, "prompt_tokens": 0,
                                       "completion_tokens": 0})
                return 200, hit, {}
        t0 = time.monotonic()
        status, data, headers = self.upstream.chat(payload)
        dt = time.monotonic() - t0
        u = data.get("usage") or {} if isinstance(data, dict) else {}
        with self.lock:
            self.stats["chat"] += 1
            if status >= 300:
                self.stats["chat_err"] += 1
                err = data.get("error") if isinstance(data, dict) else None
                msg = (err.get("message") if isinstance(err, dict) else str(err or ""))[:160]
                self.errors[f"{status} {msg}"] = self.errors.get(f"{status} {msg}", 0) + 1
            else:
                self.latency.append(dt)
            self.calls.append({"tag": tag, "sec": round(dt, 3), "cached": False, "status": status,
                               "prompt_tokens": int(u.get("prompt_tokens") or 0),
                               "completion_tokens": int(u.get("completion_tokens") or 0),
                               "cached_tokens": devenv.cached_tokens(u)})
        if status < 300:
            self.usage.add(payload.get("model", "?"), data.get("usage"))
            if self.chat_cache:
                self.cache.put("chat", key, data)
        return status, data, headers

    # -- эмбеддинги ------------------------------------------------------------
    def handle_embeddings(self, payload: dict) -> tuple[int, dict, dict]:
        model = str(payload.get("model", ""))
        if "bge-m3" not in model.lower():
            return 404, {"error": {"message": f"dev: нет модели {model}"}}, {}
        texts = payload.get("input") or []
        texts = [texts] if isinstance(texts, str) else texts
        out: list = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            hit = self.cache.get("emb", sha([self.emb_model, t]))
            if hit is not None:
                out[i] = hit
            else:
                todo.append(i)
        with self.lock:
            self.stats["emb_texts"] += len(texts)
            self.stats["emb_cached"] += len(texts) - len(todo)
        if todo:
            try:
                r = httpx.post(self.emb_url + "/embeddings", json={"model": self.emb_model,
                                                                  "input": [texts[i] for i in todo]}, timeout=120)
            except httpx.TransportError as exc:
                with self.lock:
                    self.stats["emb_err"] += 1
                return 503, {"error": {"message": f"dev embeddings: {type(exc).__name__}"}}, {}
            if r.status_code >= 300:
                with self.lock:
                    self.stats["emb_err"] += 1
                return 404 if r.status_code == 404 else 503, {"error": {"message": "dev embeddings unavailable"}}, {}
            items = sorted(r.json()["data"], key=lambda it: it.get("index", 0))
            for i, it in zip(todo, items):
                out[i] = it["embedding"]
                self.cache.put("emb", sha([self.emb_model, texts[i]]), it["embedding"])
        return 200, {"object": "list", "model": model,
                     "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(out)]}, {}

    # -- жизненный цикл --------------------------------------------------------
    def start(self) -> "DevProxy":
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

        def flusher():
            while not self._stop.wait(30):
                self.usage.flush()
        self._flusher = threading.Thread(target=flusher, daemon=True)
        self._flusher.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self.server.shutdown()
        self.usage.flush()

    def set_purpose(self, stage: str, purpose: str) -> None:
        self.usage.flush()
        self.usage.stage, self.usage.purpose = stage, purpose

    def core_env(self) -> dict[str, str]:
        """Переменные окружения для ядра, направленного в этот прокси (без ключа провайдера)."""
        env = devenv.read_dotenv()
        out = {"LLM_BASE_URL": self.url, "LLM_API_KEY": "dev-proxy"}
        for k in ("LLM_MODEL", "EMB_MODEL", "RERANKER_MODEL", "TEMPERATURE", "MAX_TOKENS", "MAX_WORKERS", "TOP_K",
                  "RERANKER_TOP_N"):
            if env.get(k):
                out[k] = env[k]
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--stage", default="dev")
    ap.add_argument("--purpose", default="ручной прогон")
    ap.add_argument("--chat-cache", action="store_true")
    args = ap.parse_args()
    p = DevProxy(args.stage, args.purpose, args.chat_cache or os.environ.get("DEV_CHAT_CACHE") == "1", args.port).start()
    print(f"dev-прокси: {p.url}  (кэш чата: {'вкл' if p.chat_cache else 'выкл'}; Ctrl+C — выход)")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        p.stop()


if __name__ == "__main__":
    main()
