"""Локальная заглушка OpenAI-совместимого прокси: /chat/completions, /embeddings, /rerank (только stdlib).

    python tools/stub_proxy.py --port 8765
    LLM_BASE_URL=http://127.0.0.1:8765/v1 LLM_API_KEY=stub python solution.py

Ответы детерминированы: эмбеддинги — псевдослучайные векторы из хеша текста, реранкер — порядок по хешу,
чат — ответ из строк самого запроса (числа в нём подтверждаются контекстом). Нужна для проверки сабмита и
веб-части без ключей и сети; качество ответов не оценивает.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fake_vector(text: str, dim: int = 64) -> list[float]:
    raw, i = b"", 0
    while len(raw) < dim * 4:
        raw += hashlib.sha256(f"{i}:{text}".encode("utf-8")).digest()
        i += 1
    v = [x / 2 ** 32 - 0.5 for x in struct.unpack(f"<{dim}I", raw[: dim * 4])]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def fake_answer(messages: list[dict]) -> str:
    h = _sha(json.dumps(messages, ensure_ascii=False, sort_keys=True))
    n = int(h[:8], 16)
    user = messages[-1].get("content") or ""
    if user.startswith("Ответь одним словом"):
        return "Готов."
    lines = [ln.strip() for ln in user.split("\n")
             if 40 <= len(ln.strip()) <= 260 and not ln.strip().startswith(("===", "[", "<"))]
    if not lines:
        return "В приложенных документах сведений по вопросу нет."
    return f"[заглушка] {lines[n % len(lines)]} {lines[(n // 7) % len(lines)]}"


class Handler(BaseHTTPRequestHandler):
    calls = {"chat": 0, "embeddings": 0, "rerank": 0}
    lock = threading.Lock()

    def log_message(self, fmt, *args):  # тихо
        pass

    def _send(self, code: int, body: dict) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:
        if not self.headers.get("Authorization", "").startswith("Bearer "):
            return self._send(401, {"error": "no key"})
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        path = self.path.rstrip("/")
        if path.endswith("/chat/completions"):
            with self.lock:
                self.calls["chat"] += 1
            return self._send(200, {"choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": fake_answer(payload.get("messages", []))}}]})
        if path.endswith("/embeddings"):
            with self.lock:
                self.calls["embeddings"] += 1
            texts = payload.get("input") or []
            texts = [texts] if isinstance(texts, str) else texts
            return self._send(200, {"data": [{"index": i, "embedding": fake_vector(t)} for i, t in enumerate(texts)]})
        if path.endswith("/rerank"):
            with self.lock:
                self.calls["rerank"] += 1
            q, docs = payload.get("query", ""), payload.get("documents") or []
            scored = sorted(((i, int(_sha(q + "\x00" + d)[:8], 16) / 2 ** 32) for i, d in enumerate(docs)),
                            key=lambda x: -x[1])[: payload.get("top_n") or len(docs)]
            return self._send(200, {"results": [{"index": i, "relevance_score": s} for i, s in scored]})
        return self._send(404, {"error": f"unknown path {self.path}"})


def start(port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    """Запуск в фоне; возвращает (сервер, базовый URL с /v1)."""
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Заглушка прокси: http://127.0.0.1:{args.port}/v1 (Ctrl+C — выход)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
