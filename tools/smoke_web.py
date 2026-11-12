"""Smoke-проверка веб-интерфейса (живого Docker-контейнера или in-process через TestClient).

Примеры:
    python tools/smoke_web.py --inprocess
    python tools/smoke_web.py --url http://127.0.0.1:8000 --wait 180
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _parse_sse(raw: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    event_name = ""
    data_lines: list[str] = []
    for line in raw.splitlines():
        if line.startswith(":"):
            continue
        if not line.strip():
            if event_name and data_lines:
                events.append((event_name, json.loads("\n".join(data_lines))))
            event_name, data_lines = "", []
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())
    if event_name and data_lines:
        events.append((event_name, json.loads("\n".join(data_lines))))
    return events


def _check_contract(events: list[tuple[str, dict]]) -> None:
    results = [payload for kind, payload in events if kind == "result"]
    assert len(results) == 1, f"Ожидалось 1 событие result, получено: {events}"
    res = results[0]
    for key in ("contract_version", "status", "text", "citations", "warnings", "context", "timing_ms"):
        assert key in res, f"В result нет поля {key}: {res.keys()}"
    assert res["contract_version"] == "1.1", f"Версия контракта {res['contract_version']} != 1.1"
    assert res["text"].strip(), "Пустой текст ответа"


def run_http(base_url: str, wait_sec: float, auth: str | None, check_chat: bool) -> None:
    base = base_url.rstrip("/")
    headers = {}
    if auth:
        token = base64.b64encode(auth.encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {token}"

    def req(path: str, data: bytes | None = None, timeout: float = 15.0) -> tuple[int, str]:
        h = dict(headers)
        if data is not None:
            h["Content-Type"] = "application/json"
        r = urllib.request.Request(base + path, data=data, headers=h, method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", errors="replace")

    deadline = time.monotonic() + wait_sec
    while True:
        try:
            code, body = req("/api/ready", timeout=5.0)
            if code == 200:
                break
        except urllib.error.URLError as exc:
            code, body = 0, str(exc)
        if time.monotonic() >= deadline:
            raise SystemExit(f"Сервис {base} не перешёл в ready за {wait_sec:.0f} с (последний ответ: {code} {body})")
        time.sleep(2.0)

    code, health_raw = req("/api/health")
    assert code == 200, f"/api/health -> {code}"
    health = json.loads(health_raw)
    assert health["state"] == "ready", f"Неожиданный state: {health}"

    code, docs_raw = req("/api/documents")
    assert code == 200, f"/api/documents -> {code}"
    docs = json.loads(docs_raw)
    assert len(docs) == 21, f"Ожидался 21 документ, получено {len(docs)}"

    code, doc_raw = req(f"/api/documents/{docs[0]['doc_id']}")
    assert code == 200 and json.loads(doc_raw).get("text"), "/api/documents/{doc_id} не вернул текст"

    code, ex_raw = req("/api/examples")
    assert code == 200 and len(json.loads(ex_raw)) >= 1, "/api/examples пуст"

    code, html = req("/")
    assert code == 200 and "<html" in html.lower(), "/ не отдал HTML фронтенда"

    if check_chat:
        payload = json.dumps({"messages": [{"role": "user", "content": "Какая арендная плата?"}]}).encode("utf-8")
        code, sse_raw = req("/api/chat", data=payload, timeout=160.0)
        assert code == 200, f"/api/chat -> {code}: {sse_raw}"
        _check_contract(_parse_sse(sse_raw))

    print(f"OK ({base}): ready за {health.get('warm_seconds')} с, документов={len(docs)}, моделей={len(health.get('models', []))}")


def run_inprocess() -> None:
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient
    import web.server as srv

    # Без реального провайдера и локального .env: проверяем прогрев базы, роуты, статику и SSE-контракт.
    for key in ("LLM_BASE_URL", "BASE_URL", "OPENAI_BASE_URL", "LLM_API_KEY", "OPENAI_API_KEY",
                "EMBEDDER_API_KEY", "RERANKER_API_KEY", "EMB_BASE_URL", "RERANK_BASE_URL"):
        os.environ.pop(key, None)
    os.environ["DATA_DIR"] = str(ROOT / "data")
    srv.UPSTREAM_BASE = srv.UPSTREAM_KEY = srv.EMB_BASE = srv.RERANK_BASE = ""
    srv.USE_ROUTER = False

    with TestClient(srv.app) as client:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            r = client.get("/api/ready")
            if r.status_code == 200:
                break
            time.sleep(0.1)
        assert r.status_code == 200, f"/api/ready не стал 200: {r.status_code} {r.text}"

        health = client.get("/api/health").json()
        assert health["state"] == "ready", health

        docs = client.get("/api/documents").json()
        assert len(docs) == 21, len(docs)

        doc = client.get(f"/api/documents/{docs[0]['doc_id']}").json()
        assert doc.get("text"), "Нет текста документа"

        examples = client.get("/api/examples").json()
        assert len(examples) == 18, len(examples)

        if (ROOT / "web" / "frontend" / "dist" / "index.html").is_file():
            root_resp = client.get("/")
            assert root_resp.status_code == 200 and "<html" in root_resp.text.lower()

        chat_resp = client.post("/api/chat", json={"messages": [{"role": "user", "content": "Какая арендная плата?"}]})
        assert chat_resp.status_code == 200, chat_resp.text
        _check_contract(_parse_sse(chat_resp.text))

        print(f"OK (inprocess): ready за {health.get('warm_seconds')} с, документов={len(docs)}, примеров={len(examples)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inprocess", action="store_true", help="Проверка через FastAPI TestClient без Docker")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="URL запущенного контейнера")
    ap.add_argument("--wait", type=float, default=120.0, help="Сколько секунд ждать /api/ready")
    ap.add_argument("--auth", default=None, help="Basic Auth в формате user:password")
    ap.add_argument("--chat", action="store_true", help="Проверить также POST /api/chat (тратит запрос к LLM)")
    args = ap.parse_args()
    if args.inprocess:
        run_inprocess()
    else:
        run_http(args.url, args.wait, args.auth, args.chat)


if __name__ == "__main__":
    main()
