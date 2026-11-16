"""Веб для жюри: общий API трёх моделей и приоритет явного окружения над .env."""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import harness


@unittest.skipIf(harness.target() == "baseline", "для веб-версии текущего ядра")
class WebProxyTest(unittest.TestCase):
    def test_explicit_empty_env_disables_old_local_addresses(self) -> None:
        # Загружаем только функцию: тест не импортирует сервер и не читает рабочий .env.
        tree = ast.parse((harness.ROOT / "web/server.py").read_text())
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_dotenv")
        ns = {"os": os, "Path": Path}
        exec(compile(ast.Module(body=[func], type_ignores=[]), "load_dotenv", "exec"), ns)
        with tempfile.TemporaryDirectory() as work, patch.dict(os.environ, {
            "EMB_BASE_URL": "", "RERANK_BASE_URL": "", "LLM_API_KEY": "environment-key",
        }, clear=True):
            path = Path(work) / ".env"
            path.write_text("EMB_BASE_URL=http://old-emb/v1\nRERANK_BASE_URL=http://old-rerank/v1\n"
                            "LLM_API_KEY=file-key\nLLM_MODEL=jury-generation\n")
            ns["load_dotenv"](path)
            self.assertEqual(os.environ["EMB_BASE_URL"], "")
            self.assertEqual(os.environ["RERANK_BASE_URL"], "")
            self.assertEqual(os.environ["LLM_API_KEY"], "environment-key")
            self.assertEqual(os.environ["LLM_MODEL"], "jury-generation")

    @unittest.skipUnless(importlib.util.find_spec("fastapi"), "нужны web/requirements.txt")
    def test_warmup_and_question_use_all_three_proxy_models(self) -> None:
        code = textwrap.dedent('''
            import json, os, threading, time
            from http.server import ThreadingHTTPServer
            from tools import stub_proxy

            calls = []
            class Handler(stub_proxy.Handler):
                def do_GET(self):
                    self._send(200, {"data": [{"id": "jury-generation"}]})
                def do_POST(self):
                    size = int(self.headers.get("Content-Length", 0))
                    raw = self.rfile.read(size)
                    import io
                    self.rfile = io.BytesIO(raw)
                    calls.append((self.path, self.headers.get("Authorization"), json.loads(raw)["model"]))
                    super().do_POST()

            stub_proxy.fake_answer = lambda messages: (
                "Арендная плата составляет 450 000 рублей в месяц. Источник: "
                "Договор аренды № А-08/2024 с ООО «Северный Склад», п. 3.1.")
            upstream = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            threading.Thread(target=upstream.serve_forever, daemon=True).start()
            base = "http://127.0.0.1:%s/v1" % upstream.server_port
            os.environ.update(LLM_BASE_URL=base, LLM_API_KEY="jury-key", LLM_MODEL="jury-generation",
                              EMB_MODEL="jury-embedding", RERANKER_MODEL="jury-reranker",
                              EMBEDDER_API_KEY="", RERANKER_API_KEY="", EMB_BASE_URL="", RERANK_BASE_URL="",
                              SOLUTION_MODE="web", SOLUTION_PREBUILT_INDEX="0", SOLUTION_REWRITE="0")
            from fastapi.testclient import TestClient
            from web import server
            assert not server.USE_ROUTER
            with TestClient(server.app) as client:
                deadline = time.monotonic() + 20
                while client.get("/api/ready").status_code != 200:
                    assert time.monotonic() < deadline, "ядро не прогрелось"
                    time.sleep(.05)
                assert server.core.bot.health()["dense_index"]
                before = sum(path.endswith("/embeddings") for path, _, _ in calls)
                response = client.post("/api/chat", json={"messages": [
                    {"role": "user", "content": "Какая арендная плата по договору с Северным Складом?"}]})
                assert response.status_code == 200
                result = next(json.loads(block.split("\\ndata: ", 1)[1])
                              for block in response.text.split("\\n\\n") if block.startswith("event: result"))
                assert result["status"] == "answer", result
                assert result["citations"]
                assert sum(path.endswith("/embeddings") for path, _, _ in calls) > before
                expected = {"/v1/chat/completions": "jury-generation", "/v1/embeddings": "jury-embedding",
                            "/v1/rerank": "jury-reranker"}
                assert set(path for path, _, _ in calls) == set(expected)
                assert all(key == "Bearer jury-key" and model == expected[path] for path, key, model in calls)
                lamps = client.get("/api/health").json()["models"]
                assert all(lamp["state"] == "green" for lamp in lamps), lamps
            upstream.shutdown()
            print("shared-proxy-ok")
        ''')
        proc = subprocess.run([sys.executable, "-c", code], cwd=harness.ROOT,
                              env=dict(os.environ, DATA_DIR=str(harness.DATA)),
                              capture_output=True, text=True, timeout=45)
        self.assertEqual(proc.returncode, 0, proc.stderr[-4000:] + proc.stdout[-1000:])
        self.assertIn("shared-proxy-ok", proc.stdout)


if __name__ == "__main__":
    unittest.main()
