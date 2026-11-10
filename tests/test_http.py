"""HTTP-слой v1.3: повторы, 429, 4xx, разбор ответов chat/embeddings/rerank (httpx.MockTransport, без сети)."""

from __future__ import annotations

import contextlib
import json
import unittest
from unittest import mock

import httpx

import harness


class HttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.core = harness.load_core()

    @contextlib.contextmanager
    def proxy(self, responses):
        """responses: список (status, body | Exception, headers) по очереди запросов."""
        seen: list[httpx.Request] = []
        queue = list(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            status, body, headers = queue.pop(0)
            if isinstance(body, Exception):
                raise body
            content = body if isinstance(body, bytes) else json.dumps(body).encode()
            return httpx.Response(status, content=content, headers=headers or {})

        real_client = httpx.Client

        def client(*args, **kwargs):
            return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)

        flags = self.core.FLAGS
        saved = {k: v for k, v in vars(flags).items() if k != "lock"}
        saved["stats"] = dict(flags.stats)
        patched = {"API_BASE": "http://proxy.test/v1", "LLM_KEY": "k-llm", "EMBED_KEY": "k-emb", "RERANK_KEY": "k-rr"}
        old = {k: getattr(self.core, k) for k in patched}
        for k, v in patched.items():
            harness.patch_everywhere(self.core, k, v)
        try:
            with mock.patch.object(httpx, "Client", client), mock.patch("time.sleep"):
                yield seen
        finally:
            for k, v in old.items():
                harness.patch_everywhere(self.core, k, v)
            for k, v in saved.items():
                setattr(flags, k, v)

    @staticmethod
    def chat_body(content, finish="stop"):
        return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}

    # --- post_json ----------------------------------------------------------
    def test_ok_request_shape(self) -> None:
        with self.proxy([(200, {"ok": 1}, None)]) as seen:
            self.assertEqual(self.core.post_json("/x", "k1", {"a": 1}), {"ok": 1})
        self.assertEqual(str(seen[0].url), "http://proxy.test/v1/x")
        self.assertEqual(seen[0].headers["authorization"], "Bearer k1")
        self.assertEqual(json.loads(seen[0].content), {"a": 1})

    def test_4xx_not_retried(self) -> None:
        with self.proxy([(400, {"error": "bad"}, None)]) as seen:
            with self.assertRaises(self.core.APIError) as cm:
                self.core.post_json("/x", "k", {})
        self.assertFalse(cm.exception.transient)
        self.assertEqual(len(seen), 1)

    def test_5xx_retried_then_ok(self) -> None:
        with self.proxy([(503, {}, None), (502, {}, None), (200, {"ok": 2}, None)]) as seen:
            self.assertEqual(self.core.post_json("/x", "k", {}, attempts=3), {"ok": 2})
        self.assertEqual(len(seen), 3)

    def test_5xx_exhausted(self) -> None:
        with self.proxy([(503, {}, None)] * 3) as seen:
            with self.assertRaises(self.core.APIError) as cm:
                self.core.post_json("/x", "k", {}, attempts=3)
        self.assertTrue(cm.exception.transient)
        self.assertEqual(len(seen), 3)

    def test_429_retry_after(self) -> None:
        with self.proxy([(429, {}, {"retry-after": "0"}), (200, {"ok": 3}, None)]) as seen:
            self.assertEqual(self.core.post_json("/x", "k", {}, attempts=1), {"ok": 3})
        self.assertEqual(len(seen), 2)

    def test_transport_error(self) -> None:
        err = httpx.ConnectError("refused")
        with self.proxy([(0, err, None), (0, err, None)]):
            with self.assertRaises(self.core.APIError) as cm:
                self.core.post_json("/x", "k", {}, attempts=2)
        self.assertTrue(cm.exception.transient)
        self.assertIn("ConnectError", str(cm.exception))

    def test_bad_json_retried(self) -> None:
        with self.proxy([(200, b"not json", None), (200, {"ok": 4}, None)]):
            self.assertEqual(self.core.post_json("/x", "k", {}, attempts=2), {"ok": 4})

    def test_no_key(self) -> None:
        with self.proxy([]):
            with self.assertRaises(self.core.APIError):
                self.core.post_json("/x", "", {})

    # --- chat ---------------------------------------------------------------
    def test_chat_payload_and_think(self) -> None:
        with self.proxy([(200, self.chat_body("<think>x</think> Ответ."), None)]) as seen:
            out = self.core._chat([{"role": "user", "content": "q"}], 256, 30.0)
        self.assertEqual(out, "Ответ.")
        body = json.loads(seen[0].content)
        self.assertEqual(sorted(body), ["max_tokens", "messages", "model", "temperature"])
        self.assertEqual(body["max_tokens"], 256)
        self.assertEqual(seen[0].headers["authorization"], "Bearer k-llm")

    def test_chat_length_cut(self) -> None:
        text = "Первое предложение готово. Второе тоже готово. Третье оборва"
        with self.proxy([(200, self.chat_body(text, "length"), None)]):
            out = self.core._chat([{"role": "user", "content": "q"}], 256, 30.0)
        self.assertEqual(out, "Первое предложение готово. Второе тоже готово.")

    def test_chat_empty_length_doubles_tokens(self) -> None:
        with self.proxy([(200, self.chat_body("", "length"), None), (200, self.chat_body("Ок."), None)]) as seen:
            out = self.core._chat([{"role": "user", "content": "q"}], 1024, 30.0)
        self.assertEqual(out, "Ок.")
        self.assertEqual([json.loads(r.content)["max_tokens"] for r in seen], [1024, 2048])

    def test_chat_content_parts(self) -> None:
        body = {"choices": [{"message": {"content": [{"text": "Часть 1. "}, {"text": "Часть 2."}]}}]}
        with self.proxy([(200, body, None)]):
            self.assertEqual(self.core._chat([{"role": "user", "content": "q"}], 256, 30.0), "Часть 1. Часть 2.")

    def test_chat_empty(self) -> None:
        with self.proxy([(200, self.chat_body(""), None)]):
            with self.assertRaises(self.core.APIError):
                self.core._chat([{"role": "user", "content": "q"}], 256, 30.0)

    # --- embeddings и rerank --------------------------------------------------
    def test_embed_order_and_norm(self) -> None:
        body = {"data": [{"index": 1, "embedding": [0, 2]}, {"index": 0, "embedding": [3, 4]}]}
        with self.proxy([(200, body, None)]):
            out = self.core.embed(["a", "b"])
        self.assertEqual(out, [[0.6, 0.8], [0.0, 1.0]])

    def test_embed_wrong_count(self) -> None:
        with self.proxy([(200, {"data": [{"index": 0, "embedding": [1]}]}, None)]):
            with self.assertRaises(self.core.APIError):
                self.core.embed(["a", "b"])

    def test_rerank_parse(self) -> None:
        body = {"results": [{"index": 2, "relevance_score": 0.9}, {"index": 0, "score": 0.5}, {"index": 9}]}
        with self.proxy([(200, body, None)]) as seen:
            out = self.core.rerank("q", ["a", "b", "c"], 5)
        self.assertEqual(out, [(2, 0.9), (0, 0.5)])
        self.assertEqual(json.loads(seen[0].content)["top_n"], 3)

    def test_rerank_off_after_permanent_errors(self) -> None:
        with self.proxy([(404, {}, None), (404, {}, None)]):
            self.assertIsNone(self.core.rerank("q", ["a"], 1))
            self.assertFalse(self.core.FLAGS.rerank_off)
            self.assertIsNone(self.core.rerank("q", ["a"], 1))
            self.assertTrue(self.core.FLAGS.rerank_off)


if __name__ == "__main__":
    unittest.main()
