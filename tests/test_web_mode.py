"""Веб-режим фасада (пакет 3): принудительный режим web, дедлайн запроса, освобождение рабочих потоков."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

import harness


def run_py(code: str, env: dict, timeout: int = 120) -> str:
    out = subprocess.run([sys.executable, "-c", code], cwd=harness.ROOT, capture_output=True, text=True,
                         env=dict(os.environ, **env), timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    return out.stdout.strip().splitlines()[-1]


@unittest.skipIf(harness.target() == "baseline", "фасада нет в v1.3")
class WebModeTest(unittest.TestCase):
    ENV = {**harness.TEST_ENV, "SOLUTION_PREBUILT_INDEX": "0", "SOLUTION_MODE": ""}

    def test_web_mode_forced_after_batch_import(self) -> None:
        code = ("import config, retrieval\n"
                "before = config.MODE\n"
                "from assistant import LegalAssistant\n"
                "LegalAssistant()\n"
                "print(before, config.MODE, retrieval.MODE, config.remaining() > 3000)")
        self.assertEqual(run_py(code, self.ENV), "batch web web True")

    def test_chat_wait_respects_request_deadline(self) -> None:
        code = ("import time, config, runtime, llm\n"
                "runtime.FLAGS.chat_fail_streak = 10; runtime.FLAGS.last_probe = time.monotonic()\n"
                "config.MODE = 'web'\n"
                "config.set_request_deadline(8)\n"
                "t = time.monotonic()\n"
                "try:\n    llm.chat([{'role': 'user', 'content': 'x'}])\nexcept llm.APIError:\n    pass\n"
                "print(round(time.monotonic() - t))")
        self.assertLessEqual(int(run_py(code, self.ENV)), 5)   # без дедлайна запроса ждал бы почти час

    def test_timed_out_answer_frees_worker(self) -> None:
        code = ("import time, runtime\n"
                "from assistant import LegalAssistant\n"
                "bot = LegalAssistant(answer_timeout=2)\n"
                "bot.warmup()\n"
                "futs = []\n"
                "orig = bot._pool.submit\n"
                "bot._pool.submit = lambda *a, **k: futs.append(orig(*a, **k)) or futs[-1]\n"
                "runtime.FLAGS.chat_fail_streak = 10; runtime.FLAGS.last_probe = time.monotonic()\n"
                "r = bot.answer_text('Какая арендная плата по договору с «Северным Складом»?')\n"
                "t = time.monotonic()\n"
                "futs[0].result(timeout=20)\n"
                "print(r.status, r.warnings[0]['code'] in ('timeout', 'llm_unavailable'), round(time.monotonic() - t) <= 6)")
        env = {**self.ENV, "LLM_BASE_URL": "http://127.0.0.1:9/v1", "LLM_API_KEY": "x", "LLM_MODEL": "m"}
        self.assertEqual(run_py(code, env, timeout=90), "error True True")   # без дедлайна поток висел бы почти час

    def test_batch_has_no_request_deadline(self) -> None:
        code = "import config\nprint(config.request_remaining() == float('inf'))"
        self.assertEqual(run_py(code, self.ENV), "True")


if __name__ == "__main__":
    unittest.main()
