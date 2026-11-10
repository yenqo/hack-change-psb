"""CLI для ручной проверки фасада: вопрос или диалог → JSON-результат (контракт docs/integration.md).

    python tools/ask.py "Какая арендная плата по договору с «Северным Складом»?"
    python tools/ask.py --dialog "Подскажите гарантию по договору с «Альфой»." \\
                                 "Уточните: «Альфа-Трейд» или «Альфа-Трейдинг»?" \\
                                 "«Альфа-Трейдинг». На сколько гарантия?"
    python tools/ask.py --docs                      # каталог документов
    python tools/ask.py --doc <doc_id>              # документ с разметкой пунктов
    python tools/ask.py --health

Окружение ядра — как обычно (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, DATA_DIR …); ключи не выводятся.

--dev: поднять tools/dev_proxy.py (провайдер из .env, локальная bge-m3) и направить ядро в него — только для разработки.
--stub: поднять tools/stub_proxy.py (детерминированная заглушка без сети).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Вопрос → JSON фасада LegalAssistant")
    ap.add_argument("question", nargs="?", help="вопрос (или строка в формате платформы с историей)")
    ap.add_argument("--dialog", nargs="+", help="реплики по очереди: пользователь, ассистент, …, пользователь")
    ap.add_argument("--docs", action="store_true", help="каталог документов")
    ap.add_argument("--doc", help="документ по doc_id")
    ap.add_argument("--health", action="store_true")
    ap.add_argument("--debug", action="store_true", help="заполнить поле debug")
    ap.add_argument("--dev", action="store_true", help="через tools/dev_proxy.py (только разработка)")
    ap.add_argument("--stub", action="store_true", help="через tools/stub_proxy.py (без сети)")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    overrides = {}
    proxy = None
    if args.dev:
        import dev_proxy
        proxy = dev_proxy.DevProxy(stage="dev", purpose="tools/ask.py").start()
        overrides = proxy.core_env()
    elif args.stub:
        import stub_proxy
        _, url = stub_proxy.start()
        overrides = {"LLM_BASE_URL": url, "LLM_API_KEY": "stub", "LLM_MODEL": "stub-model"}
    if not os.environ.get("DATA_DIR") and (ROOT / "data" / "knowledge-base.zip").is_file():
        overrides.setdefault("DATA_DIR", str(ROOT / "data"))

    from assistant import LegalAssistant
    bot = LegalAssistant(overrides=overrides, answer_timeout=args.timeout, debug=args.debug)
    try:
        health = bot.warmup()
        if args.health:
            out = health
        elif args.docs:
            out = bot.list_documents()
        elif args.doc:
            out = bot.get_document(args.doc) or {"error": f"нет документа {args.doc}"}
        elif args.dialog:
            msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(args.dialog)]
            out = bot.answer(msgs, on_stage=lambda s: print(f"… {s}", file=sys.stderr)).to_dict()
        elif args.question:
            out = bot.answer_text(args.question, on_stage=lambda s: print(f"… {s}", file=sys.stderr)).to_dict()
        else:
            ap.print_help()
            return 2
        print(json.dumps(out, ensure_ascii=False, indent=2))
    finally:
        if proxy is not None:
            proxy.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
