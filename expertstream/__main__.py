"""Command line: python -m expertstream chat | serve"""

from __future__ import annotations

import argparse

from .chat import Chat
from .engine import Engine
from .policies import POLICIES
from .server import make_server


def main() -> None:
    parser = argparse.ArgumentParser(prog="expertstream", description=__doc__)
    parser.add_argument("command", choices=("chat", "serve"))
    parser.add_argument("--model", default="models/granite")
    parser.add_argument("--budget-mb", type=int, default=256, help="memory for cached experts")
    parser.add_argument("--policy", choices=sorted(POLICIES), default="lfu")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    engine = Engine(args.model, args.budget_mb, args.policy, args.prefetch)
    chat = Chat(args.model)

    if args.command == "serve":
        print(f"http://127.0.0.1:{args.port}/v1  (Ctrl+C to stop)", flush=True)
        try:
            make_server(engine, chat, args.port).serve_forever()
        except KeyboardInterrupt:
            return
    messages: list[dict] = []
    while True:
        try:
            messages.append({"role": "user", "content": input("> ")})
        except (EOFError, KeyboardInterrupt):
            return
        out: list[int] = []
        shown = 0
        for token in engine.generate(chat.encode(messages), limit=400):
            out.append(token)
            text = chat.decode(out)
            if not text.endswith("�"):  # wait for the rest of a split character
                print(text[shown:], end="", flush=True)
                shown = len(text)
        print()
        messages.append({"role": "assistant", "content": chat.decode(out)})


if __name__ == "__main__":
    main()
