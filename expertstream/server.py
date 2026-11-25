"""A small OpenAI-compatible server: POST /v1/chat/completions (no streaming) and GET /v1/models.

It listens on this machine only and has no authentication. Requests are answered one at a time.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .chat import Chat
from .engine import Engine

NAME = "granite-3.1-1b-a400m-instruct"
MAX_BODY = 1_000_000


class BadRequest(ValueError):
    pass


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(isinstance(p, dict) and p.get("type") == "text" for p in content):
        return "".join(str(p.get("text", "")) for p in content)
    raise BadRequest("message content must be text")


def complete(engine: Engine, chat: Chat, body: dict, context: int) -> dict:
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
        raise BadRequest("'messages' must be a non-empty list")
    if body.get("stream"):
        raise BadRequest("streaming is not supported")
    try:
        messages = [{"role": m["role"], "content": text_of(m["content"])} for m in body["messages"]]
        ids = chat.encode(messages)
        limit = int(body.get("max_completion_tokens") or body.get("max_tokens") or 400)
        temperature = float(body.get("temperature") or 0.0)
        seed = abs(int(body.get("seed") or 0))
    except BadRequest:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise BadRequest(f"bad request: {exc}") from exc
    room = context - len(ids)
    if room < 1 or limit < 1 or temperature < 0:
        raise BadRequest(f"the prompt is {len(ids)} tokens; the limit is {context}" if room < 1
                         else "max_tokens must be positive and temperature must not be negative")
    out = list(engine.generate(ids, min(limit, room), temperature, seed, context))
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion",
        "created": int(time.time()), "model": NAME,
        "choices": [{"index": 0, "finish_reason": "length" if len(out) >= min(limit, room) else "stop",
                     "message": {"role": "assistant", "content": chat.decode(out)}}],
        "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out),
                  "total_tokens": len(ids) + len(out)},
    }


def make_server(engine: Engine, chat: Chat, port: int, context: int = 4096) -> ThreadingHTTPServer:
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        timeout = 60

        def reply(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def error(self, status: int, message: str) -> None:
            self.reply(status, {"error": {"message": message, "type": "invalid_request_error"}})

        def do_GET(self) -> None:
            if self.path == "/v1/models":
                self.reply(200, {"object": "list", "data": [{"id": NAME, "object": "model"}]})
            else:
                self.error(404, "not found")

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                return self.error(404, "not found")
            # Browsers can post text/plain to local servers from any site; JSON needs permission.
            if self.headers.get_content_type() != "application/json":
                return self.error(415, "send application/json")
            try:
                size = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return self.error(411, "Content-Length is required")
            if not 0 < size <= MAX_BODY:
                return self.error(413, "the request is too large or empty")
            try:
                body = json.loads(self.rfile.read(size))
                with lock:
                    result = complete(engine, chat, body, context)
            except (BadRequest, json.JSONDecodeError, UnicodeDecodeError) as exc:
                return self.error(400, str(exc))
            self.reply(200, result)

        def log_message(self, *args) -> None:
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
