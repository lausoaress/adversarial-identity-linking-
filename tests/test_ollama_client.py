"""Tests OllamaClient against a tiny fake HTTP server that mimics the
real Ollama /api/chat contract (https://github.com/ollama/ollama/blob/main/docs/api.md#chat-request-with-history):

    POST /api/chat  {"model", "messages", "stream": false, ...}
    -> {"message": {"role": "assistant", "content": "..."}}

This exists so bugs in our request/response handling are caught *before*
spending time pulling a multi-GB model on real hardware (Ollama) to test
against - if this fails, the integration is wrong regardless of which
model you point it at.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.agents.llm_client import OllamaClient

LAST_REQUEST: dict = {}


class _FakeOllamaHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - silence test server logs
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        LAST_REQUEST.clear()
        LAST_REQUEST.update(body)

        if self.path != "/api/chat":
            self.send_response(404)
            self.end_headers()
            return

        reply_content = (
            '{"verdict": "accept", "explanation": "stub", "cited_evidence_descriptions": []}'
            if body.get("format") == "json"
            else "plain text reply"
        )
        response = {
            "model": body.get("model", "unknown"),
            "message": {"role": "assistant", "content": reply_content},
            "done": True,
        }
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture()
def fake_ollama_server():
    server = HTTPServer(("127.0.0.1", 0), _FakeOllamaHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_ollama_client_sends_chat_shaped_request(fake_ollama_server):
    client = OllamaClient(model="llama3.1", host=fake_ollama_server)
    result = client.complete("system prompt", "user prompt", json_mode=False)

    assert result == "plain text reply"
    # Exactly the request shape the real Ollama /api/chat endpoint expects.
    assert LAST_REQUEST["model"] == "llama3.1"
    assert LAST_REQUEST["stream"] is False
    assert LAST_REQUEST["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]
    assert "format" not in LAST_REQUEST


def test_ollama_client_requests_json_format_when_asked(fake_ollama_server):
    client = OllamaClient(model="llama3.1", host=fake_ollama_server)
    result = client.complete("system prompt", "user prompt", json_mode=True)

    assert LAST_REQUEST["format"] == "json"
    parsed = json.loads(result)
    assert parsed["verdict"] == "accept"


def test_ollama_client_raises_on_connection_refused():
    # Nothing listening on this port - simulates "forgot to run ollama serve"
    # (on Windows, the installer runs this automatically in the background;
    # a refused connection there usually means Ollama isn't installed/running).
    client = OllamaClient(model="llama3.1", host="http://127.0.0.1:1")
    with pytest.raises(Exception):
        client.complete("system prompt", "user prompt")
