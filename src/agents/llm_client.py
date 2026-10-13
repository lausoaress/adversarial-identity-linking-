"""Provider-agnostic LLM client.

You said you want to run against a local model first (no API costs while
the architecture is still being shaped) and switch providers later. Every
agent in this project talks to the model only through the `LLMClient`
interface below, so swapping Ollama -> OpenAI -> Anthropic later is a
one-line change in config/config.yaml, not a rewrite of the agents.

Three implementations:

  - OllamaClient   — talks to a local Ollama server (http://localhost:11434
                     by default). This is the "local model first" path:
                     `ollama pull llama3.1` (or mistral, qwen2.5, etc.),
                     then `ollama serve`, and this client just works.
  - OpenAIClient / AnthropicClient — thin wrappers for later, once you
    want to compare local-model quality against a hosted one. Both are
    optional imports: the project runs fine without those packages
    installed if you're only using Ollama.
  - StubLLMClient  — returns scripted responses. Used by tests and by
    scripts/run_demo.py so the pipeline is runnable and testable *before*
    any real model (local or hosted) is wired up.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, *, json_mode: bool = False) -> str:
        """Returns the model's raw text response. If `json_mode` is True,
        callers expect (and implementations should request, where the
        provider supports it) a single JSON object as the whole response.
        """
        raise NotImplementedError


@dataclass
class OllamaClient(LLMClient):
    model: str = "llama3.1"
    host: str = "http://localhost:11434"
    temperature: float = 0.2
    timeout_s: int = 120

    def complete(self, system_prompt: str, user_prompt: str, *, json_mode: bool = False) -> str:
        import requests  # local import: keep this an optional dependency path

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if json_mode:
            payload["format"] = "json"

        resp = requests.post(
            f"{self.host}/api/chat", json=payload, timeout=self.timeout_s
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]


@dataclass
class OpenAIClient(LLMClient):
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    temperature: float = 0.2

    def complete(self, system_prompt: str, user_prompt: str, *, json_mode: bool = False) -> str:
        from openai import OpenAI  # optional dependency — only needed if you use this client

        client = OpenAI(api_key=self.api_key)
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        )
        return response.choices[0].message.content


@dataclass
class AnthropicClient(LLMClient):
    model: str = "claude-sonnet-4-5"
    api_key: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 2048

    def complete(self, system_prompt: str, user_prompt: str, *, json_mode: bool = False) -> str:
        from anthropic import Anthropic  # optional dependency

        client = Anthropic(api_key=self.api_key)
        suffix = "\n\nRespond with a single JSON object and nothing else." if json_mode else ""
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt + suffix,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in message.content if block.type == "text")


@dataclass
class StubLLMClient(LLMClient):
    """Scripted responses, in call order, by role. Use this to develop
    and unit-test the Proposer/Contestador/Juiz prompts and parsing logic
    without a model running at all — see tests/test_agents_smoke.py and
    scripts/run_demo.py.
    """

    scripted_responses: list[str] = field(default_factory=list)
    _calls: list[dict] = field(default_factory=list, repr=False)

    def complete(self, system_prompt: str, user_prompt: str, *, json_mode: bool = False) -> str:
        self._calls.append({"system": system_prompt, "user": user_prompt, "json_mode": json_mode})
        if not self.scripted_responses:
            raise RuntimeError(
                "StubLLMClient has no scripted responses left — add more to "
                "`scripted_responses`, one per expected agent call."
            )
        return self.scripted_responses.pop(0)

    @property
    def call_count(self) -> int:
        return len(self._calls)


def build_client(provider: str, **kwargs) -> LLMClient:
    """Factory used by config/config.yaml's `llm.provider` setting."""
    providers = {
        "ollama": OllamaClient,
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "stub": StubLLMClient,
    }
    if provider not in providers:
        raise ValueError(f"Unknown LLM provider '{provider}'. Choose from {list(providers)}.")
    return providers[provider](**kwargs)


def safe_json_parse(text: str) -> dict:
    """Agents ask for JSON but local models sometimes wrap it in prose or
    markdown fences anyway — this strips the common cases before parsing,
    so one flaky response doesn't crash a whole experiment run.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model response: {text!r}")
    return json.loads(cleaned[start : end + 1])
