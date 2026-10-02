"""Minimal OpenAI-compatible chat client with tool calling.

Gemini, Groq and Ollama all expose an OpenAI-compatible /chat/completions
endpoint, so one client covers all three free options. No SDK dependency.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

PROVIDERS = {
    # name: (base_url, api-key env var or None, default model)
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-3.8-flash"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "ollama": ("http://localhost:11434/v1", None, "qwen2.5:7b"),
}


@dataclass
class ChatClient:
    provider: str = "gemini"
    model: str | None = None
    temperature: float = 0.2
    max_retries: int = 6

    def __post_init__(self):
        base, key_env, default_model = PROVIDERS[self.provider]
        self.base = os.environ.get("SIZEAGENT_BASE_URL", base)
        self.model = self.model or default_model
        self.key = os.environ.get(key_env, "") if key_env else "ollama"
        if key_env and not self.key:
            raise RuntimeError(f"set {key_env} to use provider '{self.provider}'")

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        body = json.dumps({"model": self.model, "messages": messages, "tools": tools,
                           "tool_choice": "auto", "temperature": self.temperature}).encode()
        req = urllib.request.Request(f"{self.base}/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.key}"})
        delay = 5.0
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read())
                return data["choices"][0]["message"]
            except urllib.error.HTTPError as e:
                # Free tiers rate-limit aggressively; back off on 429/5xx.
                if e.code in (429, 500, 502, 503) and attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise RuntimeError(f"{self.provider} HTTP {e.code}: {e.read()[:500]!r}") from e
        raise RuntimeError("unreachable")
