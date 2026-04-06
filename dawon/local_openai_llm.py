from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class LocalOpenAICompatLLM:
    model: str = "Qwen"
    base_url: str = "http://127.0.0.1:1225/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    timeout: float = 600.0
    max_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        self.model = self.model or os.environ.get("DAWON_MODEL", "Qwen")
        self.base_url = (self.base_url or os.environ.get("DAWON_BASE_URL") or "http://127.0.0.1:1225/v1").rstrip("/")
        self.api_key = self.api_key or os.environ.get("DAWON_API_KEY", "EMPTY")
        self._chat_url = self._resolve_chat_url(self.base_url)
        self._session = requests.Session()

    @staticmethod
    def _resolve_chat_url(base_url: str) -> str:
        if base_url.endswith("/chat/completions"):
            return base_url
        if base_url.endswith("/v1"):
            return f"{base_url}/chat/completions"
        return f"{base_url}/v1/chat/completions"

    def healthcheck(self) -> None:
        if self.base_url.endswith("/v1"):
            health_url = f"{self.base_url[:-3]}/health"
        elif self.base_url.endswith("/chat/completions"):
            health_url = self.base_url.rsplit("/v1/", 1)[0] + "/health"
        else:
            health_url = f"{self.base_url}/health"
        try:
            response = self._session.get(health_url, timeout=min(self.timeout, 30.0))
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Could not reach the local LLM server at {health_url}. "
                "Start `dawon/start_qwen32b_vllm.sh` first or use --skip_healthcheck."
            ) from exc

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        response = self._session.post(
            self._chat_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            snippet = response.text[:2000]
            raise RuntimeError(f"LLM call failed: status={response.status_code} body={snippet}") from exc

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM returned non-JSON response: {response.text[:1000]}") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"Unexpected chat completion payload: {json.dumps(data, ensure_ascii=False)[:2000]}") from exc

        if not content:
            raise RuntimeError("Local LLM returned empty content.")
        return content
