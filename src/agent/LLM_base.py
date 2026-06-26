"""Define shared agent interfaces and LLM JSON-call helpers."""

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request

from dotenv import load_dotenv

from src.agent.agent_context import AgentResult, RunContext

load_dotenv()


class BaseAgent(ABC):
    name: str

    @abstractmethod
    def run(self, context: RunContext) -> AgentResult:
        raise NotImplementedError


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    site_url: str
    app_name: str
    max_tokens: int
    temperature: float
    timeout: float = 60.0   # seconds; a slow/congested (free) endpoint fails fast -> caller falls back


def load_llm_config() -> LLMConfig:
    return LLMConfig(
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.environ.get("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY"),
        model=os.environ.get("OPENROUTER_MODEL", "minimax/minimax-m2.5:free"),
        site_url=os.environ.get("OPENROUTER_SITE_URL", "http://localhost"),
        app_name=os.environ.get("OPENROUTER_APP_NAME", "llm-bio-automl"),
        max_tokens=int(os.environ.get("OPENROUTER_MAX_TOKENS", "2000")),
        temperature=float(os.environ.get("OPENROUTER_TEMPERATURE", "0")),
        timeout=float(os.environ.get("OPENROUTER_TIMEOUT", "60")),
    )


class LLMJsonAgent(BaseAgent):
    def __init__(self, max_tokens: int | None = None) -> None:
        self.config = load_llm_config()
        self.max_tokens = max_tokens or self.config.max_tokens

    def _extract_text_content(self, raw: dict) -> str:
        choices = raw.get("choices") or []
        if not choices:
            raise ValueError("LLM response did not include any choices.")

        first_choice = choices[0]
        message = first_choice.get("message") or {}
        content = message.get("content")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            if text_parts:
                return "\n".join(text_parts)

        legacy_text = first_choice.get("text")
        if isinstance(legacy_text, str):
            return legacy_text

        reasoning = message.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip().startswith("{"):
            return reasoning

        raise ValueError("LLM response did not include text content.")

    def _extract_json(self, text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                if text.startswith("json"):
                    text = text[4:]
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("No JSON object found in LLM response.")
        return json.loads(match.group())

    def _write_llm_log(self, log_path: Path, payload: dict) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def call_json(self, system_prompt: str, user_prompt: str) -> dict:
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.config.site_url,
            "X-Title": self.config.app_name,
        }
        req = request.Request(
            url=f"{self.config.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.config.timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenRouter request failed: {exc.code} {detail}") from exc

        text = self._extract_text_content(raw)
        return self._extract_json(text)

    def call_json_logged(self, context: RunContext, call_name: str, system_prompt: str, user_prompt: str) -> dict:
        log_path = context.run_dir / "llm_logs" / f"{call_name}.json"
        request_payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        log_payload = {
            "status": "started",
            "call_name": call_name,
            "request": request_payload,
            "response_json": None,
            "raw_text_content": None,
            "parsed_json": None,
            "error": None,
        }

        try:
            body = json.dumps(request_payload).encode("utf-8")
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.config.site_url,
                "X-Title": self.config.app_name,
            }
            req = request.Request(
                url=f"{self.config.base_url}/chat/completions",
                data=body,
                headers=headers,
                method="POST",
            )

            try:
                with request.urlopen(req, timeout=self.config.timeout) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"OpenRouter request failed: {exc.code} {detail}") from exc

            text = self._extract_text_content(raw)
            parsed = self._extract_json(text)

            log_payload["status"] = "ok"
            log_payload["response_json"] = raw
            log_payload["raw_text_content"] = text
            log_payload["parsed_json"] = parsed
            self._write_llm_log(log_path, log_payload)
            return parsed
        except Exception as exc:
            log_payload["status"] = "error"
            log_payload["error"] = str(exc)
            self._write_llm_log(log_path, log_payload)
            raise
