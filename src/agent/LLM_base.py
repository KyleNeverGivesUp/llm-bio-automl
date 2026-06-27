"""Define shared agent interfaces and LLM JSON-call helpers."""

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    timeout: float = 60.0   # seconds; a slow/congested (free) endpoint fails fast -> rotate/fall back
    models: list[str] = field(default_factory=list)   # rotation: tried in order on 429/timeout


# Free models to rotate through when one is rate-limited (verified live on OpenRouter). The shared
# free endpoints get throttled upstream independently, so trying the next usually succeeds.
_FREE_ROTATION = [
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-20b:free",
]


def load_llm_config() -> LLMConfig:
    model = os.environ.get("OPENROUTER_MODEL", _FREE_ROTATION[0])
    models_env = os.environ.get("OPENROUTER_MODELS", "").strip()
    if models_env:                                    # explicit override, comma-separated
        models = [m.strip() for m in models_env.split(",") if m.strip()]
    else:                                             # primary first, then the rest of the free rotation
        models = [model] + [m for m in _FREE_ROTATION if m != model]
    return LLMConfig(
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.environ.get("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY"),
        model=model,
        site_url=os.environ.get("OPENROUTER_SITE_URL", "http://localhost"),
        app_name=os.environ.get("OPENROUTER_APP_NAME", "llm-bio-automl"),
        max_tokens=int(os.environ.get("OPENROUTER_MAX_TOKENS", "2000")),
        temperature=float(os.environ.get("OPENROUTER_TEMPERATURE", "0")),
        timeout=float(os.environ.get("OPENROUTER_TIMEOUT", "60")),
        models=models,
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

    def _post_rotating(self, messages: list[dict]) -> dict:
        """POST the chat request, rotating through config.models on rate-limit/timeout/overload.

        The free endpoints are throttled upstream independently, so when one 429s the next usually
        works. Returns the raw response dict; raises only if EVERY model fails (caller then falls back).
        """
        models = self.config.models or [self.config.model]
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.config.site_url,
            "X-Title": self.config.app_name,
        }
        last_err: Exception | None = None
        for i, model in enumerate(models):
            payload = {"model": model, "temperature": self.config.temperature,
                       "max_tokens": self.max_tokens, "messages": messages}
            req = request.Request(url=f"{self.config.base_url}/chat/completions",
                                  data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.config.timeout) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                if i:
                    print(f"[LLM] rotated to {model} (previous {i} model(s) rate-limited/failed)")
                return raw
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                last_err = RuntimeError(f"OpenRouter {model} failed: {exc.code} {detail[:100]}")
            except Exception as exc:  # timeout / network — rotate to next model
                last_err = RuntimeError(f"OpenRouter {model} error: {exc}")
        raise last_err or RuntimeError("all models failed")

    def call_json(self, system_prompt: str, user_prompt: str) -> dict:
        raw = self._post_rotating([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
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
            raw = self._post_rotating(request_payload["messages"])  # rotates models on 429/timeout
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
