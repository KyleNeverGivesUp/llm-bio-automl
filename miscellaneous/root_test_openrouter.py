"""Standalone OpenRouter account/credits check — one file, no project deps beyond dotenv.

Reads OPENROUTER_* from .env, makes one tiny chat call, and prints a clear verdict:
  - OK            -> key + credits + model all good
  - 401           -> bad/unknown key ("User not found")
  - 402           -> insufficient credits (add at openrouter.ai/settings/credits)
  - other         -> prints the status + message

Run:  uv run python test_openrouter.py
"""

from __future__ import annotations

import json
import os
from urllib import error, request

from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free")


def main() -> None:
    print(f"base_url : {BASE_URL}")
    print(f"model    : {MODEL}")
    print(f"key      : {(API_KEY[:10] + '...') if len(API_KEY) > 12 else '(missing/placeholder)'}")
    if not API_KEY or API_KEY == "YOUR_OPENROUTER_API_KEY":
        print("\nVERDICT: no real key set in .env (OPENROUTER_API_KEY).")
        return

    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 50,
        "messages": [
            {"role": "system", "content": "Reply ONLY with one JSON object."},
            {"role": "user", "content": "Return JSON: {\"ok\": true, \"n\": 2}."},
        ],
    }
    req = request.Request(
        url=f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "llm-bio-automl",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=60) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        content = raw.get("choices", [{}])[0].get("message", {}).get("content")
        usage = raw.get("usage", {})
        print(f"\nVERDICT: ✅ OK — account + credits + model working.")
        print(f"  response content: {content!r}")
        print(f"  tokens: {usage}")
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        hint = {401: "bad/unknown API key", 402: "insufficient credits (add at openrouter.ai/settings/credits)",
                404: "model id not found", 429: "rate limited"}.get(e.code, "")
        print(f"\nVERDICT: ❌ HTTP {e.code} {('— ' + hint) if hint else ''}")
        print(f"  detail: {detail[:300]}")
    except Exception as e:
        print(f"\nVERDICT: ❌ {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    main()
