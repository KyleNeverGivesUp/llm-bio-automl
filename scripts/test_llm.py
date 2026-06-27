"""Quick OpenRouter API test — verify the key / model / credits actually work.

  uv run python scripts/test_llm.py                              # test .env OPENROUTER_MODEL
  uv run python scripts/test_llm.py google/gemini-2.5-flash-lite # test a specific model
  uv run python scripts/test_llm.py openai/gpt-4o-mini

Prints SUCCESS (reply + token count + cost estimate) or the HTTP error (402/429/404) with a hint.
Reads the key from .env but never prints it in full.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_env() -> dict:
    env = {}
    for line in Path(".env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def main() -> None:
    env = load_env()
    key = env.get("OPENROUTER_API_KEY")
    base = env.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    model = sys.argv[1] if len(sys.argv) > 1 else env.get("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")
    if not key:
        print("✗ no OPENROUTER_API_KEY in .env")
        return

    print(f"model = {model}")
    print(f"key   = {key[:10]}…{key[-4:]}  (len {len(key)})")
    print("calling …")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Reply with ONLY a JSON object, nothing else."},
            {"role": "user", "content": 'Return exactly: {"ok": true, "hi": "<one short word>"}'},
        ],
        "temperature": 0,
        "max_tokens": 50,
    }
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        resp = json.load(urllib.request.urlopen(req, timeout=30))
        dt = time.time() - t0
        msg = resp["choices"][0]["message"]["content"].strip()
        u = resp.get("usage", {})
        pt, ct = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        print(f"\n✓ SUCCESS  ({dt:.1f}s)")
        print(f"  reply : {msg[:120]}")
        print(f"  tokens: prompt={pt}  completion={ct}")
        # rough cost for THIS call + extrapolate to a full ~15-call run
        print(f"  → this one call's tokens are tiny; a full pipeline run ≈ 15 such calls (still well under 1¢)")
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        print(f"\n✗ HTTP {e.code}")
        print(f"i'm body  {body}")
        hint = {402: "余额不足 → 充值 https://openrouter.ai/settings/credits",
                429: "限流(免费档)→ 换付费模型,或给 call_json 加退避重试",
                404: "模型 ID 不存在 → 查目录 https://openrouter.ai/api/v1/models",
                401: "key 无效 → 检查 .env 的 OPENROUTER_API_KEY"}.get(e.code)
        if hint:
            print(f"  → {hint}")
    except Exception as e:  # noqa: BLE001
        print(f"\n✗ {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
