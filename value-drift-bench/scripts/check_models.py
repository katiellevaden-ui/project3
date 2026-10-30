#!/usr/bin/env python3
"""Check every model in models.yaml against OpenRouter: is the slug listed, which provider
serves it (with the pin applied), and does a minimal request succeed? Run before anything else.

    python scripts/check_models.py            # all models
    python scripts/check_models.py qwen3.5-9b # some
"""
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # same file Inspect reads
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from valuedrift.registry import load_models  # noqa: E402

key = os.environ.get("OPENROUTER_API_KEY") or sys.exit("set OPENROUTER_API_KEY (env or .env)")
listed = {m["id"] for m in requests.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]}
for m in load_models(nicks=sys.argv[1:] or None):
    if m.slug not in listed:
        print(f"{m.nick:22} {m.slug:55} NOT LISTED"); continue
    body = {"model": m.slug, "max_tokens": 5, "messages": [{"role": "user", "content": "Say OK."}]}
    if m.provider:
        body["provider"] = {"order": [m.provider], "allow_fallbacks": False}
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                      headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120).json()
    if "error" in r:
        print(f"{m.nick:22} {m.slug:55} ERROR {str(r['error'].get('message'))[:70]}")
    else:
        print(f"{m.nick:22} {m.slug:55} ok  provider={r.get('provider')}")
