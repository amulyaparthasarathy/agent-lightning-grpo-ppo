#!/usr/bin/env python3
"""Read-only OpenAI-compatible /models connectivity check, using only stdlib.

This sends no chat request, performs no tool call and does not establish that
the model can train. No automatic endpoint or cloud fallback is attempted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

MAX_RESPONSE_BYTES = 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", default=os.environ.get("LOCAL_LLM_BASE_URL"),
        help="Explicit API base URL, e.g. http://llama:9931/v1; defaults only to LOCAL_LLM_BASE_URL",
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    if not args.base_url:
        parser.error("Provide --base-url or LOCAL_LLM_BASE_URL. No fallback is selected.")
    if not 0 < args.timeout <= 60:
        parser.error("--timeout must be greater than zero and at most 60 seconds.")
    try:
        parts = urllib.parse.urlsplit(args.base_url.strip())
        port = parts.port
    except ValueError:
        parser.error("The API base URL is invalid.")
    if (
        parts.scheme not in {"http", "https"} or not parts.hostname
        or parts.username is not None or parts.password is not None
        or parts.query or parts.fragment
        or any(ord(c) < 32 for c in args.base_url)
    ):
        parser.error("Use an http(s) base URL without credentials, query, fragment or control characters.")
    del port  # Access above validates the URL's port syntax.
    base = urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))
    request_url = base + "/models"
    request = urllib.request.Request(request_url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=args.timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("response_too_large")
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise ValueError("unexpected_models_schema")
        model_ids = [
            item["id"] for item in data["data"]
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
        ]
    except urllib.error.HTTPError as exc:
        print(json.dumps({"status": "http_error", "base_url": base, "http_status": exc.code}))
        return 1
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeError, ValueError) as exc:
        # Avoid logging response bodies, proxy values, or credential material.
        print(json.dumps({"status": "check_failed", "base_url": base, "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({
        "status": "reachable",
        "base_url": base,
        "model_count": len(model_ids),
        "model_ids": [value[:256] for value in model_ids[:128]],
        "listed_ids_truncated": len(model_ids) > 128 or any(len(value) > 256 for value in model_ids[:128]),
        "chat_requests_sent": 0,
        "tool_calls_executed": 0,
        "training_verified": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
