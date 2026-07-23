"""Verify every configured credential works, without ever printing one.

Run this after rotating keys. It answers "did I paste them correctly" with a
live call per service, and reports only pass/fail plus a fingerprint.

The fingerprint is a short hash of the value, not a prefix of it. A prefix
leaks real key material into terminal scrollback and CI logs; a hash lets you
confirm two machines hold the same secret, or that a value actually changed
after a rotation, while revealing nothing.

    uv run python scripts/check_credentials.py
"""

from __future__ import annotations

import hashlib
import sys

import httpx
from rich.console import Console
from rich.table import Table

from sentinel.config import get_settings

c = Console()


def fingerprint(secret: str) -> str:
    """Short stable hash. Never a prefix of the real value."""
    if not secret:
        return "—"
    return hashlib.sha256(secret.encode()).hexdigest()[:8]


def check_openrouter(key: str) -> tuple[str, str]:
    if not key:
        return "skip", "not set — planner/reasoner tiers will fail"
    try:
        r = httpx.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
        )
        if r.status_code == 401:
            return "FAIL", "401 — key rejected (revoked or mistyped)"
        r.raise_for_status()
        data = r.json().get("data", {})
        usage, limit = data.get("usage"), data.get("limit")
        detail = f"valid · usage={usage}"
        if limit is not None:
            detail += f" limit={limit}"
        if data.get("is_free_tier"):
            detail += " · free tier"
        return "OK", detail
    except Exception as e:
        return "FAIL", f"{type(e).__name__}: {str(e)[:80]}"


def check_github(token: str) -> tuple[str, str]:
    if not token:
        return "skip", "not set — push over HTTPS will prompt instead"
    try:
        r = httpx.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if r.status_code == 401:
            return "FAIL", "401 — token rejected (revoked or mistyped)"
        r.raise_for_status()
        login = r.json().get("login", "?")

        # Auth alone is not enough: the fine-grained PAT must also carry
        # Contents:write for this repo, which is the failure that cost us
        # several rounds earlier in the project.
        probe = httpx.get(
            "https://api.github.com/repos/GautamRaju18/Sentinel",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if probe.status_code == 404:
            return "WARN", f"auth ok as {login}, but repo not visible to this token"
        perms = probe.json().get("permissions", {})
        if not perms.get("push"):
            return "WARN", f"auth ok as {login}, but no push permission"
        return "OK", f"valid as {login} · push=True"
    except Exception as e:
        return "FAIL", f"{type(e).__name__}: {str(e)[:80]}"


def check_huggingface(token: str) -> tuple[str, str]:
    if not token:
        return "skip", "not set — only needed to retrain"
    try:
        r = httpx.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if r.status_code == 401:
            return "FAIL", "401 — token rejected (revoked or mistyped)"
        r.raise_for_status()
        return "OK", f"valid as {r.json().get('name', '?')}"
    except Exception as e:
        return "FAIL", f"{type(e).__name__}: {str(e)[:80]}"


def check_langsmith(key: str, tracing: bool) -> tuple[str, str]:
    if not key:
        return "skip", "not set — tracing is off, so unused"
    try:
        r = httpx.get(
            "https://api.smith.langchain.com/info",
            headers={"x-api-key": key},
            timeout=30,
        )
        if r.status_code in (401, 403):
            return "FAIL", "rejected — key revoked or mistyped"
        r.raise_for_status()
        return "OK", f"valid · tracing={'on' if tracing else 'off'}"
    except Exception as e:
        return "FAIL", f"{type(e).__name__}: {str(e)[:80]}"


def check_ollama(base_url: str) -> tuple[str, str]:
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=15)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
        return "OK", f"{len(names)} models: {', '.join(names[:4])}"
    except Exception as e:
        return "FAIL", f"not reachable at {base_url} — {type(e).__name__}"


def check_postgres(url: str) -> tuple[str, str]:
    try:
        import psycopg

        with psycopg.connect(url, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT extname FROM pg_extension WHERE extname='vector'")
                has_vector = cur.fetchone() is not None
        return ("OK", "connected · pgvector present") if has_vector else (
            "WARN",
            "connected but pgvector extension is missing",
        )
    except Exception as e:
        return "FAIL", f"{type(e).__name__}: {str(e)[:80]}"


def main() -> int:
    s = get_settings()

    checks = [
        ("OpenRouter", s.openrouter_api_key, check_openrouter(s.openrouter_api_key)),
        ("GitHub", s.github_token, check_github(s.github_token)),
        ("HuggingFace", s.huggingface_token, check_huggingface(s.huggingface_token)),
        (
            "LangSmith",
            s.langsmith_api_key,
            check_langsmith(s.langsmith_api_key, s.langsmith_tracing),
        ),
        ("Ollama", "", check_ollama(s.ollama_base_url)),
        ("Postgres", "", check_postgres(s.database_url)),
    ]

    table = Table(title="credential + service check")
    table.add_column("service")
    table.add_column("fingerprint", style="dim")
    table.add_column("status")
    table.add_column("detail")

    failed = 0
    for name, secret, (status, detail) in checks:
        colour = {"OK": "green", "WARN": "yellow", "FAIL": "red", "skip": "dim"}[status]
        if status == "FAIL":
            failed += 1
        table.add_row(name, fingerprint(secret), f"[{colour}]{status}[/]", detail)

    c.print(table)
    c.print(
        "\n[dim]Fingerprints are sha256 prefixes, not key material — safe to share. "
        "If one is unchanged after a rotation, that value was not actually replaced.[/]"
    )

    if failed:
        c.print(f"\n[red]{failed} check(s) failed[/]")
        return 1
    c.print("\n[green]all configured credentials valid[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
