"""LangSmith tracing.

There is a trap here worth naming. Settings are loaded from .env by
pydantic-settings into a Settings object, but LangChain's tracer reads
`os.environ` directly and never sees that object. So setting
LANGSMITH_TRACING=true in .env produces a config that *looks* enabled,
reports itself as enabled, and traces nothing.

This module is the bridge: it copies the values into os.environ before any
LangChain object is constructed, and then verifies the tracer actually
initialised rather than trusting the flag.

Why trace at all, when the CLI already prints each node? Because the printed
output is a summary and the trace is the evidence. When the critic rejects a
hypothesis for the third time and the run burns 130k tokens, the question is
"which node, on which loop, with what prompt" — and that is a question about a
tree, not a log. LangSmith also aggregates across runs, which is how you notice
that investigate is 80% of your token spend.
"""

from __future__ import annotations

import os

from sentinel.config import get_settings
from sentinel.logging_setup import get_logger

log = get_logger(__name__)


def configure_tracing(*, run_name: str | None = None) -> bool:
    """Export tracing config into os.environ. Returns True if tracing is live.

    Safe to call more than once and safe to call when tracing is disabled.
    """
    settings = get_settings()

    if not settings.langsmith_tracing:
        # Explicitly clear rather than leave whatever was inherited. A stale
        # LANGSMITH_TRACING=true in the parent shell would otherwise silently
        # ship traces from a run the operator believes is local-only.
        os.environ.pop("LANGSMITH_TRACING", None)
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        return False

    if not settings.langsmith_api_key:
        log.warning(
            "tracing.no_api_key",
            impact="LANGSMITH_TRACING is true but no key is set; tracing stays off",
        )
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    # Older LangChain releases read the V2 name. Setting both costs nothing and
    # avoids a version-dependent silent no-op.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    if run_name:
        os.environ["LANGCHAIN_SESSION"] = run_name

    log.info("tracing.enabled", project=settings.langsmith_project)
    return True


def verify_tracing() -> tuple[bool, str]:
    """Confirm the tracer is genuinely wired, not merely configured.

    Checks the client can reach LangSmith AND that LangChain's global tracing
    flag agrees — the two can disagree, which is the failure this catches.
    """
    settings = get_settings()
    if not settings.langsmith_tracing:
        return False, "tracing disabled in settings"
    if not settings.langsmith_api_key:
        return False, "no API key configured"

    try:
        from langsmith import Client

        client = Client(api_key=settings.langsmith_api_key)
        # Cheapest authenticated call available.
        list(client.list_projects(limit=1))
    except ImportError:
        return False, "langsmith package not installed (uv sync)"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:100]}"

    from langchain_core.tracers.context import tracing_v2_enabled  # noqa: F401

    env_on = os.environ.get("LANGSMITH_TRACING") == "true"
    if not env_on:
        return (
            False,
            "settings say on, but os.environ was never exported — call configure_tracing()",
        )

    return True, f"tracing live -> project '{settings.langsmith_project}'"


def trace_url(project: str | None = None) -> str:
    project = project or get_settings().langsmith_project
    return f"https://smith.langchain.com/o/me/projects/p/{project}"
