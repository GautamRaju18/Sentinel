"""structlog configuration — human-readable in dev, JSON when piped."""

from __future__ import annotations

import logging
import sys

import structlog


def configure_observability(level: str = "INFO", json_logs: bool = False) -> None:
    """One call to set up both local logging and remote tracing.

    Bundled deliberately. Every entry point already calls configure_logging;
    piggybacking tracing on it means a new entry point cannot forget to enable
    tracing while remembering to enable logs.
    """
    configure_logging(level, json_logs)

    from sentinel.observability import configure_tracing

    configure_tracing()


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    # stderr, never stdout. The MCP stdio transport owns stdout: a single log
    # line written there corrupts the JSON-RPC framing and the client drops
    # the message.
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level)

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
