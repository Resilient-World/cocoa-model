"""
Structured logging configuration (structlog).

Bind ``service``, package ``version``, and optional ``trace_id`` on every event.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog

trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("resilient-cocoa-model")
    except Exception:
        return "unknown"


def _add_trace_id(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    tid = trace_id_var.get()
    if tid is not None:
        event_dict["trace_id"] = tid
    return event_dict


def configure_logging(level: str = "INFO", *, json: bool = True) -> None:
    """
    Configure structlog + stdlib logging for the application.

    Parameters
    ----------
    level
        Log level name (DEBUG, INFO, WARNING, ERROR).
    json
        If True, emit JSON lines; otherwise console-friendly key=value output.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_trace_id,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        service="cocoa-model",
        version=_package_version(),
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to the given module name."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]


__all__ = ["configure_logging", "get_logger", "trace_id_var"]
