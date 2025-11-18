"""Logfire and logging configuration for interaction_finder.

Configures dual-handler logging: RichHandler (console, WARNING+) and
LogfireLoggingHandler (structured telemetry, INFO+). This ensures clean
console output during Live progress displays while preserving comprehensive
structured logging for analysis.

Configuration is token-based (opt-in via LOGFIRE_WRITE_TOKEN).
"""

import contextlib
import logging
import os
from typing import Any


def configure_logfire(verbose: bool = False) -> None:
    """Configure logfire with console output disabled.

    Console output handled via Python logging with RichHandler to prevent
    duplicates and integrate cleanly with Live displays.

    Parameters:
        verbose: bool — unused (console output controlled via configure_logging)
    """
    import logfire

    logfire.configure(
        send_to_logfire="if-token-present",
        token=os.environ.get("LOGFIRE_WRITE_TOKEN"),
        scrubbing=False,
        console=False,
        inspect_arguments=False,
    )
    logfire.instrument_pydantic_ai()


_STANDARD_KWARGS = frozenset(("exc_info", "stack_info", "stacklevel", "extra"))


class StructuredLogger:
    """Logger wrapper auto-wrapping kwargs in extra={} for Python logging."""

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        standard = {k: v for k, v in kwargs.items() if k in _STANDARD_KWARGS}
        extra = {k: v for k, v in kwargs.items() if k not in _STANDARD_KWARGS}
        if extra:
            standard["extra"] = {**standard.get("extra", {}), **extra}
        self._logger.log(level, msg, *args, **standard)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.ERROR, msg, *args, **kwargs)


def get_logger(name: str) -> StructuredLogger:
    """Get StructuredLogger accepting logfire-style keyword arguments."""
    return StructuredLogger(name)


def configure_logging(console=None, verbose: bool = False) -> None:
    """Configure Python logging with RichHandler + LogfireLoggingHandler.

    Parameters:
        console: Console | None — Rich console (creates if None)
        verbose: bool — if True, RichHandler shows INFO+; otherwise WARNING+

    Example:
        >>> configure_logging(console=Console(), verbose=False)
        >>> logging.warning("Shown above Live display")
        >>> logging.info("Only to logfire")
    """
    try:
        import logfire
        from rich.console import Console
        from rich.logging import RichHandler
        from rich.theme import Theme

        # Custom theme with dim grey (bright_black) timestamps
        theme = Theme({"log.time": "dim bright_black"})
        if console is None:
            console = Console(theme=theme)
        else:
            # Push custom theme onto existing console (inherits other styles)
            console.push_theme(theme, inherit=True)
        console_level = logging.INFO if verbose else logging.WARNING

        handlers = [
            RichHandler(
                console=console,
                show_path=False,
                rich_tracebacks=True,
                tracebacks_show_locals=verbose,
                level=console_level,
            ),
            logfire.LogfireLoggingHandler(level=logging.INFO),
        ]

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(message)s",
            handlers=handlers,
            force=True,
        )
    except ImportError:
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format="%(levelname)s: %(message)s",
            force=True,
        )


# No-op fallback for when logfire is unavailable
class _NoOpLogfire:
    """No-op implementation of logfire for graceful fallback."""

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def span(self, *args, **kwargs):
        """Return a null context manager for span operations."""
        return contextlib.nullcontext()

    def instrument(self, *args, **kwargs):
        """No-op decorator that returns the function unchanged."""

        def decorator(func):
            return func

        return decorator


# Try to configure logfire, fall back to no-op if unavailable
try:
    import logfire as _logfire

    # Check verbose flag from environment variable
    verbose = os.environ.get("LOGFIRE_VERBOSE", "").lower() in ("1", "true", "yes")
    configure_logfire(verbose=verbose)
    logfire = _logfire
except ImportError:
    logfire = _NoOpLogfire()
