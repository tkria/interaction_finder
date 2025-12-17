"""Logfire and logging configuration for interaction_finder.

Configures multi-handler logging:
- RichHandler (console, WARNING+ or INFO+ if verbose)
- LogfireLoggingHandler (structured telemetry, INFO+) when LOGFIRE_WRITE_TOKEN set
- FileHandler or BufferingHandler for file/error logging

Configuration is token-based (opt-in via LOGFIRE_WRITE_TOKEN).
"""

import contextlib
import logging
import os
from pathlib import Path
from typing import Any

# Set LiteLLM log level early, before it's imported
# This prevents debug spam if litellm is imported before configure_logging() runs
if "LITELLM_LOG" not in os.environ:
    os.environ["LITELLM_LOG"] = "WARNING"

# Third-party loggers to route through our handlers
_THIRD_PARTY_LOGGERS = (
    "LiteLLM",
    "LiteLLM Proxy",
    "LiteLLM Router",
    "httpx",
    "httpcore",
)


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


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_ERROR_LOG = Path("interaction-finder-error.log")


def error_log_path(output_path: Path | str | None) -> Path:
    """Derive error log path from output path."""
    if output_path:
        return Path(str(output_path) + ".log")
    return DEFAULT_ERROR_LOG


class BufferingHandler(logging.Handler):
    """Handler that buffers formatted log lines in memory for later dump."""

    def __init__(self, level: int = logging.DEBUG):
        super().__init__(level)
        self._lines: list[str] = []
        self.setFormatter(logging.Formatter(LOG_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:
        self._lines.append(self.format(record))

    def dump(self, path: Path) -> bool:
        """Write buffer to file and clear. Returns True if anything written."""
        if not self._lines:
            return False
        path.write_text("\n".join(self._lines) + "\n")
        self._lines.clear()
        return True


# Module state for file/buffer logging
_log_output_handler: logging.Handler | None = None


def _logfire_active() -> bool:
    """Check if logfire is configured with a token."""
    return bool(os.environ.get("LOGFIRE_WRITE_TOKEN"))


def setup_log_output(
    path: Path | str | None = None,
    level: int = logging.INFO,
) -> None:
    """Configure log output to file or in-memory buffer. Call once at startup.

    If path specified, logs directly to file. If no path and no logfire token,
    buffers in memory (retrieve via dump_log on error). If logfire is active
    and no path specified, does nothing (logfire captures logs).

    Parameters:
        path: Path | str | None — log file path, or None for buffer/skip
        level: int — logging level for file/buffer output (default INFO)
    """
    global _log_output_handler
    if _log_output_handler is not None:
        return  # Already configured
    # If logfire is active and no explicit file requested, skip
    if not path and _logfire_active():
        return
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if path:
        handler: logging.Handler = logging.FileHandler(path)
    else:
        handler = BufferingHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(handler)
    _log_output_handler = handler


def dump_log(path: Path | str) -> Path | None:
    """Write buffered logs to path and clear buffer.

    Returns:
        Path to written file if anything was written, None otherwise.
        Returns None if not using in-memory buffering or buffer is empty.
    """
    if isinstance(_log_output_handler, BufferingHandler):
        out = Path(path)
        if _log_output_handler.dump(out):
            return out
    return None


def configure_logging(console=None, verbose: bool = False) -> None:
    """Configure console logging with RichHandler + LogfireLoggingHandler.

    Can be called multiple times (e.g., to update console for Live displays).
    Preserves any FileHandler or BufferingHandler set up via setup_log_output().

    Parameters:
        console: Console | None — Rich console (creates if None)
        verbose: bool — if True, RichHandler shows INFO+; otherwise WARNING+

    Example:
        >>> configure_logging(console=Console(), verbose=False)
        >>> logging.warning("Shown above Live display")
        >>> logging.info("Only to logfire")
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Route third-party logs through root (clear their handlers, enable propagation)
    level = logging.INFO if verbose else logging.WARNING
    for name in _THIRD_PARTY_LOGGERS:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(level)
    # Remove existing RichHandler and LogfireLoggingHandler, preserve file/buffer
    try:
        from rich.logging import RichHandler
    except ImportError:
        RichHandler = None  # type: ignore[misc, assignment]
    for handler in root.handlers[:]:
        # Remove RichHandler
        if RichHandler and isinstance(handler, RichHandler):
            root.removeHandler(handler)
        # Remove LogfireLoggingHandler (check by module to avoid import)
        elif type(handler).__module__.startswith("logfire"):
            root.removeHandler(handler)
    # Add fresh console handler
    try:
        import logfire
        from rich.console import Console
        from rich.logging import RichHandler
        from rich.theme import Theme

        theme = Theme({"log.time": "dim bright_black"})
        if console is None:
            console = Console(theme=theme)
        else:
            console.push_theme(theme, inherit=True)
        console_level = logging.INFO if verbose else logging.WARNING
        root.addHandler(
            RichHandler(
                console=console,
                show_path=False,
                rich_tracebacks=True,
                tracebacks_show_locals=verbose,
                level=console_level,
            )
        )
        root.addHandler(logfire.LogfireLoggingHandler(level=logging.INFO))
    except ImportError:
        # Fallback: simple StreamHandler if rich/logfire unavailable
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO if verbose else logging.WARNING)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        root.addHandler(handler)


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
