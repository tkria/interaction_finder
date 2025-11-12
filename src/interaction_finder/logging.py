"""Logfire configuration and setup for interaction_finder.

This module configures logfire on import, making it available throughout
the application. Configuration is token-based (opt-in via LOGFIRE_WRITE_TOKEN)
with automatic Pydantic AI instrumentation.
"""

import contextlib
import os


def configure_logfire(verbose: bool = False) -> None:
    """Configure logfire if LOGFIRE_WRITE_TOKEN is available.

    Disables inspect_arguments because the codebase already uses explicit
    keyword arguments for structured logging, making f-string introspection
    redundant while avoiding AST parsing overhead and warnings.

    Parameters:
        verbose: bool — enable verbose console output (default: False)
    """
    token = os.environ.get("LOGFIRE_WRITE_TOKEN")
    import logfire
    from logfire import ConsoleOptions

    if verbose:
        coptions = ConsoleOptions()
    else:
        coptions = ConsoleOptions(min_log_level="warn", show_project_link=False)
    _ = logfire.configure(
        send_to_logfire="if-token-present",
        token=token,
        scrubbing=False,
        console=coptions,
        inspect_arguments=False,
    )
    _ = logfire.instrument_pydantic_ai()


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
