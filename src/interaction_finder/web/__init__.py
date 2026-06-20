"""Local web UI for running and viewing interaction-finder pipelines.

A Starlette app that drives the same async pipeline the CLI uses, streaming
live progress to the browser over SSE and rendering the self-contained HTML
report. Starlette/uvicorn are imported lazily inside ``run_ui_server`` so that
importing the rest of the package never pulls in the web stack.

Public entry point: ``run_ui_server``.
"""

from interaction_finder.web.table import WebStatusTable


def run_ui_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    open_browser: bool = True,
) -> None:
    """Serve the web UI and (optionally) open a browser to it.

    Parameters:
        host: Interface to bind; defaults to loopback (single-user, local).
        port: TCP port to listen on.
        open_browser: Open the system browser at the served URL on startup.

    Blocks until the server is interrupted.
    """
    import threading
    import webbrowser

    import uvicorn

    from interaction_finder.web.app import create_app

    app = create_app()
    if open_browser:  # pragma: no cover - browser side effect
        # Fire shortly after the server starts accepting connections.
        threading.Timer(0.6, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")


__all__ = ["WebStatusTable", "run_ui_server"]
