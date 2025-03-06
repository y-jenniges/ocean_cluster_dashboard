from subprocess import Popen


def load_jupyter_server_extension(nbapp):
    """Serve the Dash app using Flask."""
    Popen([
        "python", "-m", "dashserver",
        "--host=0.0.0.0", "--port=5006"
    ])

   # Popen(["bokeh", "serve", "bokeh-app", "--allow-websocket-origin=*"])
