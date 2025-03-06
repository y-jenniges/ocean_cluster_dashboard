from subprocess import Popen


def load_jupyter_server_extension(nbapp):
    """Serve the Dash app"""
    Popen(["python", "-m", "dashboard_ocean_cluster_visualisation"])
    # Popen(["bokeh", "serve", "bokeh-app", "--allow-websocket-origin=*"])
