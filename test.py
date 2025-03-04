import dash
from dash import dcc, html

app = dash.Dash(__name__)

app.layout = html.Div("Hello, Binder!")


if __name__ == "__main__":
    #app.run_server(host="0.0.0.0", port=8050, debug=True)
    app.run_server(debug=True)
