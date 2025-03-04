import dash
from dash import dcc, html

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Hello, Binder!"),
    dcc.Graph(
        id="graph",
        figure={
            "data": [{"x": [1, 2, 3], "y": [10, 11, 12], "type": "line", "name": "Example"}],
            "layout": {"title": "Simple Dash Example"}
        }
    )
])

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
