import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import gsw
from IPython.display import display, HTML


def update_geo_and_umap(column="label", hide_noise=True, label_selection=None):
    """ Update the 3d geoplot and the UMAP plot. """
    # Update list of selected labels
    if label_selection is None:
        label_selection = []

    # Copy dataframe
    df_display = df.copy()

    # Determine labels to display
    if label_selection:
        df_display = df_display[df_display[column].isin(label_selection)]

    # Hide noise
    if hide_noise:
        df_display = df_display[df_display[column] != -1]

    # Define figures (and hover templates)
    figure_geo = go.Figure(data=go.Scatter3d(x=df_display.LONGITUDE, y=df_display.LATITUDE, z=df_display.LEV_M * -1,
                                             mode='markers',
                                             marker=dict(size=scatter_size, color=df_display.color, opacity=1),
                                             hovertemplate='Longitude: %{x}<br>' +
                                                           'Latitude: %{y}<br>' +
                                                           'Depth: %{z} m<br>' +
                                                           'Temperature: %{text[0]:.2f} °C<br>' +
                                                           'Salinity: %{text[1]:.2f} psu<br>' +
                                                           'Oxygen: %{text[2]:.2f} µmol/kg<br>' +
                                                           'Nitrate: %{text[3]:.2f} µmol/kg<br>' +
                                                           'Silicate: %{text[4]:.2f} µmol/kg<br>' +
                                                           'Phosphate: %{text[5]:.2f} µmol/kg<br>' +
                                                           'Label: %{text[6]}<extra></extra>',
                                             text=df_display[["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN",
                                                              "P_NITRATE", "P_SILICATE", "P_PHOSPHATE",
                                                              column]]
                                             ))
    figure_umap = go.Figure(data=go.Scatter3d(x=df_display.e0, y=df_display.e1, z=df_display.e2,
                                              mode='markers',
                                              marker=dict(size=scatter_size, color=df_display.color, opacity=1),
                                              hovertemplate='X: %{x:.2f}<br>' +
                                                            'Y: %{y:.2f}<br>' +
                                                            'Z: %{z:.2f}<br>' +
                                                            'Temperature: %{text[0]:.2f} °C<br>' +
                                                            'Salinity: %{text[1]:.2f} psu<br>' +
                                                            'Oxygen: %{text[2]:.2f} µmol/kg<br>' +
                                                            'Nitrate: %{text[3]:.2f} µmol/kg<br>' +
                                                            'Silicate: %{text[4]:.2f} µmol/kg<br>' +
                                                            'Phosphate: %{text[5]:.2f} µmol/kg<br>' +
                                                            'Label: %{text[6]}<extra></extra>',
                                              text=df_display[["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN",
                                                               "P_NITRATE", "P_SILICATE", "P_PHOSPHATE",
                                                               column]]
                                              ))

    # Update figure layout
    figure_geo.update_layout(margin=dict(l=margin, r=margin, t=margin, b=margin),
                             scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Depth [m]",
                                        xaxis=dict(range=[longitude_min, longitude_max]),
                                        yaxis=dict(range=[latitude_min, latitude_max]),
                                        zaxis=dict(range=[depth_min, depth_max])
                                        ),
                             uirevision=True)
    figure_umap.update_layout(margin=dict(l=margin, r=margin, t=margin, b=margin),
                              scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                              uirevision=True)

    return figure_geo, figure_umap


def update_depth(depth_idx, hide_noise=True, column="label"):
    """ Update the depth plot. """

    # Copy dataframe
    df_display = df.copy()

    # Hide noise
    if hide_noise:
        df_display = df_display[df_display[column] != -1]

    # Filter for depth
    df_display = df_display[df_display.LEV_M == depths[depth_idx]]

    # Define figure
    figure_depth = go.Figure(data=go.Scattergeo(lon=df_display.LONGITUDE,
                                                lat=df_display.LATITUDE,
                                                mode='markers',
                                                marker=dict(color=df_display.color, opacity=1),
                                                hovertemplate='Longitude: %{lon}<br>' +
                                                              'Latitude: %{lat}<br>' +
                                                              'Temperature: %{text[0]:.2f} °C<br>' +
                                                              'Salinity: %{text[1]:.2f} psu<br>' +
                                                              'Oxygen: %{text[2]:.2f} µmol/kg<br>' +
                                                              'Nitrate: %{text[3]:.2f} µmol/kg<br>' +
                                                              'Silicate: %{text[4]:.2f} µmol/kg<br>' +
                                                              'Phosphate: %{text[5]:.2f} µmol/kg<br>' +
                                                              'Label: %{text[6]}<extra></extra>',
                                                text=df_display[["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN",
                                                                 "P_NITRATE", "P_SILICATE", "P_PHOSPHATE",
                                                                 column]]
                                                ))

    # Update layout
    figure_depth.update_layout(margin=dict(l=margin, r=margin, t=margin, b=margin), uirevision=True)
    figure_depth.update_geos(
        lonaxis_range=[round(df.LONGITUDE.min()) - margin, round(df.LONGITUDE.max()) + margin],
        lataxis_range=[round(df.LATITUDE.min()) - margin, round(df.LATITUDE.max()) + margin])

    return figure_depth


def update_ts(column="label", hide_noise=True, label_selection=None):
    """ Update the T-S diagram. """
    # Update selected labels
    if label_selection is None:
        label_selection = []

    # Copy dataframe
    df_display = df.copy()

    # Determine labels to display
    if label_selection:
        df_display = df_display[df_display[column].isin(label_selection)]

    # Hide noise
    if hide_noise:
        df_display = df_display[df_display[column] != -1]

    # Define salinity and temperature limits
    smin = df_display["abs_salinity"].min() - (0.01 * df_display["abs_salinity"].min())
    smax = df_display["abs_salinity"].max() + (0.01 * df_display["abs_salinity"].max())
    tmin = df_display["cons_temperature"].min() - (0.1 * df_display["cons_temperature"].max())
    tmax = df_display["cons_temperature"].max() + (0.1 * df_display["cons_temperature"].max())

    # Number of gridcells in the x and y dimensions
    xdim = int(round((smax - smin) / 0.1 + 1, 0))
    ydim = int(round((tmax - tmin) / 0.1 + 1, 0))

    # Define empty grid
    dens = np.zeros((ydim, xdim))

    # Temperature and salinity vectors
    si = np.linspace(1, xdim - 1, xdim) * 0.1 + smin
    ti = np.linspace(1, ydim - 1, ydim) * 0.1 + tmin

    # Fill grid with densities
    for j in range(0, int(ydim)):
        for i in range(0, int(xdim)):
            dens[j, i] = gsw.rho(si[i], ti[j], 0)

    # Convert density to sigma-t
    dens = dens - 1000

    # Define figure
    figure_ts = go.Figure()

    # Add density contours
    figure_ts.add_trace(go.Contour(
        x=si, y=ti, z=dens,
        colorscale=None, showscale=False, hoverinfo="skip",
        contours=dict(
            coloring='lines',
            showlabels=True,
            labelfont=dict(color='black'),
            labelformat='.1f'  # one decimal place
        ),
        line=dict(width=0.5, dash="dash", color="black")
    ))

    # Add the scatter plot of the actual data points
    figure_ts.add_trace(go.Scatter(
        x=df_display['abs_salinity'],
        y=df_display['cons_temperature'],
        mode='markers', marker=dict(color=df_display.color, opacity=1),
        hovertemplate='Longitude: %{text[1]}<br>' +
                      'Latitude: %{text[0]}<br>' +
                      'Depth: %{text[2]} m<br>' +
                      'Cons. temperature [°C]: %{y:.2f} °C<br>' +
                      'Abs. salinity [g/kg]: %{x:.2f} psu<br>' +
                      'Label: %{text[3]}<extra></extra>',
        text=df_display[["LATITUDE", "LONGITUDE", "LEV_M", column]]
    ))

    # Update layout
    figure_ts.update_layout(margin=dict(l=margin, r=margin, t=margin, b=margin),
                            xaxis=dict(title={"text": 'Absolute salinity [g/kg]', "font": {"size": 14}}),
                            yaxis=dict(title={"text": 'Conservative temperature [°C]', "font": {"size": 14}}),
                            showlegend=False,
                            uirevision=True)

    return figure_ts


# Plot settings
scatter_size = 2
margin = 5

# Load data
df = pd.read_csv("cluster_set.csv")
data_label = "label"
df = df[df[data_label] != 10]

# Add cluster size information
sizes = df[data_label].value_counts().reset_index()
df = pd.merge(df, sizes, on=data_label, how="left")

# Compute and add information required for TS diagram
df["pressure"] = gsw.p_from_z(-1 * df["LEV_M"], df["LATITUDE"])
df["abs_salinity"] = gsw.SA_from_SP(df["P_SALINITY"], df["pressure"], df["LONGITUDE"], df["LATITUDE"])
df["cons_temperature"] = gsw.CT_from_pt(df["abs_salinity"], df["P_TEMPERATURE"])
df["rho"] = gsw.rho(df["abs_salinity"], df["cons_temperature"], df["pressure"])

# Define depth levels
depths = np.sort(df.LEV_M.unique())
cur_depth_idx = 0

# Define range of axis in geoplot
longitude_min = df.LONGITUDE.min()
longitude_max = df.LONGITUDE.max()
latitude_min = df.LATITUDE.min()
latitude_max = df.LATITUDE.max()
depth_min = (-1 * df.LEV_M).min()
depth_max = (-1 * df.LEV_M).max()

# Define figures
fig_geo, fig_umap = update_geo_and_umap(column=data_label, hide_noise=True, label_selection=[])
fig_depth = update_depth(column=data_label, hide_noise=True, depth_idx=cur_depth_idx)
fig_ts = update_ts(column=data_label, hide_noise=True, label_selection=[])

# Dash app and layout
app = Dash(__name__)
app.layout = html.Div([
    html.Div(
        dcc.Graph(id='fig-geo', figure=fig_geo),
        style={'margin': dict(l=margin, r=margin, t=margin, b=margin), 'display': 'inline-block', 'width': '49vw'}
    ),
    html.Div(
        dcc.Graph(id='fig-umap', figure=fig_umap),
        style={'margin': dict(l=margin, r=margin, t=margin, b=margin), 'display': 'inline-block', 'width': '49vw'}
    ),
    html.Div(
        dcc.Graph(id="fig-depth", figure=fig_depth, clickData={'points': [{'lon': None, 'lat': None}]}),
        style={'display': 'inline-block', 'width': '49vw'}
    ),
    html.Div(
        [dcc.Slider(id="depth-slider", min=0, max=len(depths) - 1, step=None, value=cur_depth_idx,
                    marks={i: str(x) for i, x in enumerate(depths)}, vertical=True),
         dcc.RadioItems(id="selection-state", value='select all',
                        options=['select all', 'select', 'deselect'], labelStyle={'display': 'inline-block'})],
        style={'display': 'inline-block', 'width': '14vw'}
    ),
    html.Div(
        dcc.Graph(id="fig-ts", figure=fig_ts),  # , clickData={'points': [{'lon': None, 'lat': None}]}),
        style={'display': 'inline-block', 'width': '35vw'}
    ),
    dcc.Store(id="cur-params", data={"depth_idx": cur_depth_idx, "selected_labels": [],
                                     "clickDataSel_depth": {},
                                     'clickData_depth': {'points': [{'lon': None, 'lat': None}]}}
              )
])


@app.callback(
    Output('fig-geo', 'figure'),
    Output('fig-umap', 'figure'),
    Output('fig-depth', 'figure'),
    Output('fig-ts', 'figure'),
    Output('cur-params', 'data'),

    Input('fig-geo', 'figure'),
    Input('fig-umap', 'figure'),
    Input('fig-depth', 'figure'),
    Input('fig-ts', 'figure'),
    Input('fig-depth', 'clickData'),
    Input('fig-depth', 'selectedData'),
    Input('depth-slider', 'value'),
    Input('selection-state', 'value'),
    Input('cur-params', 'data')
)
def update(figure_geo, figure_umap, figure_depth, figure_ts, clickdata_depth, clickdata_sel_depth, new_depth_idx,
           selection_state, cur_params):
    # Get data from previous state
    prev_depth_idx = cur_params["depth_idx"]
    prev_selected_labels = cur_params["selected_labels"]
    prev_clickdata_depth = cur_params["clickData_depth"]
    prev_clickdata_sel_depth = cur_params["clickDataSel_depth"]

    # Init new state
    new_geo_fig = figure_geo
    new_umap_fig = figure_umap
    new_depth_fig = figure_depth
    new_ts_fig = figure_ts
    new_params = {"depth_idx": prev_depth_idx,
                  "selected_labels": prev_selected_labels,
                  "clickData_depth": prev_clickdata_depth,
                  "clickDataSel_depth": prev_clickdata_sel_depth}

    # If depth slider changed, update depth figure
    if new_depth_idx != prev_depth_idx:
        new_depth_fig = update_depth(depth_idx=new_depth_idx, column=data_label, hide_noise=True)
        new_params["depth_idx"] = new_depth_idx

    # If a click happened in the depth label plot, show that specific cluster only (select and deselect?)
    if clickdata_sel_depth:
        if selection_state == "select all":
            new_selected_labels = []
        else:
            new_selected_labels = list(set([e["text"] for e in clickdata_sel_depth["points"]]))
        new_params["selected_labels"] = new_selected_labels
        new_params["clickDataSel_depth"] = clickdata_sel_depth

        # Update geo and umap plot accordingly
        new_geo_fig, new_umap_fig = update_geo_and_umap(label_selection=new_selected_labels, column=data_label)

        # Update TS plot
        new_ts_fig = update_ts(label_selection=new_selected_labels, column=data_label)

    elif clickdata_depth['points'][0]['lon']:
        # Find the label of the clicked point
        lat = clickdata_depth['points'][0]['lat']
        lon = clickdata_depth['points'][0]['lon']

        selected_label = df[(df.LATITUDE == lat) &
                            (df.LONGITUDE == lon) &
                            (df.LEV_M == depths[new_depth_idx])][data_label]
        if selected_label.empty:
            selected_label = []
        else:
            selected_label = [selected_label.values[0]]

        # Only update figures if the click data is different to the previous click data
        new_selected_labels = prev_selected_labels
        if selection_state == "select all":
            new_selected_labels = []

        if prev_clickdata_depth != clickdata_depth:
            if selection_state == "select":
                new_selected_labels = prev_selected_labels + selected_label
            elif selection_state == "deselect":
                new_selected_labels = [x for x in prev_selected_labels if x != selected_label[0]]

        # Update label selection
        new_params["selected_labels"] = new_selected_labels
        new_params["clickData_depth"] = clickdata_depth

        # Update geo and umap plot accordingly
        new_geo_fig, new_umap_fig = update_geo_and_umap(label_selection=new_selected_labels, column=data_label)

        # Update TS plot accordingly
        new_ts_fig = update_ts(label_selection=new_selected_labels, column=data_label)

    return new_geo_fig, new_umap_fig, new_depth_fig, new_ts_fig, new_params


# Run app
# if __name__ == "__main__":

# Increase width and height of display
display(HTML("""
<style>
    /* Increase notebook cell width */
    .container { width: 95% !important; }

    /* Increase Dash app iframe height */
    iframe {
        width: 100% !important;
        height: 1000px !important; 
    }
</style>
"""))

server = app.server  # Required for gunicorn

# app.run_server(mode="inline", host="127.0.0.1", port=8060, debug=True)  # For local windows machine
# app.run_server(mode="inline", debug=True)  # For Binder
# app.run_server(mode="inline", host="0.0.0.0", port=8050, debug=True)
app.run_server(host="0.0.0.0", port=5006, debug=True)
