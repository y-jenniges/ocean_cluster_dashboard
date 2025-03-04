import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import gsw


def update_geo_and_umap(column="label", hide_noise=True, label_selection=None):
    print("update geo and umap", label_selection)
    if label_selection is None:
        label_selection = []

    df_display = df.copy()

    # determine labels to display
    if label_selection:
        df_display = df_display[df_display[column].isin(label_selection)]

    # hide noise
    if hide_noise:
        df_display = df_display[df_display[column] != -1]

    # define figures
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

    figure_geo.update_layout(margin=dict(l=margin, r=margin, t=margin, b=margin),
                             scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Depth [m]"),
                             uirevision=True)
    figure_umap.update_layout(margin=dict(l=margin, r=margin, t=margin, b=margin),
                              scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                              uirevision=True)

    return figure_geo, figure_umap


def update_depth(depth_idx, hide_noise=True, column="label"):
    print("update depth", depth_idx)
    df_display = df.copy()

    # hide noise
    if hide_noise:
        df_display = df_display[df_display[column] != -1]

    # filter for depth
    df_display = df_display[df_display.LEV_M == depths[depth_idx]]

    # define figure
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
    figure_depth.update_layout(margin=dict(l=margin, r=margin, t=margin, b=margin), uirevision=True)
    figure_depth.update_geos(
        lonaxis_range=[round(df.LONGITUDE.min()) - margin, round(df.LONGITUDE.max()) + margin],
        lataxis_range=[round(df.LATITUDE.min()) - margin, round(df.LATITUDE.max()) + margin])
    return figure_depth


def update_ts(column="label", hide_noise=True, label_selection=None):
    """ Plot T-S diagram. """
    print("update TS", label_selection)
    if label_selection is None:
        label_selection = []

    df_display = df.copy()

    # determine labels to display
    if label_selection:
        df_display = df_display[df_display[column].isin(label_selection)]

    # hide noise
    if hide_noise:
        df_display = df_display[df_display[column] != -1]

    # limits
    smin = df_display["abs_salinity"].min() - (0.01 * df_display["abs_salinity"].min())
    smax = df_display["abs_salinity"].max() + (0.01 * df_display["abs_salinity"].max())
    tmin = df_display["cons_temperature"].min() - (0.1 * df_display["cons_temperature"].max())
    tmax = df_display["cons_temperature"].max() + (0.1 * df_display["cons_temperature"].max())

    # number of gridcells in the x and y dimensions
    xdim = int(round((smax - smin) / 0.1 + 1, 0))
    ydim = int(round((tmax - tmin) / 0.1 + 1, 0))

    # empty grid
    dens = np.zeros((ydim, xdim))

    # temperature and salinity vectors
    si = np.linspace(1, xdim - 1, xdim) * 0.1 + smin
    ti = np.linspace(1, ydim - 1, ydim) * 0.1 + tmin

    # fill grid with densities
    for j in range(0, int(ydim)):
        for i in range(0, int(xdim)):
            dens[j, i] = gsw.rho(si[i], ti[j], 0)

    # convert to sigma-t
    dens = dens - 1000

    figure_ts = go.Figure()

    # add density contours
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

    # add the scatter plot of the actual data points
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

    figure_ts.update_layout(margin=dict(l=margin, r=margin, t=margin, b=margin),
                            xaxis=dict(title='Absolute salinity [g/kg]', titlefont=dict(size=14)),
                            yaxis=dict(title='Conservative temperature [°C]', titlefont=dict(size=14)),
                            showlegend=False,
                            uirevision=True)

    return figure_ts


# plot settings
scatter_size = 2
margin = 5

# load file to visualise
# df = pd.read_csv("../output_final/dbscan/uncertainty/UMAP_DBSCAN/umap_dbscan_7.csv")
# df = utils.color_code_labels(df)

# df = pd.read_csv("../output_final/dbscan/post_processing/re-assigned_A1_R1.csv")
# data_label = "label"

# df = pd.read_csv("../output_final/kmeans_labels.csv")
# df = df[df.n_clusters == 2].rename(columns={"label_original": "label"})
# df = df[df.n_clusters == 10].rename(columns={"label_embedding": "label"})

# df = pd.read_csv("../output_final/ward_labels.csv")
# df = df[df.n_clusters == 2].rename(columns={"label_original": "label"})
# df = df[df.n_clusters == 24].rename(columns={"label_embedding": "label"})

# df = utils.color_code_labels(df)
# df[["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"]] = 0

# df = pd.read_csv("../output_final/dbscan/uncertainty/uncertainty.csv")
# df.uncertainty = round(df.uncertainty)
# cmap = matplotlib.cm.get_cmap("viridis")
# color_map = {label: matplotlib.colors.rgb2hex(cmap(label/100)) for label in np.sort(df['uncertainty'].unique())}
# df["color"] = df["uncertainty"].map(color_map)
# data_label = "uncertainty"

# df = pd.read_csv("../nemi_iteration67_uncertainty.csv")
# cmap = matplotlib.cm.get_cmap("viridis")
# color_map = {label: matplotlib.colors.rgb2hex(cmap(label / 100)) for label in np.sort(df['uncertainty'].unique())}
# df["color"] = df["uncertainty"].map(color_map)
# data_label = "uncertainty"

# df = pd.read_csv("../nemi_iteration67_uncertainty.csv").rename(columns={"label_color": "color"})
# data_label = "final_label"
# df = df[df.final_label != 10]
# df = df[(df.final_label != 10) & (df.uncertainty < 10)]
# df = df[df.final_label.isin([65, 68, 72])]

# cmap = matplotlib.cm.get_cmap("gray")
# color_map_g = {label: matplotlib.colors.rgb2hex(cmap(label / 100)) for label in
#                np.sort(df[df.uncertainty < 50].final_label.unique())}
# df.loc[df.uncertainty < 50, "color"] = df[df.uncertainty < 50].final_label.map(color_map_g)
# df.loc[df.uncertainty >= 50, "color"] = "#ff0000"

df = pd.read_csv("../output_final/dbscan/uncertainty/volume_nemi_iteration67_uncertainty.csv").rename(columns={"label_color": "color"})
data_label = "final_label"  # final_label uncertainty
df = df[df.final_label != 10]
# df = df[df.final_label.isin([29, 17, 26, 4, 21, 37])]
# df = df[df.uncertainty < 50]
# cmap = matplotlib.cm.get_cmap("gray")
# color_map_g = {label: matplotlib.colors.rgb2hex(cmap(label / 100)) for label in
#                np.sort(df[df.uncertainty < 50].final_label.unique())}
# df.loc[df.uncertainty < 50, "color"] = df[df.uncertainty < 50].final_label.map(color_map_g)
# df.loc[df.uncertainty >= 50, "color"] = "#ff0000"
# df.loc[df.uncertainty >= 50, "final_label"] = "500"
# cmap = matplotlib.cm.get_cmap("hot")
# df["color"] = df.uncertainty.map(lambda x: matplotlib.colors.rgb2hex(cmap(x/100)))


# df = pd.read_csv("../areaWeight_nemi_iteration5_uncertainty.csv").rename(columns={"label_color": "color"})
# data_label = "final_label"
# df = df[df.final_label != 9]
# df = df[(df.final_label != 9) & (df.uncertainty >= 50)]
# df = df[df.final_label.isin([65, 68, 73])]

# df = pd.read_csv("../areaWeight_nemi_iteration5_uncertainty.csv")
# cmap = matplotlib.cm.get_cmap("viridis")
# color_map = {label: matplotlib.colors.rgb2hex(cmap(label / 100)) for label in np.sort(df['uncertainty'].unique())}
# df["color"] = df["uncertainty"].map(color_map)
# df = df[df["uncertainty"] >= 50]
# data_label = "uncertainty"

# add cluster size information
sizes = df.final_label.value_counts().reset_index()
df = pd.merge(df, sizes, on="final_label", how="left")

# add information required for TS diagrams
df["pressure"] = gsw.p_from_z(-1 * df["LEV_M"], df["LATITUDE"])
df["abs_salinity"] = gsw.SA_from_SP(df["P_SALINITY"], df["pressure"], df["LONGITUDE"], df["LATITUDE"])
df["cons_temperature"] = gsw.CT_from_pt(df["abs_salinity"], df["P_TEMPERATURE"])
df["rho"] = gsw.rho(df["abs_salinity"], df["cons_temperature"], df["pressure"])

# define depths
depths = np.sort(df.LEV_M.unique())
cur_depth_idx = 0

# figures
fig_geo, fig_umap = update_geo_and_umap(column=data_label, hide_noise=True, label_selection=[])
fig_depth = update_depth(column=data_label, hide_noise=True, depth_idx=cur_depth_idx)
fig_ts = update_ts(column=data_label, hide_noise=True, label_selection=[])

# dash app and layout
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
    # print("update callback")
    # get data from previous state
    prev_depth_idx = cur_params["depth_idx"]
    prev_selected_labels = cur_params["selected_labels"]
    prev_clickdata_depth = cur_params["clickData_depth"]
    prev_clickdata_sel_depth = cur_params["clickDataSel_depth"]

    # init new state
    new_geo_fig = figure_geo
    new_umap_fig = figure_umap
    new_depth_fig = figure_depth
    new_ts_fig = figure_ts
    new_params = {"depth_idx": prev_depth_idx,
                  "selected_labels": prev_selected_labels,
                  "clickData_depth": prev_clickdata_depth,
                  "clickDataSel_depth": prev_clickdata_sel_depth}

    # if depth slider changed, update depth figure
    if new_depth_idx != prev_depth_idx:
        print(f"new_depth {new_depth_idx}, prev_depth {prev_depth_idx}")
        new_depth_fig = update_depth(depth_idx=new_depth_idx, column=data_label, hide_noise=True)
        new_params["depth_idx"] = new_depth_idx

    # if a click happened in the depth label plot, show that specific cluster only (select and deselect?)
    if clickdata_sel_depth:
        print("clickDataSel depth", clickdata_sel_depth)
        if selection_state == "select all":
            new_selected_labels = []
        else:
            new_selected_labels = list(set([e["text"] for e in clickdata_sel_depth["points"]]))
        new_params["selected_labels"] = new_selected_labels
        new_params["clickDataSel_depth"] = clickdata_sel_depth

        # update geo and umap plot accordingly
        new_geo_fig, new_umap_fig = update_geo_and_umap(label_selection=new_selected_labels, column=data_label)
        print("updated geo and umap")

        # update TS plot
        new_ts_fig = update_ts(label_selection=new_selected_labels, column=data_label)
        print("updated TS")

    elif clickdata_depth['points'][0]['lon']:
        print("clickData depth", clickdata_depth)

        # find the label of the clicked point
        lat = clickdata_depth['points'][0]['lat']
        lon = clickdata_depth['points'][0]['lon']

        selected_label = df[(df.LATITUDE == lat) &
                            (df.LONGITUDE == lon) &
                            (df.LEV_M == depths[new_depth_idx])][data_label]
        if selected_label.empty:
            selected_label = []
        else:
            selected_label = [selected_label.values[0]]

        print(lat, lon, prev_selected_labels)

        # only update figures if the click data is different to the previous click data
        new_selected_labels = prev_selected_labels
        if selection_state == "select all":
            new_selected_labels = []

        if prev_clickdata_depth != clickdata_depth:
            if selection_state == "select":
                new_selected_labels = prev_selected_labels + selected_label
            elif selection_state == "deselect":
                new_selected_labels = [x for x in prev_selected_labels if x != selected_label[0]]

        # print(lat, lon, new_selected_labels)
        print(new_selected_labels)

        # update label selection
        new_params["selected_labels"] = new_selected_labels
        new_params["clickData_depth"] = clickdata_depth

        # update geo and umap plot accordingly
        new_geo_fig, new_umap_fig = update_geo_and_umap(label_selection=new_selected_labels, column=data_label)

        # update TS plot accordingly
        new_ts_fig = update_ts(label_selection=new_selected_labels, column=data_label)

    return new_geo_fig, new_umap_fig, new_depth_fig, new_ts_fig, new_params


# import pandas as pd
# import numpy as np
# data_label = "label_embedding"
# # data_label = "label_original"
# labels = pd.read_csv("data/dbscan_labels.csv")
# df_in = pd.read_csv("data/df_wide_knn.csv")
def get_info(cluster_label, labels, df_in, eps=0.10983051, min_samples=4):
    # filter for the correct hyperparameter combination
    temp = labels[(np.round(labels.eps, 8) == np.round(eps, 8)) & (labels.min_samples == min_samples)]
    all_labels = temp[data_label].unique()  # find all labels of that clustering

    # check if cluster label exists
    if cluster_label in all_labels:
        temp = temp[temp[data_label] == cluster_label]  # filter out the cluster label
        # merge labels to parameter information
        temp_merged = pd.merge(left=temp, right=df_in, how="left", on=["LATITUDE", "LONGITUDE", "LEV_M"])
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(f"Information on cluster label {cluster_label}")
            print(temp_merged[[x for x in temp_merged.columns
                               if x not in ["eps", "min_samples", "label_embedding", "label_original"]]].describe())

        return temp_merged
    else:
        print(f"Cluster label {cluster_label} not found.")
        return


def difference_between_two_labels(a, b, labels, df_in, eps=0.10983051, min_samples=4):
    a_data = get_info(a, labels, df_in, eps, min_samples)
    b_data = get_info(b, labels, df_in, eps, min_samples)

    diff = (a_data.describe() - b_data.describe())
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(f"Information on the difference of cluster labels {a} and {b}")
        print(diff[[x for x in a_data.columns if x not in ["eps", "min_samples", "label_embedding", "label_original"]]])

    # return diff


# run app
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
