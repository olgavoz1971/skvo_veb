import logging

from dash import register_page, html, dcc, callback, ctx, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import base64
import io
import pandas as pd
import numpy as np
import plotly.express as px

app_color = {"graph_bg": '#A3E4D7', "graph_line": "#007ACE"}
app_margins = {'left': 20, 'right': 20, 'top': 20, 'bottom': 20}

register_page(__name__, name='Characteristic Curve',
              path='/igebc/cc',
              title='Characteristic Curve Tool',
              in_navbar=False)


def layout():
    return dbc.Container([
        dcc.Download(id='download-var'),
        dcc.Download(id='download-row'),
        html.H1(id='Header', children='Characteristic curve', className="text-primary text-left fs-3"),
        html.Br(),
        dbc.Row([
            dbc.Stack([
                dcc.Upload(id='upload-standards',
                           children=dbc.Button('Upload standards', size="md", class_name='me-3', color='light')),
                dcc.Upload(id='upload_inst',
                           children=dbc.Button('Upload measurements', size="md", class_name='me-3', color='light')),

                # html.Button('Download results', id='btn-download'),
                dbc.Button('Calculate var', id='btn_calc_var', size="md", class_name='me-3', color='light'),
                dbc.Button('Download var', id='btn-download-var', size="md", class_name='me-3', color='light'),
                dbc.Button('Download row', id='btn-download-row', size="md", class_name='me-3', color='light'),
                dbc.Stack([
                    dbc.Label('Column:', html_for='dropdown-column'),
                    dcc.Dropdown(
                        id="dropdown-column",
                        multi=False,
                        value=None,
                        options=[],
                        style={'width': '20ch'},
                        placeholder='Upload measurements'
                    ),
                ], direction='horizontal', gap=1),
            ], direction='horizontal', gap=1),
        ], justify="start", align="center"),  # buttons
        html.Br(),
        dbc.Row([
            dbc.Stack([
                dbc.Stack([
                    dbc.Label('Polynomial degree:', html_for='drop-degree'),
                    # html.Label('Polynomial degree:'),
                    dcc.Dropdown(id='drop-degree', options=[1, 2, 3, 4, 5, 6, 7],
                                 value=1, clearable=False),
                ], gap=1, direction='horizontal'),
                dbc.Stack([
                    dbc.Label('On point click:', html_for='click_policy'),
                    # html.Label('On point click:'),
                    dcc.RadioItems(
                        id='click_policy',
                        options=[
                            {'label': 'delete', 'value': 'delete'},
                            {'label': 'duplicate', 'value': 'duplicate'},
                        ],
                        value='delete',
                    ),
                ], gap=1, direction='horizontal'),
                dbc.Stack([
                    dbc.Label('std:', html_for='label-std'),
                    # html.Label('std:'),
                    dbc.Label('', id='label-std',
                              # html.Label('', id='label-std',
                              style={
                                  'border': '1px solid #ccc',  # Add a border to make it look like a box
                                  'borderRadius': '5px',  # Rounded corners for a more input-like appearance
                                  'padding': '5px 10px',  # Space inside the box
                                  'fontWeight': 'bold',  # Emphasize the text
                                  'height': '2em',
                                  'width': '7em',  # Set a fixed width if needed
                              }
                              ),
                ], gap=1, direction='horizontal')
            ], gap=5, direction='horizontal'),
        ], justify="start", align="center"),  # tools
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='graph-cc',
                    figure=px.scatter(),
                    config={'displaylogo': False},
                ),
            ], md=7, sm=12),  # Curve
            dbc.Col([
                dash_table.DataTable(
                    id='var_table',
                    columns=[
                        # {'name': 'name', 'id': 'name', 'type': 'text'},
                        {'name': 'jd', 'id': 'jd', 'type': 'numeric'},
                        {'name': 'mag_std', 'id': 'mag_std', 'type': 'numeric',
                         'format': dash_table.Format.Format(precision=3,
                                                            scheme=dash_table.Format.Scheme.fixed)},
                        {'name': 'measurement', 'id': 'measurement', 'type': 'numeric',
                         'format': dash_table.Format.Format(precision=2,
                                                            scheme=dash_table.Format.Scheme.fixed)},
                        {'name': 'distance', 'id': 'distance', 'type': 'numeric',
                         'format': dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.fixed)}
                    ],
                    #     page_size=100,
                    data=[],
                    fixed_rows={'headers': True},  # Freeze the header
                    style_table={
                        # 'maxHeight': '40em',
                        'height': '100%',
                        'overflowY': 'auto',  # vertical scrolling
                        'overflowX': 'auto',  # horizontal scrolling
                    },
                    page_action="native", sort_action="native",
                    style_cell={"font-size": 14, 'textAlign': 'left'},
                    cell_selectable=True,
                    style_header={"font-size": 14, 'font-family': 'courier',

                                  'fontWeight': 'bold',
                                  'color': '#000',
                                  'backgroundColor': 'var(--bs-light)',
                                  'textAlign': 'left'},
                )

            ], md=5, sm=12)  # table
        ]),  # graph and res table
        dbc.Row([
            dash_table.DataTable(
                id="measurements_сс_table",
                row_selectable="single",
                selected_rows=[],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "center"},
                style_header={"fontWeight": "bold"}
            ),
        ]),

        dcc.Store(id='store-fit'),
        dcc.Store(id='store_char_curve'),  # current (on jd) characteristic curve std_mag -- inst_mag
        dcc.Store(id='store_var'),  # jd inst_mag std_mag
        dcc.Store(id='store_current_var'),  # last var measurements inst_mag -- std_mag
        dcc.Store(id='store_standards'),  # star_name --> star_mag
    ], className="g-0", fluid=False)


# def parse_curve(contents):
#     content_type, content_string = contents.split(',')
#     decoded = base64.b64decode(content_string)
#     data = io.StringIO(decoded.decode('utf-8'))
#     df = pd.read_csv(data, sep='\s+', skiprows=1, names=["name", "mag_std", "mag_inst"])
#     return df


# def parse_meas(contents):
#     content_type, content_string = contents.split(',')
#     decoded = base64.b64decode(content_string)
#     df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), comment='#', sep='\s+', header=None,
#                      names=['name', 'mag_std', 'mag_inst'])
#     return df


@callback(
    Output("measurements_сс_table", "data"),
    Output("measurements_сс_table", "columns"),
    Output("measurements_сс_table", "selected_rows"),
    Output("dropdown-column", "options"),
    Output("dropdown-column", "value"),
    Input("upload_inst", "contents"),
    prevent_initial_call=True
)
def load_measurements_table(contents):
    if not contents:
        raise PreventUpdate
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(
        io.StringIO(decoded.decode('utf-8')),
        sep='\s+',
        na_values="-",
        dtype=float)
    columns = [{"name": col, "id": col} for col in df.columns]
    options = [{"label": col, "value": col} for col in df.columns if col != "jd"]
    return df.to_dict("records"), columns, [], options, df.columns[1]


@callback(Output('download-row', 'data'),
          Input('btn-download-row', 'n_clicks'),
          State('store_char_curve', 'data'),
          State('store-fit', 'data'),
          State('measurements_сс_table', 'data'),
          State('measurements_сс_table', "selected_rows"),
          prevent_initial_call=True)
def download_row(_, json_cc, json_fit, table_data, selected_row):
    if not selected_row or not json_cc:
        raise PreventUpdate
    row_idx = selected_row[0]
    df = pd.DataFrame(table_data)
    measurements = df.iloc[row_idx]
    jd_value = measurements.get('jd', 'unknown')
    output_file_name = f"results_{jd_value}.dat"

    valid_measurements = measurements.dropna()
    valid_measurements = valid_measurements[valid_measurements.index != "jd"]

    dff = pd.read_json(json_fit, orient='split')
    df_curve = pd.read_json(json_cc, orient='split')
    fit = np.array(dff[0])

    results = []
    for name, measurement in valid_measurements.items():
        mag_std = np.poly1d(fit)(measurement)
        min_mag = df_curve['measurement'].min()
        max_mag = df_curve['measurement'].max()
        distance = max(min_mag - measurement, 0) if measurement < min_mag else max(measurement - max_mag, 0)
        results.append({"name": name, "measurement": measurement, "mag_std": mag_std, 'distance': distance})
    results_df = pd.DataFrame(results)
    head_list = list(results_df.columns)
    head_list[0] = '#' + head_list[0]
    return dcc.send_data_frame(
            results_df.to_csv,
            output_file_name,
            header=head_list,
            sep=' ',
            float_format='%.3f',
            index=False
        )


@callback(Output('store_var', 'data'),
          Output('store_current_var', 'data'),
          Input('btn_calc_var', 'n_clicks'),
          State('dropdown-column', 'value'),
          State('measurements_сс_table', 'data'),
          State('measurements_сс_table', "selected_rows"),
          State('store_char_curve', 'data'),
          State('store_var', 'data'),
          State('store-fit', 'data'),
          prevent_initial_call=True)
def ffff(_, selected_column, table_data, selected_row, json_cc, json_var, json_fit):
    if not selected_column or not selected_row or not json_cc:
        raise PreventUpdate
    row_idx = selected_row[0]
    df = pd.DataFrame(table_data)
    jd = df.iloc[row_idx]['jd']
    measurement = df.iloc[row_idx][selected_column]
    distance = 0.0
    mag_std = -99
    if json_fit is not None:
        dff = pd.read_json(json_fit, orient='split')
        df_curve = pd.read_json(json_cc, orient='split')
        fit = np.array(dff[0])
        mag_std = np.poly1d(fit)(measurement)
        min_mag = df_curve['measurement'].min()
        max_mag = df_curve['measurement'].max()
        distance = max(min_mag - measurement, 0) if measurement < min_mag else max(measurement - max_mag, 0)

    if json_var is not None:
        dfm = pd.read_json(json_var, orient='split')
        new_row = pd.DataFrame({'jd': [jd], 'mag_std': [mag_std], 'measurement': [measurement], 'distance': [distance]})
        dfm = pd.concat([dfm, new_row], ignore_index=True)
    else:
        dfm = pd.DataFrame({'jd': [jd], 'mag_std': [mag_std], 'measurement': [measurement], 'distance': [distance]})

    return dfm.to_json(orient='split'), {'measurement': measurement, 'mag_std': mag_std}


@callback(
    Output("store_standards", "data"),
    Input("upload-standards", "contents"),
)
def upload_standards(contents):
    if not contents:
        raise PreventUpdate
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep='\s+')
        di = df.to_dict(orient="records")
    except Exception as e:
        logging.warning(f'upload_standards {e}')
        raise PreventUpdate

    return di


def prepare_curve(table_data, standards_data, selected_rows):
    row_idx = selected_rows[0]
    measurements_df = pd.DataFrame(table_data)
    row_data = measurements_df.iloc[row_idx]

    standards_df = pd.DataFrame(standards_data)

    stars = [col for col in measurements_df.columns if col != "jd"]
    # measured_mags = row_data[stars].values
    name_col = standards_df.columns[0]
    mag_col = standards_df.columns[1]

    valid_stars = [star for star in stars if star in standards_df[name_col].values]
    # valid_stars = [star for star in stars if star in standards_df[name_col].astype(str).values]

    measured_mags_filtered = [row_data[star] for star in valid_stars]
    std_mags = standards_df.set_index(name_col).reindex(valid_stars)[mag_col].values

    return std_mags, measured_mags_filtered, valid_stars


def prepare_curve_old(table_data, standards_data, selected_rows):
    row_idx = selected_rows[0]
    measurements_df = pd.DataFrame(table_data)
    row_data = measurements_df.iloc[row_idx]

    standards_df = pd.DataFrame(standards_data)

    stars = [col for col in measurements_df.columns if col != "jd"]
    measured_mags = row_data[stars].values
    name_col = standards_df.columns[0]
    mag_col = standards_df.columns[1]
    std_mags = standards_df.set_index(name_col).reindex(stars)[mag_col].values
    return std_mags, measured_mags


@callback(Output('store_char_curve', 'data', allow_duplicate=True),
          Input('graph-cc', 'clickData'),  # edit curve
          State('click_policy', 'value'),
          State('store_char_curve', 'data'),
          prevent_initial_call=True)
def update_clicked_curve(click_data, on_click, jsonified_curve):
    if click_data['points'][0]['curveNumber'] != 0:
        raise PreventUpdate
    df = pd.read_json(jsonified_curve, orient='split')
    if on_click == 'delete':
        df.drop(index=click_data['points'][0]['pointIndex'], inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:  # Duplicate the clicked point
        point_index = click_data['points'][0]['pointIndex']
        new_row = df.loc[point_index].copy()
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    return df.to_json(orient='split')


@callback(Output('store-fit', 'data'),
          Output('label-std', 'children'),
          Input('store_char_curve', 'data'),
          Input('drop-degree', 'value'),
          prevent_initial_call=True)
def fit_poly(jsonified_curve, degree):
    if jsonified_curve is None:
        raise PreventUpdate
    if degree is None:
        raise PreventUpdate
    df = pd.read_json(jsonified_curve, orient='split')
    fit, residuals, _, _, _ = np.polyfit(df['measurement'], df['mag_std'], int(degree), full=True)
    std_residuals = np.sqrt(residuals[0] / (len(df) - 1))
    dff = pd.DataFrame(fit)
    return dff.to_json(orient='split'), f'{std_residuals:.3f}'


@callback(Output('var_table', 'data'),
          Input('store_var', 'data'),
          prevent_initial_call=True)
def draw_table(jsonified_meas):
    print('draw_table')
    if jsonified_meas is not None:
        df = pd.read_json(jsonified_meas, orient='split')
        data = df.to_dict('records')
    else:
        raise PreventUpdate
    return data


@callback(Output('store_char_curve', 'data'),
          Output('store_current_var', 'data', allow_duplicate=True),
          Input("measurements_сс_table", "selected_rows"),
          State("measurements_сс_table", "data"),
          State("store_standards", "data"),
          prevent_initial_call=True)
def store_char_curve(selected_rows, table_data, standards_data):
    if not selected_rows or not table_data or not standards_data:
        raise PreventUpdate

    std_mags, measured_mags, names = prepare_curve(table_data, standards_data, selected_rows)
    df = pd.DataFrame({
        'name': names,
        'mag_std': std_mags,
        'measurement': measured_mags,
    }).dropna()
    return df.to_json(orient='split'), None


@callback(Output('graph-cc', 'figure'),
          Input('store-fit', 'data'),
          Input('store_char_curve', 'data'),
          Input('store_current_var', 'data'),
          prevent_initial_call=False)
def plot_curve(jsonified_fit, jsonified_curve, json_var):
    if jsonified_curve is None:
        fig = px.scatter()
    else:
        df = pd.read_json(jsonified_curve, orient='split')
        fig = px.scatter(df, x='measurement', y='mag_std')  # , trendline='ols')
        if jsonified_fit is not None:
            dff = pd.read_json(jsonified_fit, orient='split')
            fit = np.array(dff[0])
            x_fit = np.linspace(df['measurement'].min(), df['measurement'].max(), 50)
            fig.add_scatter(x=x_fit, y=np.poly1d(fit)(x_fit), name='fit', mode='lines',
                            hoverinfo='skip', line=dict(color='Green'), showlegend=True)
        if json_var is not None:
            # dfm = pd.read_json(json_var, orient='split')
            xm = [json_var['measurement']]
            ym = [json_var['mag_std']]
            fig.add_scatter(x=xm, y=ym, name='vars', mode='markers',
                            marker=dict(size=10, color='Orange'),
                            showlegend=True)
    fig.update_layout({'paper_bgcolor': app_color['graph_bg'],
                       'plot_bgcolor': app_color['graph_bg'],
                       'yaxis': dict(autorange='reversed'),
                       'margin': dict(t=app_margins['top'], b=app_margins['bottom'],
                                      l=app_margins['left'], r=app_margins['right']),
                       'font': dict(size=15)})
    print('Figure is ready')
    return fig


@callback(
    Output('download-var', 'data'),
    Input('btn-download-var', 'n_clicks'),
    State('store_var', 'data'),
    prevent_initial_call=True
)
def download_meas(_, jsonified_meas):
    # return dcc.send_data_frame(df.to_csv, 'my.txt')
    if jsonified_meas is None:
        raise PreventUpdate
    dfm = pd.read_json(jsonified_meas, orient='split')
    head_list = list(dfm.columns)
    head_list[0] = '#' + head_list[0]
    return dcc.send_data_frame(dfm.to_csv, "results.dat", header=head_list, sep=' ', float_format='%.3f',
                               index=False)

# if __name__ == '__main__':
#     app.run(debug=True)
