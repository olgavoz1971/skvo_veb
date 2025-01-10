import logging

import numpy as np
import plotly.graph_objects as go
from dash import register_page, dcc, html, callback
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable
from dash.dependencies import Input, Output, State
import plotly.express as px

import lightkurve as lk
from dash.exceptions import PreventUpdate
from lightkurve import LightkurveError

# from lightkurve.search import SearchResult
from skvo_veb.utils import tess_cache as cache

# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

switch_label_style = {'display': 'block', 'padding': '2px'}  # In the row, otherwise 'block'

register_page(__name__, name='TESS curve',
              order=4,
              path='/igebc/tess_lc',
              title='TESS lightcurve Tool',
              in_navbar=True)

stack_wrap_style = {'marginBottom': '5px', 'flexWrap': 'wrap'}


def layout():
    return dbc.Container([
        html.H1('TESS Lightcurve Tool', className="text-primary text-left fs-3"),
        dbc.Tabs([
            dbc.Tab(label='Search', children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Stack([
                            dbc.Label('Object', html_for='obj_name_tess_lc_input', style={'width': '7em'}),
                            dcc.Input(id='obj_name_tess_lc_input', persistence=True, type='text',
                                      style={'width': '100%'}),  # , 'border-radius': '5px'}),
                        ], direction='horizontal', gap=2, style={'marginBottom': '5px'}),
                        dbc.Stack([
                            dbc.Button('Search', id='basic_search_tess_lc_button', size="sm"),
                            dbc.Button('Cancel', id='cancel_basic_search_tess_lc_button',
                                       size="sm", disabled=True),
                        ], direction='horizontal', gap=2, style=stack_wrap_style),
                    ], md=2, sm=4, xs=12, style={'padding': '10px', 'background': 'Silver', 'border-radius': '5px'}),
                    # Search tools
                    dbc.Col([
                        dbc.Spinner(
                            children=[
                                dbc.Row([
                                    dbc.Col([
                                        html.H3("Search results", id="table_tess_lc_header"),
                                    ], md=6, sm=12),
                                    dbc.Col([
                                        dbc.Stack([
                                            dbc.Button('Download curves', id='download_tess_lc_button', size="sm"),
                                            dbc.Button('Cancel', id='cancel_download_tess_lc_button',
                                                       size="sm", disabled=True),
                                        ], direction='horizontal', gap=2, style={'marginBottom': '5px'}),
                                        # style={'width': '100%'}),
                                    ], md=6, sm=12),
                                ], style={'marginBottom': '10px', 'marginTop': '10px'}),
                                dbc.Row([
                                    DataTable(
                                        id="data_tess_lc_table",
                                        columns=[{"name": col, "id": col} for col in
                                                 ["#", "mission", "year", "author", "exptime", "target"]],
                                        data=[],
                                        row_selectable="multi",
                                        fixed_rows={'headers': True},  # Freeze the header
                                        style_table={
                                            'maxHeight': '50vh',
                                            'overflowY': 'auto',  # vertical scrolling
                                            'overflowX': 'auto',  # horizontal scrolling
                                        },
                                        page_action="native", sort_action="native",
                                        style_cell={"font-size": 14, 'textAlign': 'left'},
                                        cell_selectable=False,
                                        style_header={"font-size": 14, 'font-family': 'courier',
                                                      'color': '#000',
                                                      'backgroundColor': 'var(--bs-light)',
                                                      'textAlign': 'left'},
                                    )
                                ]),
                            ]
                        ),
                    ], md=10, sm=8, xs=12, id="table_tess_lc_row", style={"display": "none"}),  # SearchResults Table
                ], style={'marginBottom': '10px'}),  # Search and SearchResults
                dbc.Spinner(
                    dbc.Label(id="download_tess_lc_result", children='',
                              style={"color": "green", "text-align": "center"}),
                    spinner_style={
                        "align-items": "center",
                        "justify-content": "center",
                    }, color="primary",
                ),
            ], tab_id='tess_lc_search_tab'),
            dbc.Tab(label='Plot', children=[
                dbc.Row([
                    # html.H1('TESS GUI', className="text-primary text-left fs-3"),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label('Flux:', html_for='flux_tess_lc_switch'),
                                dcc.RadioItems(
                                    id='flux_tess_lc_switch',
                                    options=[
                                        {'label': 'pdc_sap', 'value': 'pdcsap'},
                                        {'label': 'sap', 'value': 'sap'},
                                        {'label': 'default', 'value': 'default'},
                                    ],
                                    value='pdcsap',
                                    labelStyle=switch_label_style,
                                ),
                            ], md=6, sm=6),
                            dbc.Col([
                                dbc.Label('Curve:', html_for='stitch_switch'),
                                dbc.Switch(id='stitch_switch', label='Stitch', value=False,
                                           # label_style=switch_label_style,
                                           style=switch_label_style,
                                           persistence=True),
                                # dbc.Checklist(options=[{'label': 'Stitch', 'value': 1}], value=0, id='stitch_switch',
                                #               persistence=True, switch=True,
                                #               # labelStyle={'flexWrap': 'wrap'},
                                #               labelStyle=switch_label_style,
                                #               ),
                            ], md=6, sm=6),
                        ]),  # tune
                        dbc.Row([

                            dbc.Col(dbc.Button('Plot', id='plot_selected_tess_lc_button', size="sm",
                                               style={'width': '100%'}), width=6),
                        ], style={'marginBottom': '5px'}),
                    ], md=2, sm=12, style={'padding': '10px', 'background': 'Silver', 'border-radius': '5px'}),  # Tools
                    dbc.Col([
                        dcc.Graph(id='graph_tess_lc',
                                  figure=go.Figure().update_layout(
                                      title='',
                                      showlegend=False,
                                      margin=dict(l=0, b=20, t=30, r=20),
                                      xaxis_title=f'time',
                                      yaxis_title=f'flux'
                                  ),
                                  config={'displaylogo': False},
                                  # style={'height': '70vh'},  # 100% of the viewport height
                                  style={'height': '40vh', 'width': '100%'},  # 100% of the viewport height
                                  # style={'height': '100%'}
                                  ),
                    ], md=10, sm=12),  # style={'padding': '2px', 'background': 'blue'})    # Graph
                ],
                    style={'marginBottom': '10px'}
                ),
            ], tab_id='tess_lc_tool', id='tess_lc_graph_tab', disabled=False),
        ], active_tab='tess_lc_search_tab', id='tess_lc_tabs', style={'marginBottom': '5px'}),
        # dcc.Store(id='tess_lc_collection_store'),
    ], className="g-10", fluid=True, style={'display': 'flex', 'flexDirection': 'column'
                                            })


@callback(
    [Output("table_tess_lc_header", "children"),
     Output("data_tess_lc_table", "data"),
     Output("data_tess_lc_table", "selected_rows"),
     Output("table_tess_lc_row", "style")],  # to show the table and Title
    [Input('basic_search_tess_lc_button', 'n_clicks'),
     State('obj_name_tess_lc_input', 'value')],
    running=[(Output('basic_search_tess_lc_button', 'disabled'), True, False),
             (Output('cancel_basic_search_tess_lc_button', 'disabled'), False, True)],
    cancel=[Input('cancel_basic_search_tess_lc_button', 'n_clicks')],
    background=True,
    prevent_initial_call=True
)
def basic_search(n_clicks, obj_name):
    if n_clicks is None:
        raise PreventUpdate
    search_lcf = cache.load("basic_search", obj_name=obj_name)
    if search_lcf is None:
        search_lcf = lk.search_lightcurve(obj_name)
        cache.save(search_lcf, "basic_search", obj_name=obj_name)
    repr(search_lcf)  # Do not touch this line :-)

    data = []
    for row in search_lcf.table:
        data.append({
            '#': row['#'],
            'mission': row['mission'],
            'year': row['year'],
            'target': row["target_name"],
            "author": row["author"],
            "exptime": row["exptime"]
        })

    return f'Basic search  for {obj_name}', data, [], {"display": "block"}


# todo 2. Add error message in case of failed download


def plot_selected_curves(selected_rows, table_data, stitch, flux_method):
    import re
    if not selected_rows or not table_data:
        raise PreventUpdate
    selected_data = [table_data[i] for i in selected_rows]

    search_cache_key = "search_lcf_refined"
    lc_list = []
    authors = []
    sectors = []
    pdc_methods = []
    flux_origins = []
    for row in selected_data:
        target = f'TIC {row.get("target", None)}'
        author = row["author"]
        exptime = row["exptime"]
        match = re.search(r'Sector (\d+)', row.get('mission', ''))
        if match:
            sector = int(match.group(1))
        else:
            sector = -1
        args = {
            'target': target,
            'author': author,
            'mission': 'TESS',
            'sector': sector,
            'exptime': exptime
        }
        search_lcf_refined = cache.load(search_cache_key, **args)
        if search_lcf_refined is None:
            search_lcf_refined = lk.search_lightcurve(**args)
            if len(search_lcf_refined) > 0:
                cache.save(search_lcf_refined, search_cache_key, **args)
        repr(search_lcf_refined)  # Leave it to fill '#' column
        try:
            lc = search_lcf_refined.download()  # LightKurve uses its own cache
        except LightkurveError as e:
            logging.warning(f'download_selected_pixel exception: {e}')
            # Probably, we have the corrupted cache. Let's try clean it
            # Build the filename of cached lightcurve. See lightkurve/search.py
            # I don't want to change the default cache_dir:
            import os
            # noinspection PyProtectedMember
            download_dir = search_lcf_refined._default_download_dir()
            table = search_lcf_refined.table
            path = os.path.join(
                download_dir.rstrip("/"),
                "mastDownload",
                table["obs_collection"][0],
                table["obs_id"][0],
                table["productFilename"][0],
            )
            # Remove and retry
            logging.warning(f'Removing corrupted cache: {path}')
            os.remove(path) if os.path.isfile(path) else None
            lc = search_lcf_refined.download()
        if flux_method == 'pdcsap' and 'pdcsap_flux' in lc.columns:
            lc.flux = lc.pdcsap_flux
            flux_origin = flux_method
        elif flux_method == 'sap' and 'sap_flux' in lc.columns:
            lc.flux = lc.sap_flux
            flux_origin = flux_method
        else:
            flux_origin = lc.FLUX_ORIGIN
            pass

        sectors.append(str(lc.SECTOR))
        authors.append(lc.AUTHOR)
        flux_origins.append(flux_origin)
        pdc_method = 'NO'
        try:
            logging.debug(f'{lc.meta["PDCMETHD"]=}')
            if 'pdc' in flux_origin:
                pdc_method = lc.meta["PDCMETHD"]
        except Exception as e:
            logging.warning(e)
        pdc_methods.append(pdc_method)

        lc_list.append(lc)

    if stitch:
        lkk = lk.LightCurveCollection(lc_list)
        lc_res = lkk.stitch()
        jd = lc_res.time.value
        flux = lc_res.flux
        if hasattr(flux, 'mask'):
            valid_jd = jd[~flux.mask]
            valid_flux = flux[~flux.mask]
        else:
            valid_jd = jd
            valid_flux = flux
    else:
        valid_jd = np.array([], dtype=float)
        valid_flux = np.array([], dtype=float)
        for lc in lc_list:
            flux = lc.flux.value
            jd = lc.time.value
            if hasattr(flux, 'mask'):
                valid_jd = np.append(valid_jd, jd[~flux.mask])
                valid_flux = np.append(valid_flux, flux[~flux.mask])
            else:
                valid_jd = np.append(valid_jd, jd)
                valid_flux = np.append(valid_flux, flux)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=valid_jd, y=valid_flux,
        hoverinfo='none',  # Important
        hovertemplate=None,
        mode='markers+lines',
        marker=dict(color='blue', size=6, symbol='circle'),
        line=dict(color='blue', width=1)  # , dash='dash')
    ))

    # title = f'{lc_list[0].LABEL} sector: {",".join(sectors)} author: {",".join(authors)} {",".join(flux_origin)}'
    title = f'{lc_list[0].LABEL} {", ".join(flux_origins)} {", ".join(pdc_methods)}'
    time_unit = lc_list[0].time.format
    if stitch:
        title = 'Stitched curve ' + title
        time_unit = 'some time unit'
        flux_unit = 'relative flux'
    else:
        flux_unit = str(lc_list[0].flux.unit)
    fig.update_layout(
        title=title,
        showlegend=False,
        margin=dict(l=0, b=20, t=30, r=20),
        xaxis_title=f'time, {time_unit}',
        yaxis_title=f'flux, {flux_unit}'
    )

    return fig


@callback(Output('graph_tess_lc', 'figure', allow_duplicate=True),
          [Input('plot_selected_tess_lc_button', 'n_clicks'),
           State('data_tess_lc_table', 'selected_rows'),
           State('data_tess_lc_table', 'data'),
           State('stitch_switch', 'value'),
           State('flux_tess_lc_switch', 'value')],
          prevent_initial_call=True
          )
def replot_selected_curves(n_clicks, selected_rows, table_data, stitch, flux_method):
    if n_clicks is None:
        raise PreventUpdate
    fig = plot_selected_curves(selected_rows, table_data, stitch, flux_method)
    return fig


# if __name__ == '__main__':
#     app.run_server(debug=True, port=8051)


@callback([Output('graph_tess_lc', 'figure', allow_duplicate=True),
           Output('download_tess_lc_result', 'children'),
           Output('tess_lc_graph_tab', 'disabled'),
           Output('tess_lc_tabs', 'active_tab')],
          [Input('download_tess_lc_button', 'n_clicks'),
           State('data_tess_lc_table', 'selected_rows'),
           State('data_tess_lc_table', 'data'),
           State('stitch_switch', 'value'),
           State('flux_tess_lc_switch', 'value')],
          background=True,
          running=[(Output('download_tess_lc_button', 'disabled'), True, False),
                   (Output('cancel_download_tess_lc_button', 'disabled'), False, True)],
          cancel=[Input('cancel_download_tess_lc_button', 'n_clicks')],
          prevent_initial_call=True)
def download_tess_curve(n_clicks, selected_rows, table_data, stitch, flux_method):
    """
    This method checks for the presence of light curves in the local cache.
    If any are missing, it downloads the absent light curves from the remote database.
    Note: unlike other methods handling TESS lightcurves, this one is specifically designed to accommodate long
    waiting times. It includes user feedback mechanisms, such as a spinner, and robust error handling for server
    connectivity issues.
    In contrast, other methods rely on the local cache to ensure faster response times.
    """

    if n_clicks is None:
        raise PreventUpdate

    fig = plot_selected_curves(selected_rows, table_data, stitch, flux_method)
    return fig, 'tst success', False, 'tess_lc_tool'
