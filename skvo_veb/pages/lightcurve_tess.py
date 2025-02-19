import logging

import lightkurve
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import register_page, dcc, html, callback, ctx, dash, set_props, clientside_callback
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable
from dash.dependencies import Input, Output, State, ClientsideFunction
import plotly.express as px

import lightkurve as lk
from dash.exceptions import PreventUpdate
from lightkurve import LightkurveError
from skvo_veb.components import message
from skvo_veb.utils import tess_cache as cache
from skvo_veb.utils.curve_dash import CurveDash
from skvo_veb.utils.my_tools import safe_none, PipeException

# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

register_page(__name__, name='TESS curve',
              order=4,
              path='/igebc/tess_lc',
              title='TESS lightcurve Tool',
              in_navbar=True)

label_font_size = '0.8em'
switch_label_style = {'display': 'inline-block', 'padding': '2px', 'font-size': label_font_size}
stack_wrap_style = {'marginBottom': '5px', 'flexWrap': 'wrap'}
periodogram_param_style = {'width': '4em'}
periodogram_result_style = {'width': '3em', 'fontWeight': 'bold', 'font-size': label_font_size}

jd0_tess = 2457000  # btjd format. We can use the construction Time(2000, format="btjd", scale="tbd") directly,


# but this "btjd" is not included in the original astropy.time module and appear after including lightkurve only.
# So I decided it would be safer to add this constant explicitly


def layout():
    fig_lc = px.scatter()
    fig_lc.update_traces(
        selected={'marker': {'color': 'orange', 'size': 5}},
        hoverinfo='none',  # Important
        hovertemplate=None,  # Important
    )
    fig_lc.update_layout(xaxis={'title': 'phase', 'tickformat': '.1f'},
                         yaxis_title='flux',
                         # showlegend=True,
                         margin=dict(l=0, b=20),  # r=50, t=50, b=20))
                         # dragmode='lasso'  # Enable lasso selection mode by default
                         )
    fig_pg = px.scatter()  # Periodogram
    return dbc.Container([
        html.H1('TESS Lightcurve Tool', className="text-primary text-left fs-3"),
        dbc.Tabs([
            dbc.Tab(label='Search', children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Stack([
                            dbc.Label('Object', html_for='obj_name_tess_lc_input',
                                      style={'width': '7em'}),
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
                        dbc.Spinner(children=[
                            html.Div([
                                dbc.Row([
                                    dbc.Col([
                                        html.H3("Search results", id="table_tess_lc_header"),
                                    ], md=6, sm=12),
                                    dbc.Col([
                                        dbc.Stack([
                                            dbc.Button('Download curves', id='download_tess_lc_button', size="sm"),
                                            dbc.Button('Cancel', id='cancel_download_tess_lc_button',
                                                       size="sm", disabled=True),
                                        ], direction='horizontal', gap=2, style=stack_wrap_style),
                                        # style={'marginBottom': '5px'}),
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
                            ], id="table_tess_lc_row", style={"display": "none"}),  # Search results
                            html.Div(id='div_tess_lc_search_alert', style={"display": "none"}),  # Alert
                        ]),
                    ], md=10, sm=8, xs=12),  # SearchResults Table is here
                ], style={'marginBottom': '10px'}),  # Search and SearchResults
                dbc.Spinner(children=[
                    dbc.Label(id="download_tess_lc_result", children='',
                              style={"color": "green", "text-align": "center"}),
                    html.Div(id='div_tess_lc_download_alert', style={"display": "none"}),  # Alert
                ], spinner_style={
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
                                dbc.Label('Flux:', html_for='flux_tess_lc_switch',
                                          style={'width': '7em', 'font-size': label_font_size}),
                                dcc.RadioItems(
                                    id='flux_tess_lc_switch',
                                    options=[
                                        {'label': 'pdc_sap', 'value': 'pdcsap'},
                                        {'label': 'sap', 'value': 'sap'},
                                        {'label': 'default', 'value': 'default'},
                                    ],
                                    value='pdcsap',
                                    labelStyle=switch_label_style,
                                ),  # flux type radio
                            ], md=6, sm=6),
                            dbc.Col([
                                dbc.Row(dbc.Label('Curve:', html_for='stitch_switch',
                                                  style={'width': '7em', 'font-size': label_font_size})),
                                dbc.Row(
                                    dbc.Switch(id='stitch_switch', label='Stitch', value=False,
                                               label_style=switch_label_style,
                                               # style=switch_label_style,
                                               persistence=True),
                                ),
                            ], md=6, sm=6),
                        ]),  # tune
                        dbc.Row([
                            dbc.Col([
                                dbc.Label('Period:', html_for='period_tess_lc_input',
                                          style={'width': '4em', 'font-size': label_font_size}),
                            ], width="auto"),
                            dbc.Col([
                                dbc.InputGroup([
                                    dcc.Input(id='period_tess_lc_input', inputMode='numeric', persistence=False,
                                              value=None, type='number', step=0.00001,
                                              # style={'width': '100%', 'min-width': '5ch'}),
                                              style={'width': '4em'}),  # 'font-size': label_font_size}
                                    dbc.Button('x', size='sm', color='light', id='clear_period_btn')
                                ]),
                            ], width="auto"),

                        ], style={'marginBottom': '5px', 'padding': '2px', 'alignItems': 'center'}),  # Period
                        dbc.Row([
                            dbc.Switch(id='fold_tess_lc_switch', label='Folded view', value=False,
                                       label_style=switch_label_style,
                                       # style=switch_label_style,
                                       persistence=False),
                        ], style={'marginBottom': '5px', 'marginLeft': '0px'}),  # switch Folded View
                        dbc.Row([
                            dbc.Col([
                                dbc.Button('Recreate Curves', id='recreate_selected_tess_lc_button', size="sm",
                                           style={'width': '100%'}),
                            ], width=6),  # plot button
                            dbc.Col([
                                dbc.Button('Recalc Phase', id='recalc_phase_tess_lc_button', size="sm",
                                           style={'width': '100%'}),
                            ], width=6),  # fold/unfold switch
                        ], style={'marginBottom': '5px'}, className='g-2'),  # two buttons
                        dbc.Row([
                            dbc.Stack([
                                dbc.Select(options=CurveDash.get_format_list(),
                                           value=CurveDash.get_format_list()[0],
                                           id='select_tess_lc_format',
                                           style={'max-width': '7em', 'font-size': label_font_size}),
                                dbc.Button('Download', id='btn_download_tess_lc', size="sm"),
                            ], direction='horizontal', gap=2, style=stack_wrap_style),
                        ],  # justify='between',
                            # className='gy-1',  # class adds vertical gaps between folded columns
                            style={'marginBottom': '5px', 'marginTop': '5px'}),  # download curve
                        html.Details([
                            html.Summary('Periodogram'),
                            dbc.Row([
                                dcc.RadioItems(
                                    id='period_freq_tess_lc_switch',
                                    options=[
                                        {'label': 'Period', 'value': 'period'},
                                        {'label': 'Freq', 'value': 'frequency'},
                                    ],
                                    value='period',
                                    persistence=True,
                                    labelStyle={'display': 'row', 'padding': '4px', 'font-size': label_font_size},
                                ),  # Period / frequency switch
                                dcc.RadioItems(
                                    id='method_tess_lc_switch',
                                    options=[
                                        {'label': ' Lomb-Scargle', 'value': 'ls'},
                                        {'label': 'BLS', 'value': 'bls'},
                                    ],
                                    value='ls',
                                    persistence=True,
                                    labelStyle={'display': 'row', 'padding': '4px', 'font-size': label_font_size},
                                ),  # Period / frequency switch
                                dbc.Stack([
                                    dbc.Label('Oversample:', html_for='periodogram_oversample',
                                              style={'width': '7em', 'font-size': label_font_size}),
                                    dcc.Input(id='periodogram_oversample',
                                              value=1, inputMode='numeric',
                                              type='number',
                                              # style={'width': '100%', 'font-size': label_font_size}),
                                              style=periodogram_param_style),
                                ], direction='horizontal', gap=2, style=stack_wrap_style),  # periodogram_oversample
                                dbc.Stack([
                                    dbc.Label('Period min:', html_for='periodogram_min',
                                              style={'width': '7em', 'font-size': label_font_size}),
                                    dcc.Input(id='periodogram_min', min=0,
                                              value=None, inputMode='numeric', type='number',
                                              # style={'width': '100%', 'font-size': label_font_size}),
                                              style=periodogram_param_style),
                                ], direction='horizontal', gap=2, style=stack_wrap_style),
                                dbc.Stack([
                                    dbc.Label('Period max:', html_for='periodogram_max',
                                              style={'width': '7em', 'font-size': label_font_size}),
                                    dcc.Input(id='periodogram_max', min=0,
                                              value=None, inputMode='numeric', type='number',
                                              # style={'width': '100%', 'font-size': label_font_size}),
                                              style=periodogram_param_style),
                                ], direction='horizontal', gap=2, style=stack_wrap_style),
                                dbc.Stack([
                                    dbc.Label('N terms:', html_for='Duration:',
                                              style={'width': '7em', 'font-size': label_font_size}),
                                    dcc.Input(id='periodogram_nterms', value=1, min=1,
                                              inputMode='numeric', type='number',
                                              # style={'width': '100%', 'font-size': label_font_size}),
                                              style=periodogram_param_style),
                                ], direction='horizontal', gap=2, style=stack_wrap_style),
                                dbc.Stack([
                                    dbc.Label('Duration:', html_for='periodogram_duration',
                                              style={'width': '7em', 'font-size': label_font_size}),
                                    dcc.Input(id='periodogram_duration', value=None, min=0,
                                              inputMode='numeric', type='number',
                                              # style={'width': '100%', 'font-size': label_font_size}),
                                              style=periodogram_param_style),
                                ], direction='horizontal', gap=2, style=stack_wrap_style),
                            ]),  # periodogram parameters
                            dbc.Row([
                                dbc.Col([dbc.Button('Calculate', id='periodogram_tess_lc_button', size="sm",
                                                    style={'width': '100%'})]),
                            ], style={'marginBottom': '5px'}),  # periodogram button
                            dbc.Row([
                                dbc.Stack([
                                    dbc.Label('Period:', html_for='period1_res',
                                              style={'width': '7em', 'font-size': label_font_size}),
                                    dbc.Label(id='period1_res', style=periodogram_result_style),
                                    dbc.Button('Use', id='use_period1_btn', size='sm'),
                                ], direction='horizontal', gap=2, style=stack_wrap_style),
                                dbc.Stack([
                                    dbc.Label('Period*2:', html_for='period2_res',
                                              style={'width': '7em', 'font-size': label_font_size}),
                                    dbc.Label(id='period2_res', style=periodogram_result_style),
                                    dbc.Button('Use', id='use_period2_btn', size='sm'),
                                ], direction='horizontal', gap=2, style=stack_wrap_style),
                                dbc.Stack([
                                    dbc.Label('Period*4:', html_for='period4_res',
                                              style={'width': '7em', 'font-size': label_font_size}),
                                    dbc.Label(id='period4_res', style=periodogram_result_style),
                                    dbc.Button('Use', id='use_period4_btn', size='sm'),
                                ], direction='horizontal', gap=2, style=stack_wrap_style),
                            ], id='tess_lc_periodogram_results_row', style={'display': "none"}),  # periodogram results
                        ]),
                    ], md=2, sm=12, style={'padding': '10px', 'background': 'Silver', 'border-radius': '5px'}),  # Tools
                    dbc.Col([
                        html.Div(children='', id='div_tess_lc_alert', style={'display': 'none'}),
                        dbc.Row([
                            dcc.Graph(id='graph_tess_lc',
                                      figure=fig_lc,
                                      config={'displaylogo': False},
                                      # # style={'height': '70vh'},  # 100% of the viewport height
                                      # style={'height': '40vh', 'width': '100%'},  # 100% of the viewport height
                                      # # style={'height': '100%'}
                                      ),
                        ], class_name="g-0"),  # g-0 -- Row without 'gutters'   light curve graph
                        dbc.Row([
                            dcc.Graph(
                                id='graph_tess_lc_periodogram',
                                figure=fig_pg,
                                config={'displaylogo': False}
                            )
                        ], id='tess_lc_periodogram_row', style={'display': 'none'})  # periodogram

                    ], md=10, sm=12),  # style={'padding': '2px', 'background': 'blue'})    # Graph
                ], style={'marginBottom': '10px'}),
            ], tab_id='tess_lc_graph_tab', id='tess_lc_graph_tab', disabled=False),
        ], active_tab='tess_lc_search_tab', id='tess_lc_tabs', style={'marginBottom': '5px'}),
        dcc.Store(id='store_tess_lightcurve'),
        dcc.Download(id='download_tess_lc_lightcurve'),
    ], className="g-10", fluid=True, style={'display': 'flex', 'flexDirection': 'column'})


@callback(
    # region
    output=dict(
        table_header=Output("table_tess_lc_header", "children"),
        table_data=Output("data_tess_lc_table", "data"),
        selected_rows=Output("data_tess_lc_table", "selected_rows"),
        content_style=Output("table_tess_lc_row", "style"),  # to show the table and Title
        alert_message=Output('div_tess_lc_search_alert', 'children'),
        alert_style=Output('div_tess_lc_search_alert', 'style'),
    ),
    inputs=dict(n_clicks=Input('basic_search_tess_lc_button', 'n_clicks')),
    state=dict(obj_name=State('obj_name_tess_lc_input', 'value')),
    # endregion
    running=[(Output('basic_search_tess_lc_button', 'disabled'), True, False),
             (Output('cancel_basic_search_tess_lc_button', 'disabled'), False, True)],
    cancel=[Input('cancel_basic_search_tess_lc_button', 'n_clicks')],
    background=True,
    prevent_initial_call=True
)
def basic_search(n_clicks, obj_name):
    if n_clicks is None:
        raise PreventUpdate

    output_keys = list(ctx.outputs_grouping.keys())
    output = {key: dash.no_update for key in output_keys}

    try:
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
        if data:
            output['table_header'] = f'Basic search  for {obj_name}'
            output['table_data'] = data
            output['selected_rows'] = [0] if data else []  # select the first row by default
            output['content_style'] = {'display': 'block'}  # show the table with search results
            output['alert_message'] = ''
            output['alert_style'] = {'display': 'none'}  # hide alert
        else:
            raise PipeException('No data found')
    except Exception as e:
        logging.warning(f'tess_lightcurve.search: {e}')
        output['selected_rows'] = []
        output['alert_message'] = message.warning_alert(e)
        output['alert_style'] = {'display': 'block'}  # show the alert
        output['content_style'] = {'display': 'none'}  # hide empty or wrong table

    return output


def create_lc_from_selected_rows(selected_rows, table_data, stitch, flux_method,
                                 phase_view=False, period=None) -> str:
    # return a serialized CurveDash object
    import re
    if not selected_rows or not table_data:
        raise PipeException('Search for the lightcurves first and try again')
    selected_data = [table_data[i] for i in selected_rows]

    search_cache_key = "search_lcf_refined"
    lc_list = []
    authors = []
    sectors = []
    pdc_methods = []
    flux_origins = []
    # if len(selected_rows) > 1:
    #     raise PipeException('My test exception')  # todo remove it
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
        flux = lc_res.flux.value
        flux_err = lc_res.flux_err.value
        if hasattr(flux, 'mask'):
            valid_jd = jd[~flux.mask]
            valid_flux = flux[~flux.mask]
            valid_flux_err = flux_err[~flux.mask]
        else:
            valid_jd = jd
            valid_flux = flux
            valid_flux_err = flux_err
    else:
        valid_jd = np.array([], dtype=float)
        valid_flux = np.array([], dtype=float)
        valid_flux_err = np.array([], dtype=float)
        for lc in lc_list:
            flux = lc.flux.value
            flux_err = lc.flux_err.value
            jd = lc.time.value
            if hasattr(flux, 'mask'):
                valid_jd = np.append(valid_jd, jd[~flux.mask])
                valid_flux = np.append(valid_flux, flux[~flux.mask])
                valid_flux_err = np.append(valid_flux_err, flux_err[~flux.mask])
            else:
                valid_jd = np.append(valid_jd, jd)
                valid_flux = np.append(valid_flux, flux)
                valid_flux_err = np.append(valid_flux_err, flux_err)
    valid_flux_err = np.ma.filled(valid_flux_err, 0)
    # time_unit = lc_list[0].time.format
    time_unit = 'jd'
    # Add information into lc title
    title = f'{lc_list[0].LABEL} sector: {",".join(sectors)} author: {",".join(authors)} methods: {",".join(flux_origins)}'
    if stitch:
        title = 'Stitched curve ' + title
        flux_unit = 'relative flux'
    else:
        flux_unit = str(lc_list[0].flux.unit)

    # todo: add flux_err
    lcd = CurveDash(jd=valid_jd + jd0_tess, flux=valid_flux, flux_err=valid_flux_err,
                    time_unit=time_unit, timescale='tdb',
                    flux_unit=flux_unit, title=title,
                    folded_view=phase_view,
                    period=period,
                    period_unit='d')
    return lcd.serialize()


def plot_lc(js_lightcurve: str, phase_view: bool):
    lcd = CurveDash(js_lightcurve)
    title = lcd.title
    flux_unit = lcd.flux_unit

    if phase_view:
        x = lcd.phase
        x_column = 'phase'
        xaxis_title = 'phase'
    else:
        jd0 = 2450000
        x = lcd.jd - jd0
        x_column = 'jd'
        xaxis_title = f'jd-{jd0}, {safe_none(lcd.time_unit)} {lcd.timescale}'

    df = pd.concat([x, lcd.flux, lcd.perm_index], axis=1)
    fig = px.scatter(df, x=x_column, y='flux', custom_data='perm_index')
    fig.update_traces(
        selected={'marker': {'color': 'orange', 'size': 5}},
        hoverinfo='none',  # Important
        hovertemplate=None,  # Important
        mode='markers',
        marker=dict(color='blue', size=5, symbol='circle')
    )
    fig.update_layout(xaxis={'title': 'phase', 'tickformat': '.1f'},
                      yaxis_title='flux',
                      margin=dict(l=0, b=20),  # r=50, t=50, b=20))
                      # dragmode='lasso'  # Enable lasso selection mode by default
                      )

    # fig = go.Figure()

    # fig.add_trace(go.Scatter(
    #     x=x, y=lcd.flux,
    #     error_y=dict(type='data', array=lcd.flux_err, visible=True),
    #     selected={'marker': {'color': 'orange', 'size': 5}},
    #     hoverinfo='none',  # Important
    #     hovertemplate=None,
    #     # mode='markers+lines',
    #     mode='markers',
    #     marker=dict(color='blue', size=6, symbol='circle'),
    #     # line=dict(color='blue', width=1)  # , dash='dash')
    # ))

    fig.update_layout(
        title=title,
        showlegend=False,
        margin=dict(l=0, b=20, t=30, r=20),
        xaxis_title=xaxis_title,
        yaxis_title=f'flux, {safe_none(flux_unit)}'
    )

    return fig


# @callback(
#     Output('store_tess_lightcurve', 'data', allow_duplicate=True),
#     Input('fold_tess_lc_switch', 'value'),
#     State('store_tess_lightcurve', 'data'),
#     State('period_tess_lc_input', 'value'),
#     prevent_initial_call=True
# )
# def fold(fold_lc, js_lightcurve: str, period):
#     period_unit = 'd'
#     # if n_clicks is None:
#     #     raise PreventUpdate
#     lcd = CurveDash(js_lightcurve)
#     lcd.folded_view = fold_lc
#     lcd.period = period
#     lcd.period_unit = period_unit
#     return lcd.serialize()


@callback(
    Output('store_tess_lightcurve', 'data', allow_duplicate=True),
    [Input('recreate_selected_tess_lc_button', 'n_clicks'),
     State('data_tess_lc_table', 'selected_rows'),
     State('data_tess_lc_table', 'data'),
     State('stitch_switch', 'value'),
     State('flux_tess_lc_switch', 'value'),
     State('fold_tess_lc_switch', 'value'),
     State('period_tess_lc_input', 'value')],
    prevent_initial_call=True
)
def replot_selected_curves(n_clicks, selected_rows, table_data, stitch, flux_method, phase_view, period):
    if n_clicks is None:
        raise PreventUpdate
    try:
        lc = create_lc_from_selected_rows(selected_rows, table_data, stitch, flux_method, phase_view, period)
        set_props('div_tess_lc_alert', {'children': None, 'style': {'display': 'none'}})
        return lc

    except Exception as e:
        logging.warning(f'lightcurve_tess.replot_selected_curves: {e}')
        alert_message = message.warning_alert(e)
        set_props('div_tess_lc_alert', {'children': alert_message, 'style': {'display': 'block'}})
        return dash.no_update


@callback(
    Output('store_tess_lightcurve', 'data', allow_duplicate=True),
    Input('recalc_phase_tess_lc_button', 'n_clicks'),
    State('store_tess_lightcurve', 'data'),
    State('period_tess_lc_input', 'value'),
    prevent_initial_call=True)
def recalculate_phase(n_clicks, js_lightcurve, period):
    if n_clicks is None:
        raise PreventUpdate
    try:
        lcd = CurveDash(js_lightcurve)
        lcd.period = period
        period_unit = 'd'
        lcd.period_unit = period_unit
        set_props('div_tess_lc_alert', {'children': None, 'style': {'display': 'none'}})
        return lcd.serialize()
    except Exception as e:
        logging.warning(f'lightcurve_tess.recalculate_phase: {e}')
        alert_message = message.warning_alert(e)
        set_props('div_tess_lc_alert', {'children': alert_message, 'style': {'display': 'block'}})
        return dash.no_update


# Switch between folded and time view. Recalculate phases if needed
@callback([Output('store_tess_lightcurve', 'data', allow_duplicate=True),
           Output('fold_tess_lc_switch', 'value')],
          Input('fold_tess_lc_switch', 'value'),
          State('store_tess_lightcurve', 'data'),
          State('period_tess_lc_input', 'value'),
          prevent_initial_call=True
          )
def fold(phase_view, js_lightcurve, period):
    try:
        if phase_view and not period:
            raise PipeException('Set the period and try again')
        lcd = CurveDash(js_lightcurve)
        if phase_view:
            lcd.period = period
            period_unit = 'd'
            lcd.period_unit = period_unit
        lcd.folded_view = phase_view
        set_props('div_tess_lc_alert', {'children': None, 'style': {'display': 'none'}})
        return lcd.serialize(), dash.no_update
    except Exception as e:
        logging.warning(f'lightcurve_tess.fold: {e}')
        alert_message = message.warning_alert(e)
        set_props('div_tess_lc_alert', {'children': alert_message, 'style': {'display': 'block'}})
        return dash.no_update, False


@callback(Output('graph_tess_lc', 'figure', allow_duplicate=True),
          Input('store_tess_lightcurve', 'data'),
          State('fold_tess_lc_switch', 'value'),
          prevent_initial_call=True
          )
def plot_tess_curve(js_lightcurve, phase_view):
    try:
        fig = plot_lc(js_lightcurve, phase_view)
        set_props('div_tess_lc_alert', {'children': None, 'style': {'display': 'none'}})
        return fig
    except Exception as e:
        logging.warning(f'lightcurve_tess.plot_tess_curve: {e}')
        alert_message = message.warning_alert(e)
        set_props('div_tess_lc_alert', {'children': alert_message, 'style': {'display': 'block'}})
        return dash.no_update


@callback(
    output=dict(
        pg_fig=Output('graph_tess_lc_periodogram', 'figure'),
        pg_row_style=Output('tess_lc_periodogram_row', 'style'),
        period1_res=Output('period1_res', 'children'),
        period2_res=Output('period2_res', 'children'),
        period4_res=Output('period4_res', 'children'),
        results_row_style=Output('tess_lc_periodogram_results_row', 'style'),
    ),
    inputs=dict(n_clicks=Input('periodogram_tess_lc_button', 'n_clicks')),
    state=dict(
        js_lightcurve=State('store_tess_lightcurve', 'data'),
        period_freq=State('period_freq_tess_lc_switch', 'value'),
        method=State('method_tess_lc_switch', 'value'),
        nterms=State('periodogram_nterms', 'value'),
        oversample=State('periodogram_oversample', 'value'),
        p_min=State('periodogram_min', 'value'),
        p_max=State('periodogram_max', 'value'),
        duration=State('periodogram_duration', 'value'),
    ), prevent_initial_call=True)
def periodogram(n_clicks, js_lightcurve, period_freq, method, nterms, oversample, p_min, p_max, duration):
    if not n_clicks:
        raise PreventUpdate
    output_keys = list(ctx.outputs_grouping.keys())
    output = {key: dash.no_update for key in output_keys}

    try:
        lcd = CurveDash(js_lightcurve)
        if lcd.lightcurve is None:
            raise PipeException('Download curves first')
        kurve = lightkurve.LightCurve(time=lcd.jd, flux=lcd.flux, flux_err=lcd.flux_err)

        if method == 'ls':
            kwargs = dict(method=method, oversample_factor=oversample,
                          minimum_period=p_min, maximum_period=p_max, nterms=nterms)
        else:  # BLS
            if duration is None:
                kwargs = dict(method=method, minimum_period=p_min, maximum_period=p_max)
            else:
                kwargs = dict(method=method, minimum_period=p_min, maximum_period=p_max, duration=duration)

        # pg = kurve.to_periodogram(method=method, oversample_factor=oversample,
        #                           minimum_period=p_min, maximum_period=p_max, nterms=nterms, duration=duration)
        pg = kurve.to_periodogram(**kwargs)

        output['period1_res'] = f'{pg.period_at_max_power.value: .4f}'
        output['period2_res'] = f'{pg.period_at_max_power.value * 2: .4f}'  # double P
        output['period4_res'] = f'{pg.period_at_max_power.value * 4: .4f}'  # quadruple P

        if period_freq == 'frequency':
            x = pg.frequency
            xaxis_title = 'Frequency, 1/d'
            xaxis_type = 'linear'
        else:
            x = pg.period
            xaxis_title = 'Period, d'
            xaxis_type = 'log'

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=pg.power,
            # hoverinfo='none',  # Important
            hovertemplate='%{x:.4f}<extra></extra>',
            # %{x:.4f}: x-format; <extra></extra>: removes the default trace info
            mode='lines',
            # mode='markers',
            # marker=dict(color='blue', size=6, symbol='circle'),
            line=dict(color='blue', width=1)  # , dash='dash')
        ))

        title = 'Periodogram'
        fig.update_layout(
            title=title,
            showlegend=False,
            margin=dict(l=0, b=20, t=30, r=20),
            xaxis_type=xaxis_type,
            xaxis_title=xaxis_title,
            yaxis_title='Power'
        )
        output['pg_fig'] = fig
        output['pg_row_style'] = {'display': 'block'}
        output['results_row_style'] = {'display': 'block'}
        set_props('div_tess_lc_alert', {'children': None, 'style': {'display': 'none'}})
    except Exception as e:
        logging.warning(f'lightcurve_tess.periodogram: {e}')
        alert_message = message.warning_alert(e)
        set_props('div_tess_lc_alert', {'children': alert_message, 'style': {'display': 'block'}})

    return output


# ----------- The clientside part ---------

# Plot light curve
# clientside_callback(
#     ClientsideFunction(
#         namespace='clientside',
#         function_name='plotLightcurveFromStore'
#     ),
#     Output('graph_tess_lc', 'figure'),
#     Input('store_tess_lightcurve', 'data'),
#     State('graph_tess_lc', 'figure'),
#     prevent_initial_call=True
# )

# # Switch between folded and time view. All phases have been recalculated already
# clientside_callback(
#     ClientsideFunction(
#         namespace='clientside',
#         function_name='updateFoldedView'
#     ),
#     Output('store_tess_lightcurve', 'data', allow_duplicate=True),
#     Input('fold_tess_lc_switch', 'value'),
#     State('store_tess_lightcurve', 'data'),
#     prevent_initial_call=True
# )

# Mark data as selected
clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='selectData'
    ),
    Output('store_tess_lightcurve', 'data', allow_duplicate=True),
    Input('graph_tess_lc', 'selectedData'),
    Input('graph_tess_lc', 'clickData'),
    State('store_tess_lightcurve', 'data'),
    prevent_initial_call=True
)


# Unmark data
# clientside_callback(
#     ClientsideFunction(
#         namespace='clientside',
#         function_name='unselectData'
#     ),
#     Output('store_tess_lightcurve', 'data', allow_duplicate=True),
#     Input('btn_tess_unselect', 'n_clicks'),
#     State('store_tess_lightcurve', 'data'),
#     prevent_initial_call=True
# )


# Delete selected points
# clientside_callback(
#     ClientsideFunction(
#         namespace='clientside',
#         function_name='deleteSelected'
#     ),
#     Output('store_tess_lightcurve', 'data', allow_duplicate=True),
#     Input('btn_tess_delete', 'n_clicks'),
#     State('store_tess_lightcurve', 'data'),
#     prevent_initial_call=True
# )


@callback(
    output=dict(
        lightcurve=Output('store_tess_lightcurve', 'data', allow_duplicate=True),
        message_results=Output('download_tess_lc_result', 'children'),
        graph_tab_disabled=Output('tess_lc_graph_tab', 'disabled'),
        active_tab=Output('tess_lc_tabs', 'active_tab'),
    ),
    inputs=dict(n_clicks=Input('download_tess_lc_button', 'n_clicks')),
    state=dict(
        selected_rows=State('data_tess_lc_table', 'selected_rows'),
        table_data=State('data_tess_lc_table', 'data'),
        stitch=State('stitch_switch', 'value'),
        flux_method=State('flux_tess_lc_switch', 'value')
    ),
    background=True,
    running=[(Output('download_tess_lc_button', 'disabled'), True, False),
             (Output('cancel_download_tess_lc_button', 'disabled'), False, True)],
    cancel=[Input('cancel_download_tess_lc_button', 'n_clicks')],
    prevent_initial_call=True)
def download_tess_lc_curve(n_clicks, selected_rows, table_data, stitch, flux_method):
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
    output_keys = list(ctx.outputs_grouping.keys())
    output = {key: dash.no_update for key in output_keys}

    try:
        # Store the loaded light curve into dcc.Store
        output['lightcurve'] = create_lc_from_selected_rows(selected_rows, table_data, stitch, flux_method)
        output['graph_tab_disabled'] = False
        output['active_tab'] = 'tess_lc_graph_tab'
        output['message_results'] = 'Success, switch to the next Tab'
        set_props('div_tess_lc_download_alert', {'children': '', 'style': {'display': 'none'}})
    except Exception as e:
        logging.warning(f'lightcurve_tess.download_tess_curve {e}')
        alert_message = message.warning_alert(e)
        output['graph_tab_disabled'] = True
        output['message_results'] = ''
        set_props('div_tess_lc_download_alert', {'children': alert_message, 'style': {'display': 'block'}})

    set_props('fold_tess_lc_switch', {'value': False})
    return output


@callback(Output('download_tess_lc_lightcurve', 'data'),  # ------ Download -----
          Input('btn_download_tess_lc', 'n_clicks'),
          State('store_tess_lightcurve', 'data'),
          State('select_tess_lc_format', 'value'),
          prevent_initial_call=True)
def download_to_user_tess_lc_lightcurve(n_clicks, js_lightcurve, table_format):
    """
    Downloads light curve into user's computer
    """
    if not n_clicks:
        raise PreventUpdate
    if js_lightcurve is None:
        raise PreventUpdate
    try:
        lcd = CurveDash(js_lightcurve)
        # raise PipeException('test')
        # bstring is "bytes"
        file_bstring = lcd.download(table_format)

        outfile_base = f'lc_tess_' + lcd.title.replace(' ', '_').replace(':', '').replace(',', '')
        ext = lcd.get_file_extension(table_format)
        outfile = f'{outfile_base}.{ext}'

        ret = dcc.send_bytes(file_bstring, outfile)
        set_props('div_tess_lc_alert', {'children': '', 'style': {'display': 'none'}})

    except Exception as e:
        logging.warning(f'tess_lc.download_tess_lc_lightcurve: {e}')
        alert_message = message.warning_alert(e)
        set_props('div_tess_lc_alert', {'children': alert_message, 'style': {'display': 'block'}})
        ret = dash.no_update

    return ret


@callback(Output('period_tess_lc_input', 'value', allow_duplicate=True),
          Input('use_period1_btn', 'n_clicks'),
          State('period1_res', 'children'),
          prevent_initial_call=True)
def use_period1(n_clicks, period_str):
    if not n_clicks:
        raise PreventUpdate
    try:
        period = float(period_str)
    except ValueError:
        logging.warning(f'lightcurve_tess.use_period1: {period_str} could not be converted into the float')
        period = None
    return period


@callback(Output('period_tess_lc_input', 'value', allow_duplicate=True),
          Input('use_period2_btn', 'n_clicks'),
          State('period2_res', 'children'),
          prevent_initial_call=True)
def use_period2(n_clicks, period_str):
    if not n_clicks:
        raise PreventUpdate
    try:
        period = float(period_str)
    except ValueError:
        logging.warning(f'lightcurve_tess.use_period2: {period_str} could not be converted into the float')
        period = None
    return period


@callback(Output('period_tess_lc_input', 'value', allow_duplicate=True),
          Input('use_period4_btn', 'n_clicks'),
          State('period4_res', 'children'),
          prevent_initial_call=True)
def use_period4(n_clicks, period_str):
    if not n_clicks:
        raise PreventUpdate
    try:
        period = float(period_str)
    except ValueError:
        logging.warning(f'lightcurve_tess.use_period4: {period_str} could not be converted into the float')
        period = None
    return period


clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='clearInput'
    ),
    Output('period_tess_lc_input', 'value'),
    Input('clear_period_btn', 'n_clicks'),
    prevent_initial_call=True
)
