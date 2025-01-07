import logging

import aladin_lite_react_component
import astropy.units as u
import dash
import dash_bootstrap_components as dbc
import lightkurve
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS
from dash import dcc, html, Input, Output, State, register_page, callback, clientside_callback, set_props
from dash.dash_table import DataTable
from dash.exceptions import PreventUpdate
from lightkurve import search_targetpixelfile, search_tesscut, LightkurveError
from lightkurve.correctors import PLDCorrector

from skvo_veb.components import message
from skvo_veb.utils import tess_cache as cache
from skvo_veb.utils.curve_dash import CurveDash
from skvo_veb.utils.my_tools import PipeException

register_page(__name__, name='TESS cutout',
              order=3,
              path='/igebc/tess',
              title='TESS cutout Tool',
              in_navbar=True)


# Auxiliary
def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def log_gamma(data, gamma=0.9, log=True):
    """
    Gamma correction to enhance dark regions
    :param log: disable if False
    :param data:
    :param gamma: Adjust gamma to control the contrast in dark regions
    :return:
    """
    if not log:
        return data
    # data = normalize(data)
    # log_data = normalize(np.log1p(data))
    log_data = np.log1p(data)
    # return normalize(np.power(log_data, gamma))
    return np.power(log_data, gamma)


# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

switch_label_style = {'display': 'inline-block', 'padding': '5px'}  # In the row, otherwise 'block'
# switch_label_style = {'display': 'block', 'padding': '2px'}  # In the row, otherwise 'block'
label_font_size = '0.8em'
stack_wrap_style = {'marginBottom': '5px', 'flexWrap': 'wrap'}


def imshow_logscale(img, scale_method=None, show_colorbar=False, gamma=0.99, **kwargs):
    # from engineering_notation import EngNumber
    import matplotlib.ticker as ticker

    img_true_min = img[img > 0].min()
    if scale_method:
        img[img <= 0] = img_true_min
        log_data = scale_method(img, gamma=gamma)
    else:
        log_data = img
    fig = px.imshow(
        img=log_data,
        **kwargs,
    )

    if show_colorbar:
        val_min = img.min()
        val_max = img.max()
        val_range = val_max - val_min
        left = val_min
        left = left if left > 0 else img_true_min
        right = val_max + val_range / 100
        right = right if right > 0 else img_true_min
        locator = ticker.MaxNLocator(nbins=5)
        TICKS_VALS = np.array(locator.tick_values(left, right))
        TICKS_VALS[0] = left

        TICKS_VALS = TICKS_VALS[TICKS_VALS >= 0]
        TICKS_VALS[TICKS_VALS == 0] = img_true_min
        ticks_text = [f'{val:.0f}' for val in TICKS_VALS]
        # ticks_text = [f'{EngNumber(val)}' for val in TICKS_VALS]
        if scale_method is not None:
            tickvals = [scale_method(val, gamma=gamma) for val in TICKS_VALS]
        else:
            tickvals = TICKS_VALS
        fig.update_layout(
            coloraxis_colorbar=dict(
                tickvals=tickvals,
                ticktext=ticks_text,
            ),
        )
    else:
        fig.update_layout(coloraxis_showscale=False),

    fig.data[0]['customdata'] = img  # store here not-logarithmic values
    fig.data[0]['hovertemplate'] = '%{customdata:.0f}<extra></extra>'
    return fig


def layout():
    return dbc.Container([
        html.H1('TESS Cutout Tool', className="text-primary text-left fs-3"),
        dbc.Tabs([
            dbc.Tab(label='Search Sector', children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Stack([
                            dbc.Label('Object', html_for='obj_name_tess_input', style={'width': '7em'}),
                            dcc.Input(id='obj_name_tess_input', persistence=True, type='text', style={'width': '100%'}),
                        ], direction='horizontal', gap=2, style={'marginBottom': '5px'}),
                        dbc.Stack([
                            dbc.Label('RA', html_for='ra_input', style={'width': '7em'}),
                            dcc.Input(id='ra_tess_input', persistence=True, type='text', style={'width': '100%'}),
                        ], direction='horizontal', gap=2, style={'marginBottom': '5px'}),
                        dbc.Stack([
                            dbc.Label('DEC', html_for='dec_tess_input', style={'width': '7em'}),
                            dcc.Input(id='dec_tess_input', persistence=True, type='text', style={'width': '100%'}),
                        ], direction='horizontal', gap=2, style={'marginBottom': '5px'}),
                        dbc.Stack([
                            dbc.Label('Radius', html_for='radius_tess_input', style={'width': '7em'}),
                            dcc.Input(id='radius_tess_input', persistence=True, type='number', min=1, value=11,
                                      style={'width': '100%'}),
                        ], direction='horizontal', gap=2, style={'marginBottom': '5px'}),
                        dbc.Stack([
                            dbc.Button('Search', id='search_tess_button', size='sm'),
                            dbc.Button('Cancel', id='cancel_search_tess_button', size='sm', disabled=True),
                        ], direction='horizontal', gap=2, style=stack_wrap_style),
                        dcc.RadioItems(
                            id='ffi_tpf_switch',
                            options=[
                                {'label': 'FFI', 'value': 'ffi'},
                                {'label': 'TPF', 'value': 'tpf'}
                            ],
                            value='tpf',
                            labelStyle={'display': 'inline-block', 'padding': '5px'}),
                    ],
                        lg=2, md=3, sm=4, xs=12, style={'padding': '10px', 'background': 'Silver'}),  # SearchTools
                    dbc.Col([
                        dbc.Spinner(children=[
                            html.Div([
                                dbc.Row([
                                    dbc.Col([
                                        html.H3("Search results", id="table_tess_header"),
                                    ], md=6, sm=12),
                                    dbc.Col([
                                        dbc.Stack([
                                            dbc.Label('Size', html_for='size_ffi_input', style={'width': '7em'}),
                                            dcc.Input(id='size_ffi_input', persistence=True, type='number', min=1,
                                                      value=11,
                                                      style={'width': '100%'}),
                                            dbc.Button('Download sector', id='download_sector_button', size="sm",
                                                       style={'width': '100%'}),
                                            dbc.Button('Cancel', id='cancel_download_sector_button', size="sm",
                                                       style={'width': '100%'}),
                                        ], direction='horizontal', gap=2, style={'marginBottom': '5px'}),
                                    ], md=6, sm=12),
                                ]),
                                dbc.Row([
                                    DataTable(
                                        id="data_tess_table",
                                        columns=[{"name": col, "id": col} for col in
                                                 ["#", "mission", "year", "author", "exptime", "target", "distance"]],
                                        data=[],
                                        row_selectable="single",
                                        fixed_rows={'headers': True},  # Freeze the header
                                        style_table={
                                            'maxHeight': '30vh',
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
                            ], id="search_results_row", style={"display": "none"}),  # Search results
                            html.Div(id='div_tess_search_alert', style={"display": "none"}),  # Alert
                        ]
                        ),
                    ],  # md=10, xs=12),
                        lg=10, md=9, sm=8, xs=12),
                    # ], id="search_results_row", style={"display": "none"}, md=10, sm=12),
                ], style={'marginBottom': '10px'}),  # Search and SearchResults
                dbc.Spinner(children=[
                    dbc.Label(id="download_sector_result", children='',
                              style={"color": "green", "text-align": "center"}),
                    html.Div(id='div_tess_download_alert', style={"display": "none"}),  # Alert
                ], spinner_style={
                    "align-items": "center",
                    "justify-content": "center",
                }, color="primary")
            ], tab_id='tess_search_tab'),  # Search and SearchResults Tab
            dbc.Tab(label='Plot', children=[  # The Second Tab containing the content
                # html.Div([
                dbc.Row([
                    dbc.Col([
                        # dbc.Row([
                        #     dbc.Label('Visualization', html_for='input_tess_gamma',
                        #               style={"textAlign": "center"}),
                        # ], align='center'),  # Visualization label
                        dbc.Row([
                            dbc.Col([
                                dbc.Stack([
                                    dbc.Label('Scale', html_for='input_tess_gamma',
                                              style={'width': '7em', 'font-size': label_font_size}),
                                    dcc.Input(id='input_tess_gamma', inputMode='numeric', persistence=True,
                                              value=1, type='number', style={'width': '100%'}),
                                ], direction='horizontal', gap=2),  # Scale and Sum
                            ], width='auto'),

                        ], justify="between", style={'marginBottom': '5px'}, ),  # Visualization switches
                        dbc.Row([
                            dbc.Col([
                                dbc.Checklist(options=[{'label': 'Sum', 'value': 1}], value=0, id='sum_switch',
                                              persistence=True, switch=True,
                                              style={'font-size': label_font_size}),  # style={'margin-left': 'auto'}),
                            ], width='auto'),
                            dbc.Col(dbc.Button('Plot pixel', id='replot_pixel_button', size="sm",
                                               style={'width': '100%'}), width='auto'),
                        ], justify="between", style={'marginBottom': '10px'}),  # plot button
                        # dbc.Row([
                        #     dbc.Label('Mask', style={"textAlign": "center"}),
                        # ]),  # style={'marginBottom': '5px'}),  # Mask label
                        dbc.Row([
                            dbc.Col([
                                dbc.Stack([
                                    dbc.Label('Mask thresh', html_for='thresh_input',
                                              style={'width': '7em', 'font-size': label_font_size}),
                                    dcc.Input(id='thresh_input', inputMode='numeric', persistence=True,
                                              value=1, type='number',
                                              style={'width': '100%'}),
                                ], direction='horizontal', gap=2),  # Sc
                            ], width='auto'),  # Mask Threshold
                            dbc.Col([
                                dbc.Checklist(options=[{'label': 'Auto mask', 'value': 1}], value=0,
                                              id='auto_mask_switch',
                                              style={'font-size': label_font_size},
                                              persistence=True, switch=True),

                                dcc.RadioItems(
                                    id='mask_switch',
                                    options=[
                                        {'label': 'pipe', 'value': 'pipeline'},
                                        {'label': 'thresh', 'value': 'threshold'},
                                    ],
                                    value='threshold',
                                    labelStyle=switch_label_style,
                                    style={'font-size': label_font_size},
                                ),
                            ], width='auto'),  # Mask switches
                        ], justify='between', style={'marginBottom': '5px'}),  # mask switches
                    ], md=2, sm=4, style={'padding': '10px', 'background': 'Silver'}),  # tools
                    dbc.Col([
                        dcc.Markdown(
                            '_**Select mask and build the lightcurve**_:\n'
                            '* Click on a star in the **Aladin** applet to mark it on the pixel image\n'
                            '* **Handmade Mask:** Click on a pixel to set/unset mask\n'
                            '* **Auto-mask:** Click on a pixel to create a threshold mask around it\n'
                            '* **Pipeline mask:** Use the mask provided by the team\n'
                            '* **Flux/Cent X/Y:** View its time dependence\n'
                            '* **Flatten:** Not functional yet\n',
                            style={"font-size": 12, 'font-family': 'courier'}
                        ),
                    ], md=3, sm=8),  # Description
                    dbc.Col([
                        dcc.Graph(id='px_tess_graph',
                                  config={'displaylogo': False},
                                  style={'height': '250px'},  # 'margin': '0 auto'},
                                  # style={'height': '35vh'},
                                  # style={'height': '35vh'},
                                  # style={'height': '100%'},
                                  # style={'height': '100%', 'aspect-ratio': '1'},
                                  # style={'height': '45vh', 'aspect-ratio': '1'}),
                                  # style={'height': '40vh', 'aspect-ratio': '1'}
                                  ),
                    ], align='center', md=3, sm=6),  # pixel graph
                    dbc.Col([
                        aladin_lite_react_component.AladinLiteReactComponent(
                            id='aladin_tess',
                            width=300,
                            height=250,
                            fov=round(2 * 10) / 60,  # in degrees
                            target='02:03:54 +42:19:47',
                            # stars=stars,
                        ),
                    ], align='center', md=4, sm=6)  # aladin
                ], style={'marginBottom': '10px'}),  # align='center'),  # Px graph and Aladin
                dbc.Row([
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                dbc.Checklist(options=[{'label': 'Sub bkg', 'value': 1}], value=0,
                                              style={'font-size': label_font_size},
                                              id='sub_bkg_switch', persistence=True, switch=True),
                            ], width='auto'),
                            dbc.Col([
                                dbc.Checklist(options=[{'label': 'Fatten', 'value': 1}], value=0,
                                              style={'font-size': label_font_size},
                                              id='flatten_switch', persistence=True, switch=True),
                            ], width='auto'),
                        ], style={'marginBottom': '5px'}),
                        dbc.Row([
                            dcc.RadioItems(
                                id='star_tess_switch',
                                options=[
                                    {'label': 'Star1', 'value': '1'},
                                    {'label': 'Star2', 'value': '2'},
                                    {'label': 'Star3', 'value': '3'},
                                ],
                                value='1',
                                labelStyle=switch_label_style,
                                style={'font-size': label_font_size},
                            ),
                            dcc.RadioItems(
                                id='compare_switch',
                                options=[
                                    {'label': 'divide', 'value': 'divide'},
                                    {'label': 'subtract', 'value': 'subtract'},
                                ],
                                value='divide',
                                labelStyle=switch_label_style,
                                style={'font-size': label_font_size},
                            ),
                            dcc.RadioItems(
                                id='ordinate_switch',
                                options=[
                                    {'label': 'flux', 'value': 'flux'},
                                    {'label': 'cent x', 'value': 'x'},
                                    {'label': 'cent y', 'value': 'y'},
                                ],
                                value='flux',
                                labelStyle=switch_label_style,
                                style={'font-size': label_font_size},
                            ),
                        ], style={'marginBottom': '5px'}),  # curve tune
                        dbc.Row([
                            dbc.Stack([
                                # dbc.Col(
                                dbc.Button('Plot curve', id='plot_curve_tess_button', size="sm"),
                                #                   style={'width': '100%'}),
                                #         width='auto'),
                                # dbc.Col(
                                dbc.Button('Compare', id='plot_difference_button', size="sm"),
                                # style={'width': '100%'}),
                                # width='auto'),
                            ], direction='horizontal', gap=2, style=stack_wrap_style),
                        ], justify='between',
                            className='gy-1',
                            style={'marginBottom': '5px'}),  # plot buttons
                        dbc.Row([
                            # dbc.Col([
                            dbc.Stack([
                                dbc.Select(options=CurveDash.get_format_list(),
                                           # handler.get_format_list(),
                                           value=CurveDash.get_format_list()[0],
                                           # value=handler.get_format_list()[0],
                                           id='select_tess_format',
                                           style={'max-width': '7em', 'font-size': label_font_size}),
                                # ], width='auto'),
                                # dbc.Col([
                                dbc.Button('Download', id='btn_download_tess_lc', size="sm"),
                                # style={'width': '100%'}),
                                # ], direction='horizontal', gap=2)
                                # ], width='auto'),  # select a format
                            ], direction='horizontal', gap=2, style=stack_wrap_style),
                        ], justify='between',
                            className='gy-1',  # class adds vertical gaps between folded columns
                            style={'marginBottom': '5px', 'marginTop': '5px'}),  # download curve
                    ], lg=2, md=3, sm=4, xs=12, style={'padding': '10px', 'background': 'Silver'}),  # Light Curve Tools
                    dbc.Col([
                        html.Div(children='', id='div_tess_lc_alert', style={'display': 'none'}),
                        dbc.Accordion([
                            dbc.AccordionItem([
                                dcc.Graph(id='curve_graph_1',
                                          figure=go.Figure().update_layout(
                                              title='',
                                              margin=dict(l=0, b=20, t=30, r=20),
                                              xaxis_title=f'time',
                                              yaxis_title=f'flux',
                                          ),
                                          config={'displaylogo': False},
                                          style={'height': '40vh'}),
                            ], title='First Light Curve', item_id='accordion_item_1'),
                            dbc.AccordionItem([
                                dcc.Graph(id='curve_graph_2',
                                          figure=go.Figure().update_layout(
                                              title='',
                                              margin=dict(l=0, b=20, t=30, r=20),
                                              xaxis_title=f'time',
                                              yaxis_title=f'flux',
                                          ),
                                          config={'displaylogo': False},
                                          style={'height': '40vh'}),
                            ], title='Second Light Curve', item_id='accordion_item_2'),
                            dbc.AccordionItem([
                                dcc.Graph(id='curve_graph_3',
                                          figure=go.Figure().update_layout(
                                              title='',
                                              margin=dict(l=0, b=20, t=30, r=20),
                                              xaxis_title=f'time',
                                              yaxis_title=f'flux',
                                          ),
                                          config={'displaylogo': False},
                                          style={'height': '40vh'}),
                            ], title='Third Light Curve', item_id='accordion_item_3'),
                        ], id='accordion_tess_lc', start_collapsed=False,
                            active_item=['accordion_item_1', 'accordion_item_2', 'accordion_item_3'],
                            always_open=True)  # Light Curves
                    ], lg=10, md=9, sm=8, xs=12),  # Light Curves Accordion

                ], style={'marginBottom': '10px'}),  # Light Curves
            ], tab_id='tess_graph_tab', id='tess_graph_tab', disabled=True),  # Plot Tab
        ], active_tab='tess_search_tab', id='tess_tabs', style={'marginBottom': '5px'}),
        dcc.Store(id='store_search_result'),  # things showed in the data table
        dcc.Store(id='store_pixel_metadata'),  # stuff for recreation current pixel
        dcc.Store(id='mask_store'),
        dcc.Store(id='mask_slow_store'),
        dcc.Store(id='mask_fast_store'),
        dcc.Store(id='wcs_store'),
        dcc.Store(id='store_tess_lightcurve'),
        dcc.Store(id='store_tess_metadata'),
        dcc.Store(id='lc2_store'),
        # dcc.Store(id="active_item_store", storage_type="memory"),  # Allows tracking recently opened accordion item
        dcc.Download(id='download_tess_lc'),
    ], className="g-10", fluid=True, style={'display': 'flex', 'flexDirection': 'column'})


def create_shapes(target_mask):
    # Create a list of shapes to mark mask
    shapes = []
    for i in range(target_mask.shape[0]):
        for j in range(target_mask.shape[1]):
            if target_mask[i, j]:  # Only draw shapes for the masked pixels
                # Add red border (rectangle)
                shapes.append(
                    dict(
                        type="rect",
                        x0=j - 0.5, y0=i - 0.5,
                        x1=j + 0.5, y1=i + 0.5,
                        line=dict(color="red", width=1)
                    )
                )
                # First diagonal line (/)
                shapes.append(
                    dict(
                        type="line",
                        x0=j - 0.5, y0=i - 0.5,
                        x1=j + 0.5, y1=i + 0.5,
                        line=dict(color="red", width=1)
                    )
                )
    return shapes


def get_tpf(target, radius=10):
    """
    lightkurve search says:
    :param radius: float or `astropy.units.Quantity` object
        Conesearch radius.  If a float is given, it will be assumed to be in
        units of arcseconds.  If `None` then we default to 0.0001 arcsec.
    :param target: str, int, or `astropy.coordinates.SkyCoord` object
        Target around which to search. Valid inputs include:
            * The name of the object as a string, e.g. "Kepler-10".
            * The KIC or EPIC identifier as an integer, e.g., 11904151.
            * A coordinate string in decimal format, e.g. "285.67942179 +50.24130576".
            * A coordinate string in sexagesimal format, e.g. "19:02:43.1 +50:14:28.7".
            * An `astropy.coordinates.SkyCoord` object.
    :return:
    """
    tpf = cache.load('tpf', target=target, radius=radius)
    if tpf is None:
        tpf = search_targetpixelfile(target=target, mission='TESS', radius=radius)
        cache.save(tpf, 'tpf', target=target, radius=radius)
    repr(tpf)  # Do not touch this line :-)
    return tpf


def get_ffi(target):
    """
    lightkurve search_tesscut says:
    :param target: str, int, or `astropy.coordinates.SkyCoord` object
        Target around which to search. Valid inputs include:
            * The name of the object as a string, e.g. "Kepler-10".
            * The KIC or EPIC identifier as an integer, e.g., 11904151.
            * A coordinate string in decimal format, e.g. "285.67942179 +50.24130576".
            * A coordinate string in sexagesimal format, e.g. "19:02:43.1 +50:14:28.7".
            * An `astropy.coordinates.SkyCoord` object.
    :return:
    """
    ffi = cache.load('ffi', target=target)
    if ffi is None:
        ffi = search_tesscut(target)
        cache.save(ffi, prefix='ffi', target=target)
    repr(ffi)  # Do not touch this line :-)
    return ffi


def parse_table_data(selected_rows, table_data):
    # Parse the firsts selected row only
    if not selected_rows or not table_data:
        raise PreventUpdate
    if len(selected_rows) < 1:
        raise PreventUpdate
    selected_data = [table_data[i] for i in selected_rows]
    return selected_data[0]


# lll
@callback(
    [Output('store_pixel_metadata', 'data'),
     Output('wcs_store', 'data', allow_duplicate=True),
     Output('aladin_tess', 'target'),
     Output('download_sector_result', 'children'),
     Output('tess_graph_tab', 'disabled'),
     Output('tess_tabs', 'active_tab')],
    [Input('download_sector_button', 'n_clicks'),
     State('data_tess_table', 'selected_rows'),
     State('data_tess_table', 'data'),
     State('store_search_result', 'data'),
     State('size_ffi_input', 'value')],
    running=[(Output('download_sector_button', 'disabled'), True, False),
             (Output('cancel_download_sector_button', 'disabled'), False, True)],
    cancel=[Input('cancel_download_sector_button', 'n_clicks')],

    background=True,
    prevent_initial_call=True
)
def download_sector(n_clicks, selected_rows, table_data, pixel_di, size):
    if n_clicks is None:
        raise PreventUpdate
    print('print: download_sector')
    logging.debug('debug: download_sector')
    logging.info('info: download_sector')
    logging.warning('warning: download_sector')

    pixel_metadata = dash.no_update
    wcs_di = dash.no_update
    aladin_target = dash.no_update
    graph_tab_disabled = dash.no_update
    active_tab = dash.no_update

    try:
        pixel_metadata, pixel_data = download_selected_pixel(selected_rows, table_data, pixel_di, size)
        pixel_metadata['path'] = pixel_data.path
        pixel_metadata['shape'] = pixel_data.shape
        pixel_metadata['pipeline_mask'] = pixel_data.pipeline_mask
        wcs_di = dict(pixel_data.wcs.to_header())
        aladin_target = f'{pixel_data.ra} {pixel_data.dec}'
        sector_results_message = 'Success. Switch to the next Tab'
        graph_tab_disabled = False
        active_tab = 'tess_graph_tab'
        set_props('div_tess_download_alert', {'children': '', 'style': {'display': 'none'}})
    except Exception as e:
        logging.warning(f'tess_cutout.download_sector {e}')
        alert_message = message.warning_alert(e)
        sector_results_message = ''
        set_props('div_tess_download_alert', {'children': alert_message, 'style': {'display': 'block'}})

    return pixel_metadata, wcs_di, aladin_target, sector_results_message, graph_tab_disabled, active_tab


@callback(
    [Output('px_tess_graph', 'figure', allow_duplicate=True),
     Output('mask_store', 'data', allow_duplicate=True)],
    [Input('replot_pixel_button', 'n_clicks'),
     Input('store_pixel_metadata', 'data'),
     State('input_tess_gamma', 'value'),
     State('thresh_input', 'value'),
     State('sum_switch', 'value'),
     State('mask_switch', 'value'),
     State('auto_mask_switch', 'value')],
    prevent_initial_call=True
)
def plot_pixel(n_clicks, pixel_metadata, gamma, threshold, sum_it, mask_type, auto_mask):
    ctx = dash.callback_context
    if ctx.triggered and ctx.triggered[0]["prop_id"] == "replot_pixel_button.n_clicks":
        if n_clicks is None:
            raise PreventUpdate
    pixel_data = lightkurve.targetpixelfile.TessTargetPixelFile(pixel_metadata['path'])
    px_shape = pixel_data.shape[1:]
    mask = np.full(px_shape, False)
    if auto_mask:
        mask = (
            pixel_data.pipeline_mask if mask_type == "pipeline"
            else pixel_data.create_threshold_mask(threshold=threshold, reference_pixel="center")
        )
    mask_shapes = create_shapes(mask)

    if sum_it:
        data_to_show = np.sum(pixel_data.flux[:], axis=0)
    else:
        data_to_show = pixel_data.flux[0]  # take only the first crop

    # fig = px.imshow(data_to_show.value, color_continuous_scale='Viridis', origin='lower') fig = imshow_logscale(
    # data_to_show.value, scale_method=log_gamma, color_continuous_scale='Viridis', origin='lower')
    # print(f'{gamma=} {type(gamma)=}')
    show_colorbar = False
    fig = imshow_logscale(data_to_show.value, scale_method=log_gamma, color_continuous_scale='Viridis', origin='lower',
                          show_colorbar=show_colorbar, gamma=gamma)
    # fig.update_traces(hovertemplate="%{z:.0f}<extra></extra>", hoverinfo="z")
    if show_colorbar:
        coloraxis_colorbar = dict(len=0.9,  # Set the length (fraction of plot height, e.g., 0.5 = half the plot height)
                                  thickness=15  # Set the thickness (in pixels)
                                  )
        coloraxis_showscale = True
    else:
        coloraxis_colorbar = None
        coloraxis_showscale = False
    fig.update_layout(title=dict(
        text=f'{pixel_metadata.get("target", "")} {pixel_metadata.get("author", "")}',
        font=dict(size=12)
    ),
        coloraxis_showscale=coloraxis_showscale,
        coloraxis_colorbar=coloraxis_colorbar,
        xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
        showlegend=False, margin=dict(l=20, b=20, t=20, r=20),
        shapes=mask_shapes)

    return fig, mask.tolist()


def download_selected_pixel(selected_rows, table_data, pixel_di, size):
    pixel_args = parse_table_data(selected_rows, table_data)
    # restore SearchResults
    pixel_table = Table.from_pandas(pd.DataFrame.from_dict(pixel_di))
    pixel = lightkurve.SearchResult(pixel_table)
    if len(pixel) <= pixel_args['#']:
        raise PreventUpdate  # todo specify this exception
    if pixel_args.get('author', '') == 'TESScut':
        # lk cache it in .lighkurve/... folder but can't found to reuse (
        kwargs = {
            'target_name': pixel.target_name[pixel_args['#']],
            'mission': pixel.mission[pixel_args['#']],
            'size': size
        }
        pixel_data = cache.load_ffi_fits(**kwargs)
        if pixel_data is None:
            pixel_data = pixel[pixel_args['#']].download(cutout_size=size)
            cache.save_ffi_fits(pixel_data, **kwargs)
        pixel_args['pixel_type'] = 'FFI'
    else:
        # lk caches it in .lighkurve/... folder
        try:
            pixel_data = pixel[pixel_args['#']].download()
        except LightkurveError as e:
            import os
            logging.warning(f'download_selected_pixel exception: {e}')
            # Probably, we have the corrupted cache. Let's try clean it
            # Build the filename of cached lightcurve. See lightkurve/search.py
            # Sorry, but I don't want to change the default cache_dir:
            # noinspection PyProtectedMember
            download_dir = pixel[pixel_args['#']]._default_download_dir()
            table = pixel[pixel_args['#']].table
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
            pixel_data = pixel[pixel_args['#']].download()

        pixel_args['pixel_type'] = 'TPF'

    # return pixel_args, pixel_data
    return pixel_args, pixel_data


# Synchronize masks
clientside_callback(
    """
    function synchronizeMasksTriggerSlow(slowMask) {
        console.log("Synchronizing masks... Trigger = Slow");
        if (!slowMask) {
            window.dash_clientside.no_update;
        }
        return slowMask;
    }
    """,
    Output("mask_store", "data", allow_duplicate=True),
    Input("mask_slow_store", "data"),
    prevent_initial_call=True
)
clientside_callback(
    """
    function synchronizeMasksTriggerFast(fastMask) {
        if (!fastMask) {
            window.dash_clientside.no_update;
        }
        return fastMask;
    }
    """,
    Output("mask_store", "data", allow_duplicate=True),
    Input("mask_fast_store", "data"),
    prevent_initial_call=True
)


@callback(
    # Output('aladin_tess', 'target', allow_duplicate=True),
    Output('mask_slow_store', 'data', allow_duplicate=True),
    [Input("px_tess_graph", "clickData"),
     State("px_tess_graph", "figure"),
     State('store_pixel_metadata', 'data'),
     State('mask_switch', 'value'),
     State('auto_mask_switch', 'value'),
     State('thresh_input', 'value')],
    prevent_initial_call=True,
)
def create_mask(clickData, fig, pixel_metadata,
                mask_type, auto_mask, threshold):
    if not auto_mask:  # todo count here pipeline mask if selected and presented
        raise PreventUpdate
    if clickData is None:
        logging.debug('update_mask: nothing')
        raise PreventUpdate

    x = int(clickData['points'][0]['x'])
    y = int(clickData['points'][0]['y'])

    # aladin_target = dash.no_update
    if mask_type == 'pipeline':
        mask = np.array(pixel_metadata['pipeline_mask'])
    else:
        path_to_pixel_data = pixel_metadata['path']
        pixel_data = lightkurve.targetpixelfile.TessTargetPixelFile(path_to_pixel_data)
        # pixel_args, pixel_data = download_selected_pixel(selected_rows, table_data, pixel_di, size)
        # target, pixel_data = search_pixel_data(pixel_type=pixel_type, obj_name=obj_name, ra=ra, dec=dec,
        #                                        search_type=search_type, radius=radius, sector=sector, size=size)
        logging.debug(f'create_mask: {x}, {y}, {threshold=}')
        mask = pixel_data.create_threshold_mask(threshold=threshold, reference_pixel=(x, y))
        # coord = pixel_data.wcs.pixel_to_world(x, y)
        # aladin_target = f'{coord.ra.deg} {coord.dec.deg}'

    mask_shapes = create_shapes(mask)

    fig["layout"]["shapes"] = mask_shapes
    # return aladin_target, mask.tolist()
    return mask.tolist()


clientside_callback(
    """
    function updateFastMask(clickData, autoMask, maskList) {
        console.log('updateFastMask', autoMask, clickData);

        if (autoMask && autoMask.length > 0) {
            console.log('updateFastMask: no_update')
            return window.dash_clientside.no_update;
        }

        if (!clickData) {
            return window.dash_clientside.no_update;
        }

        const x = Math.round(clickData.points[0].x);
        const y = Math.round(clickData.points[0].y);
        const updatedMask = [...maskList];
        updatedMask[y][x] = updatedMask[y][x] ? 0 : 1;

        return updatedMask;
    }
    """,
    Output("mask_fast_store", "data", allow_duplicate=True),
    [Input("px_tess_graph", "clickData")],
    [State("auto_mask_switch", "value"),
     State("mask_store", "data")],
    prevent_initial_call=True
)

clientside_callback(
    """
    function updateFigureWithMask(mask, fig) {
        console.log('updateFigureWithMask');

        if (!mask || !fig) {
            return window.dash_clientside.no_update;
        }
        console.log('fig =', fig);
        // console.log('fig.layout=', fig.layout);
        
        // Recreate figure to trigger show updates
        const updatedShapes = mask.flatMap((row, rowIndex) =>
            row.map((val, colIndex) => {
                if (val) {
                    // Square
                    const rect = {
                        type: "rect",
                        x0: colIndex - 0.5,
                        x1: colIndex + 0.5,
                        y0: rowIndex - 0.5,
                        y1: rowIndex + 0.5,
                        line: {color: "red", width: 1},
                    };
                    // Diagonal
                    const line = {
                        type: "line",
                        x0: colIndex - 0.5,
                        x1: colIndex + 0.5,
                        y0: rowIndex - 0.5,
                        y1: rowIndex + 0.5,
                        line: {color: "red", width: 1},
                    };
                    return [rect, line];  // return square and diagonal
                }
                return null;
            })
        ).filter(Boolean).flat();  // flat array

        console.log('updatedShapes=', updatedShapes);

        const newLayout = {
            ...fig.layout,
            shapes: updatedShapes,
            selections: undefined
        };

        // Copy and recreate figure to trigger rendering on the user screen
        const newFigure = {
             ...fig,
             layout: newLayout
        };

        console.log('newLayout:', newLayout);

        return newFigure;
    }
    """,
    Output("px_tess_graph", "figure", allow_duplicate=True),
    Input("mask_store", "data"),
    State("px_tess_graph", "figure"),
    prevent_initial_call=True
)


@callback(
    output=dict(
        fig1=Output('curve_graph_1', 'figure'),
        fig2=Output('curve_graph_2', 'figure'),
        fig3=Output('curve_graph_3', 'figure'),
        lc1=Output('store_tess_lightcurve', 'data'),
        lc2=Output('lc2_store', 'data'),
        meta=Output('store_tess_metadata', 'data'),
    ),
    inputs=dict(n_clicks=Input('plot_curve_tess_button', 'n_clicks')),
    state=dict(
        pixel_metadata=State('store_pixel_metadata', 'data'),
        mask_list=State('mask_store', 'data'),
        star_number=State('star_tess_switch', 'value'),
        sub_bkg=State('sub_bkg_switch', 'value'),
        flatten=State('flatten_switch', 'value'),
        ordinate=State('ordinate_switch', 'value')
    ),
    prevent_initial_call=True
)
def plot_lightcurve(n_clicks, pixel_metadata, mask_list, star_number, sub_bkg, flatten, ordinate):
    if n_clicks is None:
        raise PreventUpdate

    output_keys = ['fig1', 'fig2', 'fig3', 'lc1', 'lc2', 'meta']
    output = {key: dash.no_update for key in output_keys}

    try:
        path_to_pixel_data = pixel_metadata['path']
        pixel_data = lightkurve.targetpixelfile.TessTargetPixelFile(path_to_pixel_data)

        if mask_list is None:
            logging.warning('No aperture mask provided')
            raise PipeException('No aperture mask provided')
        mask = np.array(mask_list)
        if mask.sum() < 1:
            logging.warning('No valid aperture mask provided')
            raise PipeException('No valid aperture mask provided')

        # noinspection PyTypeChecker
        lc = pixel_data.to_lightcurve(aperture_mask=mask)

        quality_mask = lc['quality'] == 0  # mask by TESS quality
        lc = lc[quality_mask]
        jd = lc.time.value
        flux_unit = None
        flux_err = None
        if ordinate == 'flux':
            flux_unit = str(lc.flux.unit)
            flux_err = lc.flux_err  # todo: take into account background errors?
            if sub_bkg:
                yaxis_title = f'flux - bkg, {flux_unit}'
                bkg = pixel_data.estimate_background(aperture_mask='background')
                # noinspection PyUnresolvedReferences
                flux = lc.flux - bkg.flux[quality_mask] * mask.sum() * u.pix  # todo check this !
            else:
                yaxis_title = f'flux, {flux_unit}'
                flux = lc.flux
            if flatten:
                pld = PLDCorrector(pixel_data)
                corrected_lc = pld.correct(pld_aperture_mask=mask)
                yaxis_title = f'PLD corrected, {flux_unit}'
                # l, trend = lc.flatten(return_trend=True)
                # flux = lc.flux - trend.flux
                # flux = trend.flux
                flux = corrected_lc.flux
                jd = corrected_lc.time.value
        elif ordinate == 'x':
            yaxis_title = 'centroid_col, px'
            flux = lc.centroid_col  # todo rename it somehow
        else:
            yaxis_title = 'centroid_row, px'
            flux = lc.centroid_row  # todo rename it somehow
        time_unit = lc.time.format

        name = lc.LABEL if lc.LABEL else pixel_metadata.get('target', '')
        curve_dash = CurveDash(jd=jd, flux=flux, flux_err=flux_err, name=name, time_unit=time_unit, flux_unit=flux_unit)

        # df = pd.DataFrame({'jd': jd, 'flux': flux})
        jsons = curve_dash.serialize()
        # jsons = json.dumps(df.to_dict())

        lc_metadata = {'target': name, 'img': pixel_metadata.get('pixel_type', '').upper(), 'sector': lc.sector}
        output['meta'] = lc_metadata
        title = (f'{pixel_metadata.get("pixel_type", "").upper()} '
                 f'{pixel_metadata.get("target", "")} {lc.LABEL} sector:{lc.SECTOR} {pixel_metadata.get("author", "")}')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=jd, y=flux,
                                 hoverinfo='none',  # Important
                                 hovertemplate=None,
                                 mode='markers+lines',
                                 marker=dict(color='blue', size=6, symbol='circle'),
                                 line=dict(color='blue', width=1)))
        fig.update_layout(title=title,
                          showlegend=False,
                          margin=dict(l=0, b=20, t=30, r=20),
                          xaxis_title=f'time, {time_unit}',
                          yaxis_title=yaxis_title,
                          )

        if star_number == '1':
            output['fig1'] = fig
            output['lc1'] = jsons
            active_item = ['accordion_item_1']
        elif star_number == '2':
            output['fig2'] = fig
            output['lc2'] = jsons
            active_item = ['accordion_item_2']
        else:
            output['fig3'] = fig
            active_item = ['accordion_item_3']
        set_props('div_tess_lc_alert', {'children': '', 'style': {'display': 'none'}})
        set_props('accordion_tess_lc', {'active_item': active_item})

    except Exception as e:
        logging.warning(f'tess_cutout.plot_lightcurve: {e}')
        alert_message = message.warning_alert(e)
        set_props('div_tess_lc_alert', {'children': alert_message, 'style': {'display': 'block'}})

    return output


@callback(
    Output('curve_graph_3', 'figure', allow_duplicate=True),
    [Input('plot_difference_button', 'n_clicks'),
     State('store_tess_lightcurve', 'data'),
     State('lc2_store', 'data'),
     State('compare_switch', 'value')],
    prevent_initial_call=True
)
def plot_difference(n_clicks, jsons_1, jsons_2, comparison_method):
    if n_clicks is None:
        raise PreventUpdate
    fig = dash.no_update
    try:
        dash_lc1 = CurveDash(jsons_1)
        dash_lc2 = CurveDash(jsons_2)

        # Both curves have the same length because they were calculated from the same set of cutouts
        jd = dash_lc1.jd  # todo: add time units
        if comparison_method == 'divide':
            flux = dash_lc2.flux / dash_lc1.flux
            title = 'Curve2 / Curve1'
        else:
            flux = dash_lc2.flux - dash_lc1.flux
            title = 'Curve2 - Curve1'

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=jd, y=flux,
                                 hoverinfo='none',  # Important
                                 hovertemplate=None,
                                 mode='markers+lines',
                                 marker=dict(color='blue', size=6, symbol='circle'),
                                 line=dict(color='blue', width=1)))
        fig.update_layout(title=title,
                          showlegend=False,
                          margin=dict(l=0, b=20, t=30, r=20),
                          xaxis_title=f'time, {str(dash_lc1.time_unit)}',
                          yaxis_title=f'flux',
                          # xaxis={'dtick': 1000},
                          # 'showticklabels': False},# todo tune it
                          )
        active_item = ['accordion_item_3']
        set_props('div_tess_lc_alert', {'children': '', 'style': {'display': 'none'}})
        set_props('accordion_tess_lc', {'active_item': active_item})
    except Exception as e:
        logging.warning(f'tess_cutout.plot_difference: {e}')
        alert_message = message.warning_alert(e)
        set_props('div_tess_lc_alert', {'children': alert_message, 'style': {'display': 'block'}})

    return fig


def mark_cross(fig, x, y, cross_size=0.3, line_width=2, color='cyan'):
    # cross_size = 10
    # line_width = 4
    # color = "blue"
    shapes = [
        {
            "type": "line",
            "x0": x - cross_size,
            "y0": y,
            "x1": x + cross_size,
            "y1": y,
            "line": {"color": color, "width": line_width},
        },
        {
            "type": "line",
            "x0": x,
            "y0": y - cross_size,
            "x1": x,
            "y1": y + cross_size,
            "line": {"color": color, "width": line_width},
        }
    ]

    # add mark to layout
    if "shapes" not in fig["layout"]:
        fig["layout"]["shapes"] = shapes
    else:
        fig["layout"]["shapes"].extend(shapes)

    return fig


@callback(
    [Output('ra_tess_input', 'value', allow_duplicate=True),
     Output('dec_tess_input', 'value', allow_duplicate=True),
     Output('px_tess_graph', 'figure', allow_duplicate=True)],
    [Input('aladin_tess', 'clickedCoordinates'),
     State('px_tess_graph', 'figure'),
     State('wcs_store', 'data')],
    prevent_initial_call=True
)
def mark_star(coord, fig, wcs_dict):
    if coord is None:
        logging.warning(f'mark_star: coord is None')
        raise PreventUpdate
    ra = coord.get('ra', None)
    dec = coord.get('dec', None)
    if ra is None or dec is None:
        logging.warning(f'mark_star: ra or dec is None')
        raise PreventUpdate
    # noinspection PyUnresolvedReferences
    sky_coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame='icrs')
    x, y = WCS(wcs_dict).world_to_pixel(sky_coord)
    # fig = add_marker(fig, x, y, marker_symbol="diamond", color="blue", size=12)
    fig = mark_cross(fig, x, y)
    return coord.get('ra'), coord.get('dec'), fig


@callback(
    [Output("table_tess_header", "children"),
     Output("data_tess_table", "data"),
     Output("data_tess_table", "selected_rows"),
     Output("search_results_row", "style"),  # show the table and Title
     Output('store_search_result', 'data'),
     Output('div_tess_search_alert', 'children'),
     Output('div_tess_search_alert', 'style')],
    [Input('search_tess_button', 'n_clicks'),
     State('ffi_tpf_switch', 'value'),
     State('obj_name_tess_input', 'value'),
     State('ra_tess_input', 'value'),
     State('dec_tess_input', 'value'),
     State('radius_tess_input', 'value')],
    running=[(Output('search_tess_button', 'disabled'), True, False),
             (Output('cancel_search_tess_button', 'disabled'), False, True),
             (Output('download_sector_result', 'children'),
              'I\'m working... Please wait', 'Press Download to get the lightcurve')],
    cancel=[Input('cancel_search_tess_button', 'n_clicks')],
    background=True,
    prevent_initial_call=True
)
def search(n_clicks, pixel_type, obj_name, ra, dec, radius):
    # """
    # Note: It might seem odd that I set an active accordionItem in this callback.
    # Actually, I need to start the application with all accordionItems open to prevent
    # unpleasant flickering during the initial loading of the lightcurve.
    # Here, I simply close all items except the first one.
    # """
    print('search')
    print(n_clicks, pixel_type, obj_name, ra, dec, radius)
    if n_clicks is None:
        raise PreventUpdate

    # set_props('download_sector_result', {'children': 'I\'m working... Please wait'})
    if obj_name:
        target = obj_name
    else:
        target = f'{ra} {dec}'
    try:
        if pixel_type == 'ffi':
            pixel = get_ffi(target=target)
        else:
            pixel = get_tpf(target, radius=radius)

        data = []
        if len(pixel) == 0:
            raise Exception('No data found')
        for row in pixel.table:
            data.append({
                '#': row['#'],
                'mission': row['mission'],
                'year': row['year'],
                'target': row["target_name"],
                "author": row["author"],
                "exptime": row["exptime"],
                "distance": row["distance"]
            })
        # Serialize Lightkurve.SearchResult to store it
        pixel_di = pixel.table.to_pandas().to_dict()
        selected_row = [0] if data else []  # select the first row by default
        content_style = {'display': 'block'}  # show the table
        alert_style = {'display': 'none'}  # hide the alert
        alert_message = ''
        # set_props('download_sector_result', {'children': 'Press Download to get the lightcurve'})

    except Exception as e:
        logging.warning(f'tess_cutout.search: {e}')  # todo add error message instead of Table with results
        alert_message = message.warning_alert(e)
        alert_style = {'display': 'block'}  # show the alert
        data = dash.no_update
        selected_row = []
        pixel_di = {}
        content_style = {'display': 'none'}  # hide the table
        # raise PreventUpdate

    return f'{pixel_type.upper()} {target}', data, selected_row, content_style, pixel_di, alert_message, alert_style


@callback(Output('download_tess_lc', 'data'),  # ------ Download -----
          Input('btn_download_tess_lc', 'n_clicks'),
          State('store_tess_lightcurve', 'data'),
          State('store_tess_metadata', 'data'),
          State('select_tess_format', 'value'),
          prevent_initial_call=True)
def download_tess_lc(_, js_lightcurve, di_metadata, table_format):
    if js_lightcurve is None:
        raise PreventUpdate
    try:
        lcd = CurveDash(js_lightcurve)
        # bstring is "bytes"
        file_bstring = lcd.download(table_format)

        outfile_base = f'lc_tess_' + "_".join(f"{key}_{value}" for key, value in di_metadata.items()).replace(" ", "_")
        ext = lcd.get_file_extension(table_format)
        outfile = f'{outfile_base}.{ext}'

        ret = dcc.send_bytes(file_bstring, outfile)
        set_props('div_tess_lc_alert', {'children': '', 'style': {'display': 'none'}})

    except Exception as e:
        logging.warning(f'tess_cutout.download_tess_lc: {e}')
        alert_message = message.warning_alert(e)
        set_props('div_tess_lc_alert', {'children': alert_message, 'style': {'display': 'block'}})
        ret = dash.no_update

    return ret


# todo: Add cut out lightcurve button
# def download_tess_lc(_, js_lightcurve, di_metadata, table_format):
#     # todo rewrite it
#     # todo add errors
#     if js_lightcurve is None:
#         raise PreventUpdate
#     # bstring is "bytes"
#     import io
#     df = pd.DataFrame.from_dict(json.loads(js_lightcurve))
#     if 'left' in di_metadata and 'right' in di_metadata:
#         df = df[(df['jd'] >= di_metadata['left']) & (df['jd'] <= di_metadata['right'])]
#     tab = Table.from_pandas(df)
#     if table_format in kurve._format_dict_text:
#         my_weird_io = io.StringIO()
#     elif table_format in kurve._format_dict_bytes:
#         my_weird_io = io.BytesIO()
#     else:
#         raise PipeException(f'Unsupported format {table_format}\n Valid formats: {str(kurve.format_dict.keys())}')
#     tab.write(my_weird_io, format=table_format, overwrite=True)
#     my_weird_string = my_weird_io.getvalue()
#     if isinstance(my_weird_string, str):
#         # instead, we could choose  dcc.send_string() or dcc.send_bytes() for text or byte string in Dash application
#         # I prefer to place all io-logic in one place, here, and convert all stuff into bytes
#         my_weird_string = bytes(my_weird_string, 'utf-8')
#     my_weird_io.close()  # todo Needed?
#
#     outfile_base = f'lc_tess_' + "_".join(f"{key}_{value}" for key, value in di_metadata.items()).replace(" ", "_")
#     ext = kurve.get_file_extension(table_format)
#     outfile = f'{outfile_base}.{ext}'
#
#     ret = dcc.send_bytes(my_weird_string, outfile)
#     return ret


@callback(
    Output('store_tess_metadata', 'data', allow_duplicate=True),
    # Output('curve_graph_1', 'selectedData'),
    Input('curve_graph_1', 'selectedData'),
    State('store_tess_metadata', 'data'),
    prevent_initial_call=True
)
def update_box_select_data(selected_data, di_metadata):
    print('update_box_select_data')
    if selected_data is None:
        raise PreventUpdate
    if 'range' in selected_data:
        if 'x' in selected_data['range']:
            left_border, right_border = selected_data['range']['x']
            print(f"Left border: {left_border}, Right border: {right_border}")
            di_metadata['left'] = np.round(left_border, 1)
            di_metadata['right'] = np.round(right_border, 1)
    return di_metadata
