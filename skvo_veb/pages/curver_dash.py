import json
import logging

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

import pandas as pd
from astropy.table import Table

from astropy.stats import sigma_clipped_stats

from dash import dcc, html, Input, Output, State, register_page, callback
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable

from dash.exceptions import PreventUpdate
from lightkurve import search_targetpixelfile, search_tesscut
from lightkurve.correctors import PLDCorrector

import aladin_lite_react_component

# TESS stuff
from skvo_veb.utils import tess_cache as cache

register_page(__name__, name='TESS',
              path='/igebc/tess',
              title='TESS Curver Tool',
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

# switch_label_style = {'display': 'inline-block', 'padding': '5px'}  # In the row, otherwise 'block'
switch_label_style = {'display': 'block', 'padding': '2px'}  # In the row, otherwise 'block'


def imshow_logscale(img, scale_method=None, gamma=0.99, **kwargs):
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

    val_min = img.min()
    val_max = img.max()
    val_range = val_max - val_min
    left = val_min
    left = left if left > 0 else img_true_min
    right = val_max+val_range/100
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
    print(ticks_text)
    fig.update_layout(
        coloraxis_colorbar=dict(
            tickvals=tickvals,
            ticktext=ticks_text,
        ),
    )
    fig.data[0]['customdata'] = img     # store here not-logarithmic values
    fig.data[0]['hovertemplate'] = '%{customdata:.0f}<extra></extra>'
    return fig


def layout():
    return dbc.Container([
        dbc.Row([
            html.H1('TESS Curver', className="text-primary text-left fs-3"),
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
                    dbc.Label('Radius', html_for='size_input', style={'width': '7em'}),
                    dcc.Input(id='radius_tess_input', persistence=True, type='number', min=1, value=11,
                              style={'width': '100%'}),
                ], direction='horizontal', gap=2, style={'marginBottom': '5px'}),
                dbc.Stack([
                    dbc.Label('Size', html_for='size_input', style={'width': '7em'}),
                    dcc.Input(id='size_tess_input', persistence=True, type='number', min=1, value=11,
                              style={'width': '100%'}),
                ], direction='horizontal', gap=2, style={'marginBottom': '5px'}),

                dbc.Stack([
                    dbc.Label('Sector', html_for='sector_drop', style={'width': '7em'}),
                    dcc.Input(id='sector_drop', type='number', min=0, value=0,
                              persistence=True, style={'width': '100%'}),  # '5em'}),
                    # dcc.Dropdown(id='sector_drop', clearable=False,
                    #              options=[{'label': sector, 'value': sector} for sector in ['0', '1', '2']],
                    #              style={'width': '100%'}),  # '5em'}),
                ], direction='horizontal', gap=2, style={'marginBottom': '5px'}),
                dbc.Stack([
                    dbc.Label('Threshold', html_for='thresh_input', style={'width': '7em'}),
                    dcc.Input(id='thresh_input', type='number', min=0, value=2,
                              persistence=True, style={'width': '100%'}),  # '5em'}),
                ], direction='horizontal', gap=2, style={'marginBottom': '5px'}),
                html.Div([
                    dbc.Row([
                        dbc.Col(dbc.Button('Search', id='search_tess_button', size="sm", style={'width': '100%'}),
                                width=6),
                        dbc.Col(dbc.Button('Plot pixel', id='plot_pixel_button', size="sm", style={'width': '100%'}),
                                width=6),
                    ], style={'marginBottom': '5px'}),
                    dbc.Row([
                        dbc.Col(dbc.Button('Plot curve', id='plot_curve_tess_button', size="sm",
                                           style={'width': '100%'}),
                                width=6),
                        dbc.Col(dbc.Button('Compare', id='plot_difference_button', size="sm", style={'width': '100%'}),
                                width=6),
                    ], style={'marginBottom': '5px'}),
                    dbc.Row([
                        dbc.Col([
                            dcc.RadioItems(
                                id='ffi_tpf_switch',
                                options=[
                                    {'label': 'FFI', 'value': 'ffi'},
                                    {'label': 'TPF', 'value': 'tpf'}
                                ],
                                value='tpf',
                                labelStyle=switch_label_style,
                            )], width=6),
                        dbc.Col([
                            dcc.RadioItems(
                                id='search_tess_switch',
                                options=[
                                    {'label': 'coord', 'value': 'coord'},
                                    {'label': 'name', 'value': 'name'}
                                ],
                                value='name',
                                labelStyle=switch_label_style,
                            )], width=6),
                    ], style={'marginBottom': '5px'}),
                    dbc.Row([
                        dbc.Col([
                            dcc.RadioItems(
                                id='star_tess_switch',
                                options=[
                                    {'label': 'Star1', 'value': '1'},
                                    {'label': 'Star2', 'value': '2'},
                                    {'label': 'Star3', 'value': '3'},
                                ],
                                value='1',
                                labelStyle=switch_label_style,
                            ),
                            dcc.RadioItems(
                                id='compare_switch',
                                options=[
                                    {'label': 'divide', 'value': 'divide'},
                                    {'label': 'subtr', 'value': 'subtract'},
                                ],
                                value='divide',
                                labelStyle=switch_label_style,
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
                            ),
                        ], width=6),
                        dbc.Col([
                            dbc.Checklist(options=[{'label': 'Sum', 'value': 1}], value=0, id='sum_switch',
                                          persistence=True, switch=True),
                            dbc.Checklist(options=[{'label': 'Auto mask', 'value': 1}], value=0, id='auto_mask_switch',
                                          persistence=True, switch=True),
                            dbc.Checklist(options=[{'label': 'Sub bkg', 'value': 1}], value=0,
                                          id='sub_bkg_switch', persistence=True, switch=True),
                            dbc.Checklist(options=[{'label': 'Fatten', 'value': 1}], value=0,
                                          id='flatten_switch', persistence=True, switch=True),
                            dbc.Label('Mask:', html_for='mask_switch'),
                            dcc.RadioItems(
                                id='mask_switch',
                                options=[
                                    {'label': 'pipe', 'value': 'pipeline'},
                                    {'label': 'thresh', 'value': 'threshold'},
                                ],
                                value='threshold',
                                labelStyle=switch_label_style,
                            ),
                        ], width=6),

                    ], style={'marginBottom': '5px'}),  # switches
                    dbc.Row([
                        dbc.Label('Pixel visualisation'),
                        dbc.Stack([
                            dbc.Label('Scale:', html_for='input_tess_gamma'),
                            dbc.Input(id='input_tess_gamma', value=1, type='number'),
                        ], direction='horizontal', gap=2, style={'marginBottom': '5px'}),

                    ], style={'marginBottom': '5px'}),  # tune visualization
                ])  # buttons
            ], md=2, sm=12, style={'padding': '10px', 'background': 'Silver'}),  # 'LightGray'}),  # Tools
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='px_tess_graph',
                                  config={'displaylogo': False},
                                  # style={'height': '100%'}),
                                  # style={'height': '100%', 'aspect-ratio': '1'}),
                                  # style={'height': '45vh', 'aspect-ratio': '1'}),
                                  style={'height': '40vh', 'aspect-ratio': '1'}),
                    ], md=6, sm=12),  # pixel graph
                    dbc.Col([
                        aladin_lite_react_component.AladinLiteReactComponent(
                            id='aladin_tess',
                            width=300,
                            height=300,
                            fov=round(2 * 10) / 60,  # in degrees
                            target='02:03:54 +42:19:47',
                            # stars=stars,
                        ),
                    ], md=6, sm=12)  # aladin
                ], align='center'),  # Px graph and Aladin
                dbc.Row([
                    dcc.Graph(id='curve_graph_1',
                              config={'displaylogo': False},
                              style={'height': '30vh'}),  # 100% of the viewport height
                ]),  # First Light Curve
                dbc.Row([
                    dcc.Graph(id='curve_graph_2',
                              config={'displaylogo': False},
                              style={'height': '30vh'}),  # 100% of the viewport height
                ]),  # Second Light Curve
                dbc.Row([
                    dcc.Graph(id='curve_graph_3',
                              config={'displaylogo': False},
                              style={'height': '30vh'}),  # 100% of the viewport height
                ]),  # Third Light Curve
            ], md=10, sm=12, style={'padding': '2px', 'background': 'Silver'}),  # All graphs
        ], style={'marginBottom': '10px'}),  # Buttons and Graph are here
        dbc.Row([
            html.H3("Search results", id="table_tess_header"),
            DataTable(
                id="data_tess_table",
                columns=[{"name": col, "id": col} for col in
                         ["#", "mission", "year", "author", "exptime", "target", "distance"]],
                data=[],
                # row_selectable="multi",
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
        ], id="table_tess_row", style={"display": "none"}),  # The Table is here
        dcc.Store(id='mask_store'),
        dcc.Store(id='wcs_store'),
        dcc.Store(id='lc1_store'),
        dcc.Store(id='lc2_store'),
    ], className="g-10", fluid=True, style={'display': 'flex', 'flexDirection': 'column', 'height': '100vh'})


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
        Conesearch radius.  If a float is given it will be assumed to be in
        units of arcseconds.  If `None` then we default to 0.0001 arcsec.
    :param target: str, int, or `astropy.coordinates.SkyCoord` object
        Target around which to search. Valid inputs include:
            * The name of the object as a string, e.g. "Kepler-10".
            * The KIC or EPIC identifier as an integer, e.g. 11904151.
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


def get_tpf_data(target, radius, sector: int):
    tpf_data = cache.load_tpf_fits(target=target, radius=radius, sector=sector)
    if tpf_data is None:
        tpf = get_tpf(target=target, radius=radius)
        if len(tpf) <= sector:
            raise PreventUpdate  # todo specify this exception
        tpf_data = tpf[sector].download()
        cache.save_tpf_fits(tpf_data, target=target, radius=radius, sector=sector)
    return tpf_data


def get_ffi(target):
    """
    lightkurve search_tesscut says:
    :param target: str, int, or `astropy.coordinates.SkyCoord` object
        Target around which to search. Valid inputs include:
            * The name of the object as a string, e.g. "Kepler-10".
            * The KIC or EPIC identifier as an integer, e.g. 11904151.
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


def get_ffi_data(target: str, sector: int, size: int):
    ffi_data = cache.load_ffi_fits(target=target, sector=sector, size=size)
    if ffi_data is None:
        ffi = get_ffi(target=target)
        if len(ffi) <= sector:
            raise PreventUpdate  # todo specify this exception
        ffi_data = ffi[sector].download(cutout_size=size)
        cache.save_ffi_fits(ffi_data, target=target, sector=sector, size=size)
    return ffi_data


def search_pixel_data(pixel_type, obj_name, ra, dec, search_type, radius, sector, size):
    if search_type == 'name':
        target = obj_name
    else:
        target = f'{ra} {dec}'
    if pixel_type == 'ffi':
        pixel_data = get_ffi_data(target=target, sector=sector, size=size)
    else:
        pixel_data = get_tpf_data(target=target, radius=radius, sector=sector)
    return target, pixel_data


def subtract_background_inplace(tpf):  # todo It doesn't work this way (in place) due to protected members
    # corrected_flux = np.zeros_like(tpf.flux)
    for i, frame in enumerate(tpf.flux):
        mean, median, std = sigma_clipped_stats(frame, sigma=3.0)
        logging.debug(f'frame {i} clipped {median=}')

        # corrected_flux[i] = frame - median_background
        # noinspection PyUnresolvedReferences
        bkg = median * u.electron / u.s
        tpf.flux[i] = tpf.flux[i] - bkg
    return tpf


# from plotly_utils import imshow_logscale  # todo: It is a beta testing


@callback(
    [Output('px_tess_graph', 'figure', allow_duplicate=True),
     Output('mask_store', 'data', allow_duplicate=True),
     Output('wcs_store', 'data', allow_duplicate=True)],
    [Input('plot_pixel_button', 'n_clicks'),
     State('input_tess_gamma', 'value'),
     State('ffi_tpf_switch', 'value'),
     State('obj_name_tess_input', 'value'),
     State('ra_tess_input', 'value'),
     State('dec_tess_input', 'value'),
     State('search_tess_switch', 'value'),
     State('radius_tess_input', 'value'),
     State('sector_drop', 'value'),
     State('size_tess_input', 'value'),
     State('thresh_input', 'value'),
     State('sum_switch', 'value'),
     State('mask_switch', 'value')],
    prevent_initial_call=True
)
def plot_pixel(n_clicks, gamma, pixel_type, obj_name, ra, dec, search_type, radius, sector, size, threshold, sum_it,
               mask_type):
    if n_clicks is None:
        raise PreventUpdate

    target, pixel_data = search_pixel_data(pixel_type=pixel_type, obj_name=obj_name, ra=ra, dec=dec,
                                           search_type=search_type, radius=radius, sector=sector, size=size)
    if mask_type == 'pipeline':
        mask = pixel_data.pipeline_mask
    else:
        mask = pixel_data.create_threshold_mask(threshold=threshold, reference_pixel='center')
    mask_shapes = create_shapes(mask)
    # log = False
    if sum_it:
        logging.debug(f'Sum {len(pixel_data.flux)}')
        # data_to_show = log_gamma(np.sum(pixel_data.flux[:], axis=0), log=log)
        data_to_show = np.sum(pixel_data.flux[:], axis=0)
    else:
        data_to_show = pixel_data.flux[0]  # take only the first crop

    # fig = px.imshow(data_to_show.value, color_continuous_scale='Viridis', origin='lower') fig = imshow_logscale(
    # data_to_show.value, scale_method=log_gamma, color_continuous_scale='Viridis', origin='lower')
    print(f'{gamma=} {type(gamma)=}')
    fig = imshow_logscale(data_to_show.value, scale_method=log_gamma, color_continuous_scale='Viridis', origin='lower',
                          gamma=gamma)
    # fig.update_traces(hovertemplate=None, hoverinfo='none')
    # fig.update_traces(hovertemplate="%{z:.0f}<extra></extra>", hoverinfo="z")
    fig.update_layout(title=pixel_type.upper(),  # coloraxis_showscale=False,
                      coloraxis_colorbar=dict(
                          len=0.9,  # Set the length (fraction of plot height, e.g., 0.5 = half the plot height)
                          thickness=15,  # Set the thickness (in pixels)
                      ),
                      xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
                      showlegend=False, margin=dict(l=20, b=20, t=30, r=20),
                      shapes=mask_shapes)

    wcs_di = dict(pixel_data.wcs.to_header())
    return fig, mask.tolist(), wcs_di


@callback(
    [Output("px_tess_graph", "figure", allow_duplicate=True),
     Output('mask_store', 'data', allow_duplicate=True),
     Output('aladin_tess', 'target', allow_duplicate=True)],
    [Input("px_tess_graph", "clickData"),
     State("px_tess_graph", "figure"),
     State('auto_mask_switch', 'value'),
     State('mask_store', 'data'),
     State('ffi_tpf_switch', 'value'),
     State('obj_name_tess_input', 'value'),
     State('ra_tess_input', 'value'),
     State('dec_tess_input', 'value'),
     State('search_tess_switch', 'value'),
     State('radius_tess_input', 'value'),
     State('sector_drop', 'value'),
     State('size_tess_input', 'value'),
     State('thresh_input', 'value')],
    prevent_initial_call=True,
)
def update_mask(clickData, fig, auto_mask, mask_list,
                pixel_type, obj_name, ra, dec, search_type, radius, sector, size, threshold):
    logging.debug(f'update_mask: {clickData}')
    if clickData is None:
        logging.debug('update_mask: nothing')
        raise PreventUpdate
    x = int(clickData['points'][0]['x'])
    y = int(clickData['points'][0]['y'])
    logging.debug(f'updated_mask: {x}, {y}')
    aladin_target = dash.no_update
    if auto_mask:  # Recreate mask around selected pixel
        target, pixel_data = search_pixel_data(pixel_type=pixel_type, obj_name=obj_name, ra=ra, dec=dec,
                                               search_type=search_type, radius=radius, sector=sector, size=size)
        coord = pixel_data.wcs.pixel_to_world(x, y)
        aladin_target = f'{coord.ra.deg} {coord.dec.deg}'
        mask = pixel_data.create_threshold_mask(threshold=threshold, reference_pixel=(x, y))
    else:
        mask = np.array(mask_list)
        mask[y, x] = not mask[y, x]  # Invert mask in the clicked pixel
    shapes = create_shapes(mask)
    fig["layout"]["shapes"] = shapes
    # current_fig.update_layout(shapes=shapes)
    # return current_fig, f'Clicked on pixel: ({x}, {y})'
    return fig, mask.tolist(), aladin_target


@callback(
    [Output('curve_graph_1', 'figure'),
     Output('curve_graph_2', 'figure'),
     Output('curve_graph_3', 'figure'),
     Output('lc1_store', 'data'),
     Output('lc2_store', 'data')],
    [Input('plot_curve_tess_button', 'n_clicks'),
     State('ffi_tpf_switch', 'value'),
     State('obj_name_tess_input', 'value'),
     State('ra_tess_input', 'value'),
     State('dec_tess_input', 'value'),
     State('search_tess_switch', 'value'),
     State('radius_tess_input', 'value'),
     State('mask_store', 'data'),
     State('sector_drop', 'value'),
     State('size_tess_input', 'value'),
     State('thresh_input', 'value'),
     State('star_tess_switch', 'value'),
     State('sub_bkg_switch', 'value'),
     State('flatten_switch', 'value'),
     State('ordinate_switch', 'value')],
    prevent_initial_call=True
)
def plot_lightcurve(n_clicks, pixel_type, obj_name, ra, dec, search_type, radius, mask_list, sector, size, threshold,
                    star_number, sub_bkg, flatten, ordinate):
    if n_clicks is None:
        raise PreventUpdate
    target, pixel_data = search_pixel_data(pixel_type=pixel_type, obj_name=obj_name, ra=ra, dec=dec,
                                           search_type=search_type, radius=radius, sector=sector, size=size)
    if mask_list is None:
        mask = pixel_data.create_threshold_mask(threshold=threshold, reference_pixel='center')
    else:
        mask = np.array(mask_list)
    # if sub_bkg:
    #     pixel_data = subtract_background_inplace(pixel_data)
    lc = pixel_data.to_lightcurve(aperture_mask=mask)

    # t = Table(lc)
    # di = t.to_pandas(index=False).to_dict()
    # di = lc.to_pandas().to_dict()        # time-column has gone into index!
    # jsons = json.dumps(di)

    quality_mask = lc['quality'] == 0  # mask by TESS quality
    lc = lc[quality_mask]
    jd = lc.time.value
    if ordinate == 'flux':
        flux_unit = str(lc.flux.unit)
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
    df = pd.DataFrame({'jd': jd, 'flux': flux})
    jsons = json.dumps(df.to_dict())

    time_unit = lc.time.format
    title = f'{pixel_type.upper()} {target} {lc.LABEL} sector:{lc.SECTOR}'
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
        return fig, dash.no_update, dash.no_update, jsons, dash.no_update
    elif star_number == '2':
        return dash.no_update, fig, dash.no_update, dash.no_update, jsons
    else:
        return dash.no_update, dash.no_update, fig, dash.no_update, dash.no_update


@callback(
    Output('curve_graph_3', 'figure', allow_duplicate=True),
    [Input('plot_difference_button', 'n_clicks'),
     State('lc1_store', 'data'),
     State('lc2_store', 'data'),
     State('compare_switch', 'value')],
    prevent_initial_call=True
)
def plot_difference(n_clicks, jsons_1, jsons_2, compar_method):
    if n_clicks is None:
        raise PreventUpdate
    df1 = pd.DataFrame.from_dict(json.loads(jsons_1))
    # df1 = df1.reset_index()

    df2 = pd.DataFrame.from_dict(json.loads(jsons_2))
    # df2 = df2.reset_index()

    lc1 = Table.from_pandas(df1)
    lc2 = Table.from_pandas(df2)
    # lc = lc[lc['quality'] == 0]  # mask by TESS quality
    # jd = lc1['index']   # todo !!!
    jd = lc1['jd']  # todo !!!
    if compar_method == 'divide':
        flux = lc2['flux'] / lc1['flux']
        title = 'Curve2 / Curve1'
    else:
        flux = lc2['flux'] - lc1['flux']
        title = 'Curve2 - Curve1'
    # time_unit = lc1.time.format
    # flux_unit = str(lc1.flux.unit)

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
                      xaxis_title=f'time',
                      yaxis_title=f'flux',
                      xaxis={'dtick': 1000},
                      # 'showticklabels': False},# todo tune it
                      )
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
     Output('px_tess_graph', 'figure')],
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
     Output("table_tess_row", "style"),  # to show the table and Title
     Output('ra_tess_input', 'value'),
     Output('dec_tess_input', 'value'),
     Output('aladin_tess', 'target')],
    [Input('search_tess_button', 'n_clicks'),
     State('ffi_tpf_switch', 'value'),
     State('obj_name_tess_input', 'value'),
     State('ra_tess_input', 'value'),
     State('dec_tess_input', 'value'),
     State('search_tess_switch', 'value'),
     State('radius_tess_input', 'value')],
    prevent_initial_call=True
)
def search(n_clicks, pixel_type, obj_name, ra, dec, search_type, radius):
    if n_clicks is None:
        raise PreventUpdate
    if search_type == 'name':
        target = obj_name
    else:
        # ra = float(ra)
        # dec = float(dec)
        target = f'{ra} {dec}'
    if pixel_type == 'ffi':
        pixel = get_ffi(target=target)
    else:
        pixel = get_tpf(target, radius=radius)
    try:
        target_ra_deg = pixel.ra[0]
        target_dec_deg = pixel.dec[0]
    except Exception as e:
        logging.warning(e)
        target_ra_deg = None
        target_dec_deg = None
    data = []
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

    if target_ra_deg is None or target_dec_deg is None:
        target_ra_deg = dash.no_update
        target_dec_deg = dash.no_update
        aladin_target = dash.no_update
    else:
        aladin_target = f'{target_ra_deg} {target_dec_deg:+}'
    return (f'Search {pixel_type.upper()} for {target}', data, {"display": "block"},
            target_ra_deg, target_dec_deg, aladin_target)

# if __name__ == '__main__':
#     app.run_server(debug=True)
