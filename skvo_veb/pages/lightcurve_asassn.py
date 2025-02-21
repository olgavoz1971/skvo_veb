import logging

from dash import (register_page, html, dcc, callback, clientside_callback, ClientsideFunction,
                  Input, Output, State, ctx)
from dash.dash import no_update
from dash.exceptions import PreventUpdate
import plotly.express as px
import dash_bootstrap_components as dbc

from skvo_veb.components import message
from skvo_veb.utils.curve_dash import CurveDash
from skvo_veb.utils.my_tools import is_like_gaia_id
from skvo_veb.utils.request_asassn import load_asassn_lightcurve
from skvo_veb.utils.request_gaia import decipher_source_id

register_page(__name__, name='ASAS-SN',
              order=2,
              path='/igebc/asassn',
              title='IGEBC: ASAS-SN Lightcurve',
              in_navbar=True)

row_class_name = "d-flex g-2 justify-content-end align-items-end"


def layout(source_id=None, band='g'):
    if source_id is None:
        header_txt = 'Request ASAS-SN lightcurve'
    else:
        header_txt = f'ASAS-SN lightcurve\nGAIA DR3 {source_id} {band}'
    # header = html.Div(id='h1-asassn', children=[html.H1(h1_txt, className='text-primary text-left fs-3'),
    #                                             html.H2(header_txt, className='text-primary text-left fs-3')])
    header = html.H1(header_txt, id='h1-asassn',
                     className='text-primary text-left fs-3',
                     style={'white-space': 'pre-wrap'})
    fig = px.scatter()
    fig.update_traces(selected={'marker': {'color': 'orange', 'size': 10}},
                      # hovertemplate='%{x}<br>%{y}',
                      hoverinfo='none',  # Important
                      hovertemplate=None,  # Important
                      )
    fig.update_layout(xaxis={'title': 'phase', 'tickformat': '.1f'},
                      yaxis_title='flux',
                      # showlegend=True,
                      # margin=dict(l=0, b=50),  # r=50, t=50, b=20))
                      margin=dict(l=0, b=20),  # r=50, t=50, b=20))
                      dragmode='lasso'  # Enable lasso selection mode by default
                      )

    res = dbc.Container([
        dcc.Store(id='store_asassn_lightcurve'),
        # dcc.Store(id='store-asassn-metadata'),
        html.Br(),
        header,
        html.Br(),
        dbc.Row([
            dbc.Col([
                dbc.Stack([
                    dcc.Markdown('Name'),
                    dbc.Input(placeholder='type object name', value=source_id if source_id is not None else '',
                              type='text', id='input-asassn-source-id',
                              # persistence=True
                              ),
                ], direction="horizontal", gap=2),
            ], md=5),  # Gaia ID
            dbc.Col([
                dbc.Stack([
                    dcc.Markdown('Band'),
                    dbc.Select(options=['V', 'g'], value=band, id='select-asassn-band',
                               # persistence=True,
                               style={'width': 100}),
                ], direction="horizontal", gap=2),
            ], md=2),  # Select Band
            dbc.Col([
                dbc.Stack([
                    dbc.Button('Submit', size="md", color='primary', id='btn-asassn-new-source'),
                    # dbc.Button('Clear', size="md", class_name='me-3', color='light', id='btn-asassn-clear-source_id'),
                    dbc.Button('Clear', size="md", color='light', id='btn-asassn-clear-source_id'),
                    dbc.Button('Force', size="md", color='warning', outline=True, id='btn-asassn-update'),
                    dbc.Tooltip('Forced updates may take some time', target='btn-asassn-update', placement='bottom'),
                    dbc.Button('Cancel', size="md", disabled=True, id='btn-cancel-asassn-update'),
                ], direction="horizontal", gap=2),
            ], md=2, align='end'),

        ], class_name='row_class_name'),  # class_name="g-2"),  # Select new star and band
        html.Br(),
        # dbc.Row(id='row-asassn-content', class_name='row_class_name'),  # The rest of the layout
        # html.Div([
        dbc.Row(id='row-asassn-content', children=[
            dbc.Row([
                html.Div([
                    dbc.Switch(id='switch-asassn-view', label='Folded view', value=True,
                               persistence=True),
                    # dbc.Checklist(options=[{'label': 'Folded view', 'value': 1}], value=0, id='switch-asassn-view',
                    #               persistence=True, switch=True),
                    dbc.Label('Unable to fold; the period is unknown', id='label-switch-asassn-view-warning'),
                ], style={'min-height': '30px'}),

            ], class_name="g-2"),  # Switch view/epoch
            dbc.Row([

            ], class_name="g-2"),  # Switch view/epoch alert
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        dcc.Graph(id='graph-asassn-curve',
                                  figure=fig,
                                  config={'displaylogo': False}),
                    ], class_name="g-0"),  # Graph
                ], md=12, sm=12),  # width={'size': 8, 'offset': 0, 'order': 1}),  # lightcurve Graph
            ], class_name="g-0"),  # g-0 -- Row without 'gutters'       # Lightcurve stuff
            dbc.Row([
                dcc.Markdown('_**Click on a point to select it, or use Lasso or Box selector**_',
                             style={"font-size": 14, 'font-family': 'courier', 'marginTop': -10, 'marginBottom': 10}),
            ], class_name=row_class_name),  # help text
            dbc.Row([
                dbc.Col([
                    dbc.Stack([
                        dbc.Button('Delete selected', id='btn-asassn-delete'),
                        dbc.Button('Unselect', id='btn-asassn-unselect'),
                    ], direction='horizontal', gap=2),  # Select points/unselect
                ], md=6, sm=12),  # Deal with selection
                dbc.Col([
                    dbc.Stack([
                        dbc.Select(options=CurveDash.get_format_list(),
                                   value=CurveDash.get_format_list()[0],
                                   id='select-asassn-format'),
                        dbc.Button('Download', id='btn-asassn-download-lc'),
                    ], direction='horizontal', gap=2)
                ], md=6, sm=12),  # select a format

            ], class_name='row_class_name'),  # Download and clean by hand
            dcc.Download(id='download-asassn-lc'),
        ], class_name='row_class_name', style={'display': 'none'}),  # The rest of the layout
        # ], id='row-asassn-content', style={'display': 'none'}),  # The rest of the layout
        html.Div(id='div-asassn-alert', style={'display': 'none'})
    ]),
    return res


# def _set_folded_view(folded_view: int, jdict: dict):
#     jdict['folded_view'] = folded_view


def _load_lightcurve(source_id: str, band: str, force_update=False) -> CurveDash:
    gaia_id = decipher_source_id(source_id)  # M.b. long remote call. Or m.b. not
    lcd = load_asassn_lightcurve(gaia_id, band, force_update)
    return lcd


@callback(
    output=dict(
        header=Output('h1-asassn', 'children'),
        row_content_style=Output('row-asassn-content', 'style'),
        div_alert_style=Output('div-asassn-alert', 'style'),
        alert_message=Output('div-asassn-alert', 'children'),
        switch_asassn_style=Output('switch-asassn-view', 'style'),
        warning_asassn_style=Output('label-switch-asassn-view-warning', 'style'),
        lc=Output('store_asassn_lightcurve', 'data'),
    ),
    inputs=dict(
        _1=Input('input-asassn-source-id', 'n_submit'),
        _2=Input('btn-asassn-new-source', 'n_clicks'),
        _3=Input('btn-asassn-update', 'n_clicks'),
    ),
    state=dict(
        source_id=State('input-asassn-source-id', 'value'),
        band=State('select-asassn-band', 'value'),
        phase_view=State('switch-asassn-view', 'value'),
    ),
    running=[(Output('btn-asassn-update', 'disabled'), True, False),
             (Output('btn-cancel-asassn-update', 'disabled'), False, True),
             (Output('btn-asassn-new-source', 'disabled'), True, False)],
    cancel=[Input('btn-cancel-asassn-update', 'n_clicks')],
    background=True,
    prevent_initial_call=True
)
def load_new_source(_1, _2, _3, source_id, band, phase_view):
    # folded_view = 1 if phase_view else 0
    switch_asassn_style = {'display': 'block'}
    warning_asassn_style = {'display': 'none'}

    # todo: Ensure we can load the source absent in our Gaia VEB database
    if source_id is None or source_id == '':
        raise PreventUpdate
    title = 'ASAS-SN lightcurve'
    prefix = 'GAIA DR3' if is_like_gaia_id(source_id) else ''
    header_txt = html.Span([f'{title} {prefix} {source_id}  ', html.Em(band)])
    try:
        logging.info(f'Load source data from asas-sn db: {source_id=}')
        if ctx.triggered_id == 'btn-asassn-update':
            force_update = True
        else:
            force_update = False
        lcd = _load_lightcurve(source_id, band=band, force_update=force_update)
        lcd.folded_view = phase_view

        # jdict = handler.load_lightcurve(source_id, band, catalogue, force_update)
        period = lcd.period
        period = None if not period else round(period, 5)
        epoch = lcd.epoch
        period_unit = lcd.period_unit
        content_style = {'display': 'block'}
        alert_style = {'display': 'none'}
        alert_message = ''
        # if period:
        #     header_txt += f' P={period}'
        # if period_unit:
        #     header_txt += f' {period_unit}'
        if period is None:
            switch_asassn_style = {'display': 'none'}
            warning_asassn_style = {'display': 'block'}
            lcd.folded_view = 0

        lc = lcd.serialize()

    except Exception as e:
        content_style = {'display': 'none'}
        alert_style = {'display': 'block'}
        alert_message = message.warning_alert(e)
        lc = no_update

    output = dict(header=header_txt, row_content_style=content_style,
                  div_alert_style=alert_style, alert_message=alert_message,
                  switch_asassn_style=switch_asassn_style,
                  warning_asassn_style=warning_asassn_style,
                  lc=lc)
    return output


clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='updateFoldedView'
    ),
    Output('store_asassn_lightcurve', 'data', allow_duplicate=True),
    Input('switch-asassn-view', 'value'),
    State('store_asassn_lightcurve', 'data'),
    prevent_initial_call=True
)

clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='selectData'
    ),
    Output('store_asassn_lightcurve', 'data', allow_duplicate=True),
    Input('graph-asassn-curve', 'selectedData'),
    Input('graph-asassn-curve', 'clickData'),
    State('store_asassn_lightcurve', 'data'),
    prevent_initial_call=True
)

clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='plotLightcurveFromStore'
    ),
    Output('graph-asassn-curve', 'figure'),
    Input('store_asassn_lightcurve', 'data'),
    State('graph-asassn-curve', 'figure'),
    prevent_initial_call=True
)

clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='unselectData'
    ),
    Output('store_asassn_lightcurve', 'data', allow_duplicate=True),
    Input('btn-asassn-unselect', 'n_clicks'),
    State('store_asassn_lightcurve', 'data'),
    prevent_initial_call=True
)

clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='deleteSelected'
    ),
    Output('store_asassn_lightcurve', 'data', allow_duplicate=True),
    Input('btn-asassn-delete', 'n_clicks'),
    State('store_asassn_lightcurve', 'data'),
    prevent_initial_call=True
)

clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='clearInput'
    ),
    Output('input-asassn-source-id', 'value'),
    Input('btn-asassn-clear-source_id', 'n_clicks'),
    prevent_initial_call=True
)


@callback(Output('download-asassn-lc', 'data'),  # ------ Download -----
          Input('btn-asassn-download-lc', 'n_clicks'),
          State('store_asassn_lightcurve', 'data'),
          # State('store-asassn-metadata', 'data'),
          State('select-asassn-format', 'value'),
          prevent_initial_call=True)
def download_asassn_lc(_, js_lightcurve, table_format):
    if js_lightcurve is None:
        raise PreventUpdate
    lcd = CurveDash.from_serialized(js_lightcurve)
    # bstring is "bytes"
    file_bstring = lcd.download(table_format)
    outfile_base = f'lc_asassn_{lcd.gaia_id}_{lcd.band}'.replace(' ', '_')
    ext = lcd.get_file_extension(table_format)
    outfile = f'{outfile_base}.{ext}'

    # filename, file_bstring = handler.prepare_download(js_lightcurve, js_metadata, table_format=table_format)
    ret = dcc.send_bytes(file_bstring, outfile)
    return ret
