import logging

import dash
from dash import register_page, html, dcc, callback, clientside_callback, ClientsideFunction, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import dash_bootstrap_components as dbc

from skvo_veb.components import message
from skvo_veb.utils import request_gaia
from skvo_veb.utils.curve_dash import CurveDash
from skvo_veb.utils.my_tools import is_like_gaia_id
from skvo_veb.utils.request_gaia import decipher_source_id

register_page(__name__, name='Gaia Lightcurve',
              order=1,
              path='/igebc/gaia',
              title='IGEBC: Gaia Lightcurve',
              in_navbar=False)

row_class_name = "d-flex g-2 justify-content-end align-items-end"


def layout(source_id=None, band='G'):
    if source_id is None:
        header_txt = 'Request Gaia lightcurve'
    else:
        header_txt = f'Gaia lightcurve\n{source_id} {band}'
    # header = html.Div(id='h1-gaia', children=[html.H1(h1_txt, className='text-primary text-left fs-3'),
    #                                             html.H2(header_txt, className='text-primary text-left fs-3')])
    header = html.H1(header_txt, id='h1-gaia',
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
                      margin=dict(l=0, b=20),  # r=50, t=50, b=20))
                      dragmode='lasso'  # Enable lasso selection mode by default
                      )

    res = dbc.Container([
        dcc.Store(id='store_gaia_lightcurve'),
        # dcc.Store(id='store-gaia-metadata'),
        html.Br(),
        header,
        html.Br(),
        dbc.Row([
            dbc.Col([
                dbc.Stack([
                    dcc.Markdown('Name'),
                    dbc.Input(placeholder='type object name', value=source_id if source_id is not None else '',
                              type='text', id='input-gaia-source-id',
                              # persistence=True
                              ),
                ], direction="horizontal", gap=2),
            ], md=5),  # Gaia ID
            dbc.Col([
                dbc.Stack([
                    dcc.Markdown('Band'),
                    dbc.Select(options=['G', 'Bp', 'Rp'], value=band, id='select-gaia-band',
                               # persistence=True,
                               style={'width': 100}),
                ], direction="horizontal", gap=2),
            ], md=2),  # Select Band
            dbc.Col([
                dbc.Stack([
                    dbc.Button('Submit', color='primary', size="md", id='btn-gaia-new-source'),
                    dbc.Button('Clear', size="md", class_name='me-3', color='light', id='btn-gaia-clear-source_id'),
                ], direction="horizontal", gap=2),
            ], md=2, align='end'),

        ], class_name='row_class_name'),  # class_name="g-2"),  # Select new star and band
        html.Br(),
        # dbc.Row(id='row-gaia-content', class_name='row_class_name'),  # The rest of the layout
        dbc.Row(id='row-gaia-content', children=[
            dbc.Row([
                dbc.Col([
                    dbc.Checklist(options=[{'label': 'Folded view', 'value': 1}], value=0, id='switch-gaia-view',
                                  persistence=True, switch=True),
                ]),  # switch phase view
            ], class_name="g-2"),  # Switch view/epoch
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        dcc.Graph(id='graph-gaia-curve',
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
                        dbc.Button('Delete selected', id='btn-gaia-delete'),
                        dbc.Button('Unselect', id='btn-gaia-unselect'),
                    ], direction='horizontal', gap=2),  # Select points/unselect
                ], md=6, sm=12),  # Deal with selection
                dbc.Col([
                    dbc.Stack([
                        dbc.Select(options=CurveDash.get_format_list(),
                                   value=CurveDash.get_format_list()[0],
                                   id='select-gaia-format'),
                        dbc.Button('Download', id='btn-gaia-download-lc'),
                    ], direction='horizontal', gap=2)
                ], md=6, sm=12),  # select a format

            ], class_name='row_class_name'),  # Download and clean by hand
            dcc.Download(id='download-gaia-lc'),
        ], class_name='row_class_name', style={'display': 'none'}),  # The rest of the layout
        html.Div(id='div-gaia-alert', style={'display': 'none'})
    ]),
    return res


# def _set_folded_view(folded_view: int, jdict: dict):
#     jdict['folded_view'] = folded_view


def _load_lightcurve(source_id: str, band: str) -> CurveDash:
    gaia_id = decipher_source_id(source_id)  # M.b. long remote call. Or m.b. not
    lcd = request_gaia.load_gaia_lightcurve(gaia_id, band)
    return lcd


@callback(
    output=dict(
        header=Output('h1-gaia', 'children'),
        row_content_style=Output('row-gaia-content', 'style'),
        div_alert_style=Output('div-gaia-alert', 'style'),
        alert_message=Output('div-gaia-alert', 'children'),
        lc=Output('store_gaia_lightcurve', 'data'),
    ),
    inputs=dict(
        _1=Input('input-gaia-source-id', 'n_submit'),
        _2=Input('btn-gaia-new-source', 'n_clicks'),
    ),
    state=dict(
        source_id=State('input-gaia-source-id', 'value'),
        band=State('select-gaia-band', 'value'),
        phase_view=State('switch-gaia-view', 'value'),
    )
)
def load_new_source(_1, _2, source_id, band, phase_view):
    folded_view = 1 if phase_view else 0

    if source_id is None or source_id == '':
        raise PreventUpdate
    # jdict = {'lightcurve': {}, 'metadata': {}}
    prefix = 'GAIA DR3' if is_like_gaia_id(source_id) else ''
    # header_txt = f'{prefix} {source_id} {band}'
    title = 'Gaia lightcurve'
    header_txt = html.Span([f'{title} {prefix} {source_id}  ', html.Em(band)])
    try:
        logging.info(f'Load source data from gaia db: {source_id=}')
        lcd = _load_lightcurve(source_id, band)
        lcd.folded_view = folded_view
        # epoch = jdict['metadata']['epoch_gaia']
        epoch = lcd.epoch
        # period = jdict['metadata']['period']
        period = lcd.period
        # period_unit = jdict['metadata']['period_unit']
        period_unit = lcd.period_unit
        period = None if not period else round(period, 5)
        if period is None:
            lcd.folded_view = 0
        content_style = {'display': 'block'}
        alert_style = {'display': 'none'}
        alert_message = ''
        lc = lcd.serialize()
        # if period:
        #     header_txt += f' P={period}'
        # if period_unit:
        #     header_txt += f' {period_unit}'
    except Exception as e:
        content_style = {'display': 'none'}
        alert_style = {'display': 'block'}
        alert_message = message.warning_alert(e)
        lc = dash.no_update

    # header=Output('h1-gaia', 'children'),
    #     row_content=Output('row-gaia-content', 'style'),
    #     div_alert_style=Output('div-gaia-alert', 'style'),
    #     alert_message=Output('div-gaia-alert', 'children'),
    #     lc=Output(
    output = dict(header=header_txt, row_content_style=content_style,
                  div_alert_style=alert_style, alert_message=alert_message,
                  lc=lc)
    # return (header_txt, content_style, alert_style, alert_message,
    #         handler.serialise(jdict['lightcurve']),
    #         handler.serialise(jdict['metadata']))
    return output


clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='updateFoldedView'
    ),
    Output('store_gaia_lightcurve', 'data', allow_duplicate=True),
    Input('switch-gaia-view', 'value'),
    State('store_gaia_lightcurve', 'data'),
    prevent_initial_call=True
)

clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='selectData'
    ),
    Output('store_gaia_lightcurve', 'data', allow_duplicate=True),
    Input('graph-gaia-curve', 'selectedData'),
    Input('graph-gaia-curve', 'clickData'),
    State('store_gaia_lightcurve', 'data'),
    prevent_initial_call=True
)

clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='plotLightcurveFromStore'
    ),
    Output('graph-gaia-curve', 'figure'),
    Input('store_gaia_lightcurve', 'data'),
    State('graph-gaia-curve', 'figure'),
    prevent_initial_call=True
)

clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='unselectData'
    ),
    Output('store_gaia_lightcurve', 'data', allow_duplicate=True),
    Input('btn-gaia-unselect', 'n_clicks'),
    State('store_gaia_lightcurve', 'data'),
    prevent_initial_call=True
)

clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='deleteSelected'
    ),
    Output('store_gaia_lightcurve', 'data', allow_duplicate=True),
    Input('btn-gaia-delete', 'n_clicks'),
    State('store_gaia_lightcurve', 'data'),
    prevent_initial_call=True
)

clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='clearInput'
    ),
    Output('input-gaia-source-id', 'value'),
    Input('btn-gaia-clear-source_id', 'n_clicks'),
    prevent_initial_call=True
)


@callback(Output('download-gaia-lc', 'data'),  # ------ Download -----
          Input('btn-gaia-download-lc', 'n_clicks'),
          State('store_gaia_lightcurve', 'data'),
          State('select-gaia-format', 'value'),
          prevent_initial_call=True)
def download_gaia_lc(_, js_lightcurve, table_format):
    if js_lightcurve is None:
        raise PreventUpdate
    lcd = CurveDash(js_lightcurve)
    # bstring is "bytes"
    file_bstring = lcd.download(table_format)
    outfile_base = f'lc_gaia_{lcd.gaia_id}_{lcd.band}'.replace(' ', '_')
    ext = lcd.get_file_extension(table_format)
    outfile = f'{outfile_base}.{ext}'

    # filename, file_bstring = handler.prepare_download(js_lightcurve, js_metadata, table_format=table_format)
    ret = dcc.send_bytes(file_bstring, outfile)
    return ret





# clientside_callback(
#     "clientside.clearInput",
#     # """
#     # function(_, inputValue) {
#     #     return null;  // clear Input
#     # }
#     # """,
#     Output('input-gaia-source-id', 'value'),
#     Input('btn-gaia-clear-source_id', 'n_clicks'),
#     prevent_initial_call=True
# )

