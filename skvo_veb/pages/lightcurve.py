import logging

from dash import register_page, html, dcc, callback, ctx, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs
import plotly.express as px
# import plotly.graph_objects as go
import dash_bootstrap_components as dbc
# from imageio.config.plugins import

from skvo_veb.components import message
from skvo_veb.utils import handler
from skvo_veb.utils.my_tools import PipeException, is_like_gaia_id

register_page(__name__, name='Lightcurve',
              order=2,
              path='/igebc/lc',
              title='IGEBC: Gaia Lightcurve',
              in_navbar=False)

row_class_name = "d-flex g-2 justify-content-end align-items-end"


def layout(source_id=None, catalogue='Gaia', band='G'):
    if source_id is None:
        header_txt = 'Request lightcurve'
        # header = html.H1('Request lightcurve', className="text-primary text-left fs-3")
    else:
        # header = html.H1(f'{source_id} {band}', className="text-primary text-left fs-3")
        # header = html.H1(f'{source_id}', className="text-primary text-left fs-3")
        header_txt = f'{source_id}'
    header = html.H1(header_txt, id='h1', className='text-primary text-left fs-3')
    res = dbc.Container([
        dcc.Store(id='store-lightcurve'),
        dcc.Store(id='store-metadata'),
        header,
        # html.H1('Request lightcurve', className="text-primary text-left fs-3"),
        # dbc.Row([
        #     dbc.Col([
        #         dcc.Markdown('Select Band'),
        #     ], md=3),  # Gaia ID
        #     dbc.Col([
        #         dbc.Select(options=['G', 'Bp', 'Rp'], value=band, id='select-band',
        #                    # persistence=True,
        #                    style={'width': 100}),
        #     ]),
        #     dbc.Col([
        #         # dbc.Button('Submit', color='primary', size="sm", id='btn-new-source'),
        #         dbc.Stack([
        #             dbc.Button('Submit', color='primary', size="md", id='btn-new-source'),
        #             dbc.Button('Clear', size="md", class_name='me-3', color='light', id='btn-clear-source_id'),
        #         ], direction="horizontal", gap=2),
        #     ], md=2, align='end'),
        #     # dbc.Col([], md=6),
        #
        # ], class_name='row_class_name'),
        dbc.Row([
            dbc.Col([
                dbc.Stack([
                    dcc.Markdown('Name'),
                    dbc.Input(placeholder='type object name', value=source_id if source_id is not None else '',
                              type='text', id='input-source-id',
                              # persistence=True
                              ),
                ], direction="horizontal", gap=2),
            ], md=5),  # Gaia ID
            dbc.Col([
                dbc.Stack([
                    dcc.Markdown('Band'),
                    dbc.Select(options=['G', 'Bp', 'Rp'], value=band, id='select-band',
                               # persistence=True,
                               style={'width': 100}),
                ], direction="horizontal", gap=2),
            ], md=2),  # Select Band
            #  I'm here
            dbc.Col([
                # dbc.Button('Submit', color='primary', size="sm", id='btn-new-source'),
                dbc.Stack([
                    dbc.Button('Submit', color='primary', size="md", id='btn-new-source'),
                    dbc.Button('Clear', size="md", class_name='me-3', color='light', id='btn-clear-source_id'),
                ], direction="horizontal", gap=2),
            ], md=2, align='end'),
            # dbc.Col([], md=6),

        ], class_name='row_class_name'),  # class_name="g-2"),  # Select new star and band
        html.Br(),
        dbc.Row(id='row-content', class_name='row_class_name'),  # The rest of the layout
    ]),
    return res


@callback(Output('h1', 'children'),
          Output('row-content', 'children'),
          Output('store-lightcurve', 'data'),
          Output('store-metadata', 'data'),  # todo Remove it!!!
          Input('input-source-id', 'n_submit'),
          Input('btn-new-source', 'n_clicks'),
          State('input-source-id', 'value'),
          State('select-band', 'value'),
          # prevent_initial_call=True
          )
def load_new_source(_1, _2, source_id, band):
    catalogue = 'Gaia'
    if source_id is None or source_id == '':
        raise PreventUpdate
    jdict = {'lightcurve': {}, 'metadata': {}}  # todo Fix it!!! Raise PreventUpdate exception if not found
    prefix = 'GAIA DR3' if is_like_gaia_id(source_id) else ''
    header_txt = f'{prefix} {source_id} {band=}'
    try:
        logging.info(f'Load source data from db: {source_id} from {catalogue}')
        jdict = handler.load_lightcurve(source_id, band, catalogue)
        epoch_gaia = jdict['metadata']['epoch_gaia']
        # epoch_new = jdict['metadata']['epoch_new']
        period = jdict['metadata']['period']
        period = None if not period else round(period, 5)
        period_unit = jdict['metadata']['period_unit']
        # epoch = epoch_new if epoch_new is not None else epoch_gaia
        if period:
            header_txt += f' P={period}'
        if period_unit:
            header_txt += f' {period_unit}'

        content = [
            # html.H1(title, className="text-primary text-left fs-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Checklist(options=[{'label': 'Folded view', 'value': 1}], value=0, id='switch-view',
                                  persistence=True, switch=True),
                ]),  # switch phase view
                # dbc.Col([
                #     dbc.RadioItems(options=[{'label': 'New epoch', 'value': epoch_new},
                #                             {'label': 'Gaia epoch', 'value': epoch_gaia}],
                #                    value=epoch,
                #                    id='switch-epoch', inline=True) if epoch_new is not None else None
                # ]),  # switch epoch
            ], class_name="g-2"),  # Switch view/epoch
            dbc.Row([  # layout={margin=dict(l=0, b=20)}
                dbc.Col([
                    dbc.Row([
                        dcc.Graph(id='graph-curve',
                                  # figure=go.Figure(layout={'margin': {'l': 0, 'r': 0, 'b': 20}}),
                                  figure=px.scatter(),
                                  config={'displaylogo': False,
                                          # 'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
                                          # https://community.plotly.com/t/is-it-possible-to-hide-the-floating-toolbar/4911/7
                                          }),
                    ], class_name="g-2"),  # Graph
                ], md=12, sm=12),  # width={'size': 8, 'offset': 0, 'order': 1}),  # lightcurve Graph
            ], class_name="g-0"),  # g-0 -- Row without 'gutters'       # Lightcurve stuff
            dbc.Row([
                dcc.Markdown('_**Click on a point to select it, or use Lasso or Box selector**_',
                             style={"font-size": 14, 'font-family': 'courier', 'marginTop': -10, 'marginBottom': 10
                                    # style={"font-size": 14, 'font-family': 'courier', 'marginBottom': -10
                                    }),  # 'marginTop': 20, }),
            ], class_name=row_class_name),  # help text
            # dbc.Row([
            #     dbc.Stack([
            #
            #     ], direction='horizontal'),
            #
            # ]),
            dbc.Row([
                # dbc.Col([
                #     dbc.Stack([
                #         dbc.Row([
                #             dbc.Col([
                #                 # dbc.Stack([
                #                 dcc.Markdown('Epoch, jd'),
                #                 # dbc.Input(type='number', value=epoch, min=0, step=0.01, id='inp-epoch'),
                #                 dbc.Input(type='number', min=0, value=epoch, id='inp-epoch'),
                #                 dcc.Markdown(f'Period, {period_unit}'),
                #                 dbc.Input(type='number', value=period, min=0, step=0.0001, id='inp-period'),
                #                 # ]),
                #
                #             ], md=6, sm=6),  # epoch
                #             dbc.Col([
                #                 # dbc.Stack([
                #                 dcc.Markdown('step, d'),
                #                 dbc.Input(value=0.0001, type='number', min=0, step=0.0001, id='inp-epoch-step'),
                #                 dcc.Markdown(f'step, {period_unit}'),
                #                 dbc.Input(value=0.0001, type='number', min=0, step=0.0001, id='inp-period-step'),
                #                 # ]),
                #             ], md=4, sm=4),  # epoch step
                #         ], class_name="g-2"),  # Epoch
                #
                #     ], gap=2),
                # ], md=5, sm=12),  # Lightcurve view tools
                # dbc.Col([
                #     dbc.Stack([
                #         dbc.Row([
                #             dbc.Col([
                #                 dbc.Stack([
                #                     dcc.Markdown('$\sigma_{low}$', mathjax=True),
                #                     dbc.Input(value=5.0, type='number', min=0, id='inp-autoclean-sigma-lo'),
                #                 ]),
                #             ]),  # sigma_low
                #             dbc.Col([
                #                 dbc.Stack([
                #                     dcc.Markdown('$\sigma_{up}$', mathjax=True),
                #                     dbc.Input(value=5.0, type='number', min=0, id='inp-autoclean-sigma-up'),
                #                 ]),
                #             ]),  # sigma up
                #             dbc.Col([
                #                 # html.Div([  # align right  https://getbootstrap.com/docs/5.0/utilities/flex/
                #                 dbc.Button('Clean', color='primary',
                #                            size="md", id='btn-autoclean'),
                #             ]),  # button clean
                #         ], class_name=row_class_name),  # autoclean by sig_low/sig_up
                #         dbc.Row([
                #             dbc.Col([
                #                 dcc.Markdown('$err_{max}$', mathjax=True),
                #             ]),  # label
                #             dbc.Col([
                #                 dbc.Input(value=20, type='number', id='inp-clean-flux_err'),
                #             ]),  # input
                #             dbc.Col([
                #                 dbc.Button('Clean', color='primary', size="md",
                #                            id='btn-clean-flux_err'),
                #             ]),  # button
                #         ], class_name=row_class_name),  # clean by flux_err
                #     ], gap=2),
                # ], md=4, sm=12),  # width={'size': 4, 'offset': 0, 'order': 2}),  # lightcurve tools
                dbc.Col([
                    dbc.Stack([
                        dbc.Button('Delete selected', id='btn-delete'),
                        dbc.Button('Unselect', id='btn-unselect'),
                    ], direction='horizontal', gap=2),  # Select points/unselect
                ], md=6, sm=12),  # Deal with selection

                dbc.Col([
                    dbc.Stack([
                        dbc.Select(options=handler.get_format_list(), value=handler.get_format_list()[0],
                                   id='select-format'),
                        dbc.Button('Download', id='btn-download-lc'),
                    ], direction='horizontal', gap=2)
                ], md=6, sm=12),  # select a format

            ], class_name='row_class_name'),  # Download and clean by hand
            dcc.Download(id='download-lc'),
        ]  # lightcurve layout
    except Exception as e:
        # content = dbc.Container([
        content = [
            message.warning_alert(e),
        ]

    return header_txt, content, handler.serialise(jdict['lightcurve']), handler.serialise(jdict['metadata'])


@callback(Output('graph-curve', 'figure'),
          Input('store-lightcurve', 'data'),
          Input('store-metadata', 'data'),
          # Input('inp-period', 'value'),
          # Input('inp-epoch', 'value'),
          Input('switch-view', 'value'),
          State('graph-curve', 'figure'),  # todo remove it
          prevent_initial_call=True)
# def plot_lc(js_lightcurve: str, period: float, epoch: float, phase_view: bool, fig):
def plot_lc(js_lightcurve: str, js_metadata: str, phase_view: bool, fig):
    """
    :param js_metadata: we extract period and epoch from this store
    :param js_lightcurve:  json string with lightcurve
    :param phase_view: True or False
    :param fig:
   :return:
    """
    logging.info(f'plot_lc {ctx.triggered_id=}')
    if ctx.triggered_id == 'switch-view':
        fig = px.scatter()  # reset figure, its zoom, etc
    try:
        # tab, curve_title, xaxis_title, yaxis_title = handler.prepare_plot(js_lightcurve, phase_view, period, epoch)
        # tab, xaxis_title, yaxis_title = handler.prepare_plot(js_lightcurve, phase_view, period, epoch)
        tab, xaxis_title, yaxis_title = handler.prepare_plot(js_lightcurve, phase_view, js_metadata)
        if len(tab) == 0:
            raise PipeException('Oops... Looks like we lost the lightcurve somewhere...')
        colors = ['sel' if c else 'ord' for c in tab['selected']]
        try:
            fig['data'][0]['x']  # try to touch a figure, is it empty?
            xaxis_range = fig['layout']['xaxis']['range']
            yaxis_range = fig['layout']['yaxis']['range']
        except (KeyError, IndexError):
            xaxis_range = None
            yaxis_range = None
        if phase_view:
            time_column = 'phase'
        else:
            time_column = 'jd'
        fig = px.scatter(data_frame=tab,  # title=curve_title,
                         x=time_column, y='flux', error_y='flux_err',
                         custom_data='perm_index',
                         # hover_name="flux",
                         # hover_data=None,
                         color=colors,  # whether selected for deletion 0/1
                         color_discrete_map={'ord': plotly.colors.qualitative.Plotly[0], 'sel': 'orange'},
                         )
        fig.update_traces(selected={'marker': {'color': 'lightgreen'}},
                          # hovertemplate='%{x}<br>%{y}',
                          hoverinfo='none',  # Important
                          hovertemplate=None,  # Important
                          )
        fig.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title,
                          showlegend=False,
                          margin=dict(l=0, b=20))  # r=50, t=50, b=20))
        if xaxis_range is not None and yaxis_range is not None:
            fig.update_xaxes(range=xaxis_range)
            fig.update_yaxes(range=yaxis_range)

    # except PipeException as e:
    except Exception as e:
        logging.warning(f'plot_lc exception {e}')
        fig = px.scatter()
    return fig


@callback(Output('store-lightcurve', 'data', allow_duplicate=True),  # ------------- Clean data
          Input('btn-autoclean', 'n_clicks'),
          Input('btn-clean-flux_err', 'n_clicks'),
          State('store-lightcurve', 'data'),
          State('inp-autoclean-sigma-lo', 'value'),
          State('inp-autoclean-sigma-up', 'value'),
          State('inp-clean-flux_err', 'value'),
          prevent_initial_call=True)
def clean_lightcurve(_1, _2, json_lc, sigma_lo, sigma_up, flux_err_max):
    try:
        if ctx.triggered_id == 'btn-autoclean':
            sigma = 5  # todo Remove this!
            json_lc = handler.autoclean(json_lc, sigma, sigma_lo, sigma_up)
        else:  # 'btn-clean-flux_err'
            json_lc = handler.clean_by_flux_err(json_lc, flux_err_max)
    except PipeException as e:
        logging.warning(f'_init_ clean_source: an PipeException occurred {repr(e)}')
        raise PreventUpdate
    return json_lc


@callback(Output('inp-epoch', 'value'),  # ------------ Epoch ----
          Input('switch-epoch', 'value'))
# prevent_initial_call=True)
def switch_epoch(epoch):
    return epoch


# @callback(Output('inp-period', 'step'),  # ----- Period step ----
#           Input('inp-period-step', 'value'),
#           )  # prevent_initial_call=True)
# def change_period_step(step):
#     return step


# @callback(Output('inp-epoch', 'step'),  # ----- Epoch step ----
#           Input('inp-epoch-step', 'value')
#           )  # prevent_initial_call=True)
# def change_epoch_step(step):
#     return step


@callback(Output('inp-clean-flux_err', 'value'),  # --- max flux err
          Input('store-lightcurve', 'data'))
def fill_flux_err_input(json_lc):
    return handler.suggest_flux_err_max(json_lc)


@callback(Output('store-lightcurve', 'data', allow_duplicate=True),
          Input('graph-curve', 'clickData'),
          # Input('graph-curve', 'n_clicks'),
          State('store-lightcurve', 'data'),
          prevent_initial_call=True)
def select_by_click(click_data, js_stored):
    if (js_stored is None) or (click_data is None):
        raise PreventUpdate
    selected_points = click_data.get('points', None)  # list of selected points. In this case, it has length=1
    if selected_points is None or len(selected_points) == 0:
        raise PreventUpdate
    # selected_point_indices = [p['pointNumber'] for p in selected_points]
    selected_point_indices = [p['customdata'][0] for p in selected_points]
    js = handler.mark_for_deletion(js_stored, selected_point_indices)
    return js


@callback(Output('download-lc', 'data'),  # ------ Download -----
          Input('btn-download-lc', 'n_clicks'),
          State('store-lightcurve', 'data'),
          State('store-metadata', 'data'),
          State('select-format', 'value'),
          prevent_initial_call=True)
def download_lc(_, js_lightcurve, js_metadata, table_format):
    if js_lightcurve is None:
        raise PreventUpdate
    # bstring is "bytes"
    filename, file_bstring = handler.prepare_download(js_lightcurve, js_metadata, table_format=table_format)
    # ret = dcc.send_string(bytes_io.getvalue(), filename)
    # ret = dcc.send_string(file_string, filename)
    ret = dcc.send_bytes(file_bstring, filename)
    return ret


@callback(Output('store-lightcurve', 'data', allow_duplicate=True),
          Input('graph-curve', 'selectedData'),
          State('store-lightcurve', 'data'),
          prevent_initial_call=True)
def clean_by_hand_select(select_data, js_stored):
    if (js_stored is None) or (select_data is None):
        raise PreventUpdate
    selected_points = select_data.get('points', None)  # list of selected points in lasso or rectangle
    if selected_points is None or len(selected_points) == 0:
        raise PreventUpdate
    selected_point_indices = [p['customdata'][0] for p in selected_points]
    js = handler.mark_for_deletion(js_stored, selected_point_indices)
    return js


@callback(Output('store-lightcurve', 'data', allow_duplicate=True),
          Input('btn-unselect', 'n_clicks'),
          State('store-lightcurve', 'data'),
          prevent_initial_call=True)
def unselect(_, js_stored):
    try:
        js_unselected = handler.unmark(js_stored)
    except PipeException:
        raise PreventUpdate
    return js_unselected


@callback(Output('store-lightcurve', 'data', allow_duplicate=True),
          Input('btn-delete', 'n_clicks'),
          State('store-lightcurve', 'data'),
          prevent_initial_call=True)
def delete_selected(_, js_stored):
    try:
        js_unselected = handler.delete_selected(js_stored)
    except PipeException:
        raise PreventUpdate
    return js_unselected


@callback(Output('input-source-id', 'value'),
          Input('btn-clear-source_id', 'n_clicks'),
          prevent_initial_call=True)
def clean(_):
    return None
