import logging

from dash import register_page, html, dcc, callback, clientside_callback, Input, Output, State
from dash.exceptions import PreventUpdate
from plotly.colors import qualitative as qq
import plotly.express as px
import dash_bootstrap_components as dbc

from skvo_veb.components import message
from skvo_veb.utils import handler
from skvo_veb.utils.my_tools import is_like_gaia_id

register_page(__name__, name='Gaia Lightcurve',
              order=2,
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
        dcc.Store(id='store-gaia-lightcurve'),
        dcc.Store(id='store-gaia-metadata'),
        header,
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
                        dbc.Select(options=handler.get_format_list(), value=handler.get_format_list()[0],
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


def _set_folded_view(folded_view: int, jdict: dict):
    jdict['folded_view'] = folded_view


@callback(Output('h1-gaia', 'children'),
          Output('row-gaia-content', 'style'),
          Output('div-gaia-alert', 'style'),
          Output('div-gaia-alert', 'children'),
          Output('store-gaia-lightcurve', 'data'),
          Output('store-gaia-metadata', 'data'),  # todo Remove it!!!
          Input('input-gaia-source-id', 'n_submit'),
          Input('btn-gaia-new-source', 'n_clicks'),
          State('input-gaia-source-id', 'value'),
          State('select-gaia-band', 'value'),
          State('switch-gaia-view', 'value'),
          )
def load_new_source(_1, _2, source_id, band, phase_view):
    catalogue = 'Gaia'
    folded_view = 1 if phase_view else 0

    if source_id is None or source_id == '':
        raise PreventUpdate
    jdict = {'lightcurve': {}, 'metadata': {}}
    # prefix = 'GAIA lightcurve\nGAIA DR3' if is_like_gaia_id(source_id) else ''
    title = 'Gaia lightcurve'
    prefix = 'GAIA DR3' if is_like_gaia_id(source_id) else ''
    header_txt = html.Span([f'{title} {prefix} {source_id}  ', html.Em(band)])
    # header_txt = f'{prefix} {source_id} {band}'
    try:
        logging.info(f'Load source data from gaia db: {source_id=}')
        jdict = handler.load_lightcurve(source_id, band, catalogue)
        _set_folded_view(folded_view, jdict['lightcurve'])
        epoch = jdict['metadata']['epoch_gaia']
        period = jdict['metadata']['period']
        period_unit = jdict['metadata']['period_unit']
        period = None if not period else round(period, 5)
        content_style = {'display': 'block'}
        alert_style = {'display': 'none'}
        alert_message = ''
        # if period:
        #     header_txt += f' P={period}'
        # if period_unit:
        #     header_txt += f' {period_unit}'
    except Exception as e:
        content_style = {'display': 'none'}
        alert_style = {'display': 'block'}
        alert_message = message.warning_alert(e)
    return (header_txt, content_style, alert_style, alert_message,
            handler.serialise(jdict['lightcurve']),
            handler.serialise(jdict['metadata']))


clientside_callback(        # updateFoldedView
    """
    function updateFoldedView(phase_view, dataString) {
        console.log('Updating folded view based on phase_view:', phase_view);
        let jdict;
        try {
            // Parse the JSON data from the store
            jdict = JSON.parse(dataString);
            if (!jdict) {
                return window.dash_clientside.no_update; // Prevent update if jdict is empty
            }
            // Update folded_view based on the phase_view input
            // Convert phase_view to integer (1 if not empty, otherwise 0)
            console.log('phase_view =', phase_view);
            console.log('phase_view.length =', phase_view.length);
            const folded_view = phase_view.length > 0 ? 1 : 0; // Check if phase_view has any selected values
            console.log('folded_view =', folded_view);
            console.log('jdict =', jdict);
            jdict.folded_view = folded_view; // Update jdict with the new folded_view
            console.log('Updated jdict:', jdict);
            // Return the updated dictionary as a JSON string
            return JSON.stringify(jdict);
        } catch (error) {
            console.error('Error updating folded view:', error);
            return window.dash_clientside.no_update;
        }
    }
    """,
    Output('store-gaia-lightcurve', 'data', allow_duplicate=True),
    Input('switch-gaia-view', 'value'),
    State('store-gaia-lightcurve', 'data'),
    prevent_initial_call=True
)

# @callback(Output('graph-gaia-curve', 'figure'),
#           Input('store-gaia-lightcurve', 'data'),
#           Input('store-gaia-metadata', 'data'),
#           Input('switch-gaia-view', 'value'),
#           State('graph-gaia-curve', 'figure'),  # todo remove it
#           prevent_initial_call=True)
# def plot_lc(js_lightcurve: str, js_metadata: str, phase_view: bool, fig):
#     """
#     :param js_metadata: we extract period and epoch from this store
#     :param js_lightcurve:  json string with lightcurve
#     :param phase_view: True or False
#     :param fig:
#    :return:
#     """
#     logging.info(f'plot_lc gaia {ctx.triggered_id=}')
#     if not json.loads(js_lightcurve):
#         raise PreventUpdate
#     if ctx.triggered_id == 'switch-gaia-view':
#         fig = px.scatter()  # reset figure, its zoom, etc
#     try:
#         tab, xaxis_title, yaxis_title = handler.prepare_plot(js_lightcurve, phase_view, js_metadata)
#         if len(tab) == 0:
#             raise PipeException('Oops... Looks like we lost the lightcurve somewhere...')
#         colors = ['sel' if c else 'ord' for c in tab['selected']]
#         try:
#             fig['data'][0]['x']  # try to touch a figure, is it empty?
#             xaxis_range = fig['layout']['xaxis']['range']
#             yaxis_range = fig['layout']['yaxis']['range']
#         except (KeyError, IndexError):
#             xaxis_range = None
#             yaxis_range = None
#         if phase_view:
#             time_column = 'phase'
#         else:
#             time_column = 'jd'
#         fig = px.scatter(data_frame=tab,  # title=curve_title,
#                          x=time_column, y='flux', error_y='flux_err',
#                          custom_data='perm_index',
#                          # hover_name="flux",
#                          # hover_data=None,
#                          color=colors,  # whether selected for deletion 0/1
#                          color_discrete_map={'ord': qq.Plotly[0], 'sel': 'orange'},
#                          # color_discrete_map={'ord': plotly.colors.qualitative.Plotly[0], 'sel': 'orange'},
#                          )
#         fig.update_traces(selected={'marker': {'color': 'lightgreen'}},
#                           # hovertemplate='%{x}<br>%{y}',
#                           hoverinfo='none',  # Important
#                           hovertemplate=None,  # Important
#                           )
#         fig.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title,
#                           showlegend=False,
#                           margin=dict(l=0, b=20))  # r=50, t=50, b=20))
#         if xaxis_range is not None and yaxis_range is not None:
#             fig.update_xaxes(range=xaxis_range)
#             fig.update_yaxes(range=yaxis_range)
#
#     # except PipeException as e:
#     except Exception as e:
#         logging.warning(f'plot_lc gaia exception {e}')
#         fig = px.scatter()
#     return fig


clientside_callback(    # select_data
    """
    function select_data(selectedData, clickData, dataString) {
        console.log("select_data");

        try {
            // `dash_clientside.callback_context.triggered[0]` contains information on which input fired the callback.
            const triggered = dash_clientside.callback_context.triggered[0];

            if (!triggered) {
                // If nothing triggered the callback, return no update.
                console.log("False callback 1");
                return window.dash_clientside.no_update;
            }

            // Extracting the ID and property that triggered the callback.
            console.log(`triggered.prop_id=${triggered.prop_id}`);
            const [trigger_id, trigger_prop] = triggered.prop_id.split('.'); // Splitting "id.property"
            console.log(`trigger_prop=${trigger_prop}`);

            // Determine whether the event was triggered by `selectedData` or `clickData`.
            let triggerData;
            if (trigger_prop === 'selectedData') {
                // If lasso or box selection triggered the callback.
                triggerData = selectedData;
            } else {
                // Otherwise, it was a click event.
                triggerData = clickData;
            }

            // If there's no actual data in the event, skip further processing.
            if (!triggerData) {
                console.log("False callback 2");
                return window.dash_clientside.no_update;
            }

            // Parse the `dataString` which is expected to be a JSON string from dcc.Store
            // dataString has the structure you get with pandas.to_dict(orient='split', index=False)
            let data = JSON.parse(dataString);
            console.log("parsed data =", data);

            // Extract the columns and rows from the `data` (a table structure).
            const { columns, data: rows } = data.data;

            // Find the column index for the `perm_index` (unique identifier for rows).
            const permIndexColumn = columns.indexOf('perm_index');

            // Map perm_index to the row index for quick lookup.
            const permIndexMap = {};
            rows.forEach((row, index) => {
                const permIndex = row[permIndexColumn];
                permIndexMap[permIndex] = index;  // Create a mapping from perm_index to row index.
            });
            console.log("Map permIndexMap:", permIndexMap);

            // Get the indices of selected points from the `customdata` of the triggered event.
            console.log("triggerData =", triggerData);
            const selected_indices = triggerData.points.map(point => point.customdata[0]);
            console.log("selected_indices =", selected_indices);

            // For each selected index, mark the corresponding point as selected in the table data.
            selected_indices.forEach(index => {
                if (index in permIndexMap) {
                    const rowIndex = permIndexMap[index];  // Find row index from the permIndexMap.
                    rows[rowIndex][columns.indexOf('selected')] = 1;  // Set 'selected' flag to 1.
                    console.log(`Updated the row ${rowIndex}:`, rows[rowIndex]);
                }
            });

            // Log the modified data and return it as a JSON string to update the dcc.Store.
            console.log("Renewed data:", data);
            return JSON.stringify(data);

        } catch (error) {
            // In case of any error during execution, log the error and return no update.
            console.error("Error:", error.message);
            return window.dash_clientside.no_update;
        }
    }
    """,
    Output('store-gaia-lightcurve', 'data', allow_duplicate=True),
    Input('graph-gaia-curve', 'selectedData'),
    Input('graph-gaia-curve', 'clickData'),
    State('store-gaia-lightcurve', 'data'),
    prevent_initial_call=True
)

clientside_callback(    # plot_lightcurve_store
    """
    function plot_lightcurve_from_store(dataString, figure) {
        console.log('Updating figure from dcc.Store data');
        console.log('dataString =', dataString);

        try {
            // Parse the JSON string stored in dcc.Store into an object.
            let lightcurve = JSON.parse(dataString);
            // dataString has the structure you get with pandas.to_dict(orient='split', index=False)
            if (Object.keys(lightcurve).length === 0) {
                console.log('empty lightcurve');
                return window.dash_clientside.no_update;
            }
            console.log('lightcurve:', lightcurve);
            console.log('lightcurve.data:', lightcurve.data);
            const folded_view = lightcurve.folded_view; // Extract whether the folded view is active.
            console.log('folded_view =', folded_view)
            const { columns, data: rows } = lightcurve.data; // Extract columns and rows from the stored lightcurve.
            console.log('rows =', rows);
            
            // Based on the folded_view status, determine which column to use for the x-axis.
            let xColIndex;
            let newX, xaxis_title
            const yaxis_title = `flux, ${lightcurve.hasOwnProperty('flux_unit') ? lightcurve.flux_unit : ''}`;
            if (folded_view) {
                xColIndex = columns.indexOf('phase');  // Use 'phase' if folded view is active.
                newX = rows.map(row => row[xColIndex]);
                xaxis_title = `phase`;
                
            } else {
                xColIndex = columns.indexOf('jd');     // Otherwise, use 'jd' for time.
                const jd0 = 2450000;
                xaxis_title = `jd-${jd0}`;
                newX = rows.map(row => row[xColIndex] - jd0);  // Subtract jd0 from each value
            }

            // Extract the column indices for y-axis (flux), error (flux_err), customdata, and selection state.
            const yColIndex = columns.indexOf('flux');
            const yErrColIndex = columns.indexOf('flux_err');
            const customDataColIndex = columns.indexOf('perm_index');
            const selectedColIndex = columns.indexOf('selected');

            // Map the rows to extract the x (jd or phase), y (flux), error values, and customdata for each row.
            // const newX = rows.map(row => row[xColIndex]);
            const newY = rows.map(row => row[yColIndex]);            
            const newErrY = rows.map(row => row[yErrColIndex]);
            const newCustomData = rows.map(row => [row[customDataColIndex]]);

            // Determine which points are selected by checking where the 'selected' column equals 1.                        
            const selectedPoints = rows
                .map((row, index) => (row[selectedColIndex] === 1 ? index : null))
                .filter(index => index !== null);  // Filter out any null values (non-selected rows).
            
            console.log('selectedPoints =', selectedPoints);

            // Update the figure data:
            const newData = figure.data.map(trace => {
                return {
                    ...trace,
                    x: newX,          // Update x values (jd or phase).
                    y: newY,          // Update y values (flux).
                    error_y: { array: newErrY },  // Update error values.
                    customdata: newCustomData,    // Update customdata (perm_index).
                    selectedpoints: selectedPoints  // Highlight selected points.
                };
            });
            console.log('newData =', newData);

            // Copy the figure layout and remove any selections (e.g., lasso or box selection paths).
            const newLayout = { ...figure.layout,
                                xaxis: { ...figure.layout.xaxis, title: xaxis_title},  
                                yaxis: { ...figure.layout.yaxis, title: yaxis_title}
                              };            
            delete newLayout.selections;        // remove lasso or box path

            // Return the updated figure with the new data and layout.
            const newFigure = { ...figure, data: newData, layout: newLayout };
            console.log('newLayout:', newLayout);
            return newFigure;

        } catch (error) {
            // If there's an error, log it and prevent any update to the figure.
            console.error("Error:", error.message);
            return window.dash_clientside.no_update;
        }
    }
    """,
    Output('graph-gaia-curve', 'figure'),
    Input('store-gaia-lightcurve', 'data'),
    State('graph-gaia-curve', 'figure'),
    prevent_initial_call=True
)

clientside_callback(  # unselect
    """
    function unselect(_, js_stored) {
        console.log("unselect");
        try {
            if (!js_stored) {
                console.log("No data to unselect");
                return window.dash_clientside.no_update;
            }

            let data = JSON.parse(js_stored);
            console.log("parsed data for unselect =", data);
            const { columns, data: rows } = data.data;

            const selectedColumnIndex = columns.indexOf('selected');
            if (selectedColumnIndex === -1) {
                console.error("'selected' column does not exist");
                return window.dash_clientside.no_update;
            }

            rows.forEach(row => {
                row[selectedColumnIndex] = 0;  // mark point as unselected
            });

            console.log("All points unselected:", data);
            return JSON.stringify(data);
        } catch (error) {
            console.error("Error in unselect:", error.message);
            return window.dash_clientside.no_update;
        }
    }
    """,
    Output('store-gaia-lightcurve', 'data', allow_duplicate=True),
    Input('btn-gaia-unselect', 'n_clicks'),
    State('store-gaia-lightcurve', 'data'),
    prevent_initial_call=True
)

clientside_callback(  # delete_selected
    """
    function delete_selected(deleteClick, dataString) {
        console.log("Delete selected points");
        try {
            if (!deleteClick) {
                return window.dash_clientside.no_update;
            }

            // Parse the data from dcc.Store
            let data = JSON.parse(dataString);
            const { columns, data: rows } = data.data;
            const selectedColIndex = columns.indexOf('selected');

            // Filter out rows where selected is 1
            const newRows = rows.filter(row => row[selectedColIndex] !== 1);

            // Update the data in dcc.Store
            data.data.data = newRows;

            console.log("Updated data (after deletion):", data);
            return JSON.stringify(data);
        } catch (error) {
            console.error("Error in deleting selected points:", error.message);
            return window.dash_clientside.no_update;
        }
    }
    """,
    Output('store-gaia-lightcurve', 'data', allow_duplicate=True),
    Input('btn-gaia-delete', 'n_clicks'),
    State('store-gaia-lightcurve', 'data'),
    prevent_initial_call=True
)


@callback(Output('download-gaia-lc', 'data'),  # ------ Download -----
          Input('btn-gaia-download-lc', 'n_clicks'),
          State('store-gaia-lightcurve', 'data'),
          State('store-gaia-metadata', 'data'),
          State('select-gaia-format', 'value'),
          prevent_initial_call=True)
def download_lc(_, js_lightcurve, js_metadata, table_format):
    if js_lightcurve is None:
        raise PreventUpdate
    # bstring is "bytes"
    filename, file_bstring = handler.prepare_download(js_lightcurve, js_metadata, table_format=table_format)
    ret = dcc.send_bytes(file_bstring, filename)
    return ret


clientside_callback(
    """
    function(_, inputValue) {
        return null;  // clear Input
    }
    """,
    Output('input-gaia-source-id', 'value'),
    Input('btn-gaia-clear-source_id', 'n_clicks'),
    prevent_initial_call=True
)

# @callback(Output('input-gaia-source-id', 'value'),
#           Input('btn-gaia-clear-source_id', 'n_clicks'),
#           prevent_initial_call=True)
# def clean(_):
#     return None
