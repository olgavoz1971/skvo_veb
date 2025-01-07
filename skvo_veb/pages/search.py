import dash
from dash import html, dcc, callback, Input, Output, State, clientside_callback
import dash_bootstrap_components as dbc
from skvo_veb.utils.my_tools import timeit
from skvo_veb.utils.coord import is_it_coord

dash.register_page(__name__, name='Search',
                   order=1,
                   title='IGEBC: search',
                   description='Search through local DataBase description',
                   in_navbar=True,
                   path='/igebc/search')

query_id_coord = dbc.Stack([
    dbc.Row([
        dbc.Col([
            dbc.Stack([
                dbc.Label('Name or Coordinates'),
                dbc.Input(placeholder='object name or coordinates', type='text', id='inp-id-coord', persistence=True),
            ]),
        ], md=3, sm=12),
        dbc.Col([
            dbc.Stack([
                dbc.Label('Radius [arc min]'),
                dbc.Input(value=10, type='number', min=0, id='inp-radius', persistence=True),
            ]),
        ], md=2, sm=12),
        dbc.Col([
            dcc.Markdown(
                '_**Gaia DR3 Id or Simbad name or ICRS ra,dec**_:\n'
                '* 5284186916701857536\n'
                '* Gaia DR3 1000119251255360896\n'
                '* AB And\n'
                '* V* SS Ari\n'
                '* 20 54 05.689 +37 01 17.38\n'
                '* 10:12:45.3-45:17:50\n'
                '* 15h17m-11d10m\n'
                '* 12h -17d\n'
                '* 350.1d-17.3d\n',
                style={"font-size": 12, 'font-family': 'courier'}
            ),
        ], md=4, sm=12),
    ]),
    dbc.Row([
        dbc.Stack([
            dbc.Button('Submit', size="sm", id='btn-submit-id-coord'),
            dbc.Button('Clear', size="sm", color='light', id='btn-clear-id-coord'),
        ], direction="horizontal", gap=2),
    ]),
], gap=2)  # coordinates or name query


@timeit
def layout():
    return dbc.Container([
        dcc.Location(id='location-search'),
        html.H1('Request source', className="text-primary text-left fs-3"),
        html.Br(),
        dbc.Row([query_id_coord]),
        # accordion_query,
    ], className="g-0", fluid=True)


@callback(Output('location-search', 'href', allow_duplicate=True),
          Input('btn-submit-id-coord', 'n_clicks'),
          Input('inp-id-coord', 'n_submit'),
          State('inp-id-coord', 'value'),
          State('inp-radius', 'value'),
          prevent_initial_call=True)
def handle_input(_1, _2, id_coord_str, radius):
    if is_it_coord(id_coord_str):
        return f'/igebc/coo?coords={id_coord_str}&radius={radius}'
    return f'/igebc/star?source_id={id_coord_str}'
    # return f'/igebc/star?source_id={id_coord_str}&catalogue={catalogue}'


# @callback(Output('inp-id-coord', 'value'),
#           Input('btn-clear-id-coord', 'n_clicks'),
#           prevent_initial_call=True)
# def clean(_):
#     return None


clientside_callback(
    """
    function(_, inputValue) {
        return null;  // clear Input
    }
    """,
    Output('inp-id-coord', 'value'),
    Input('btn-clear-id-coord', 'n_clicks'),
    prevent_initial_call=True
)
