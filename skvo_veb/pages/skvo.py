import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
# import dash_aladin_lite
from skvo_veb.utils.my_tools import timeit

# Gaia Eclipsing Binary Catalog - IGEBC
dash.register_page(__name__, name='SKVO',
                   title='SKVO home',
                   description='SKVO Home Page',
                   in_navbar=False,
                   path='/')


@timeit
def layout():
    return dbc.Container([
        html.H1('SKVO - Slovak Virtual Observatory', className="text-primary text-left fs-3"),
        html.Br(),
        html.Div([
            # html.H3('The page is under construction'),
            html.Img(src=dash.get_asset_url('under_constriction.jpeg')),
            html.Br(), html.Br(), html.Br(),
            dcc.Markdown('IGEBC - Interactive Gaia Eclipsing Binary Catalog is [here](/igebc)')
        ], style={'textAlign': 'center'}),
        html.Div(
            [
                # dash_aladin_lite.DashAladinLite(target='AA And'),
            ]
        )
    ], className="g-0", fluid=True)
