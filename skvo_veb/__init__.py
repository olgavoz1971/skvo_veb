from dash import Dash, dcc  # html, dcc, callback, Input, Output
import dash
import dash_bootstrap_components as dbc
import flask
from skvo_veb.components import footer
import logging

logging.basicConfig(level=logging.DEBUG)

server = flask.Flask(__name__)


@server.route("/")
def home():
    return app.index()


app = Dash(__name__, server=server, use_pages=True, suppress_callback_exceptions=True)
# todo replace it with app.validation_layout
app.title = 'Gaia VEB lightcurves Dashboard'

app.layout = dbc.Container([
    dbc.Row([
        dbc.NavbarSimple([
            dbc.NavItem(dbc.NavLink(page['name'], href=page['relative_path']))
            for page in dash.page_registry.values() if page.get('in_navbar', False)
        ], brand='VEB Gaia',  # brand_href="#", color="primary",
            light=True, fluid=True, className="w-100",  # to make sure it is the full width of the Row
        ),
    ], className="flex-grow-1"  # to make sure it expands to fill the available horizontal space
    ),

    # html.H1('Gaia VEB Multi-page dash app', className="text-primary text-center fs-3"),
    dash.page_container,
    # dcc.Loading(
    #     id='page-loading-indicator',
    #     type="circle",
    #     children=dash.page_container,
    # ),
    dbc.Row(
        footer.footer,
    ),
], fluid=True)

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8051)
