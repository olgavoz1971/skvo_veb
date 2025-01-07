from dash import Dash, DiskcacheManager  # dcc, html, dcc, callback, Input, Output
import dash
import dash_bootstrap_components as dbc
import flask

from skvo_veb.components import footer
import logging
logging.basicConfig(level=logging.DEBUG)

from os import getenv
from dotenv import load_dotenv
load_dotenv()  # Note: it is important to load dotenv here,
# because celery worker runs this file and not main.py or *.wsgi


server = flask.Flask(__name__)

USE_REDIS = getenv('USE_REDIS', 'false').upper() == 'TRUE' or getenv('USE_REDIS', 'false') == '1'

# USE_REDIS = True
# USE_REDIS = False


@server.route("/")
def home():
    return app.index()


# Configure background callbacks
if USE_REDIS:
    from celery import Celery
    from dash import CeleryManager

    celery_app = Celery(__name__,
                        broker='redis://localhost:6379/0',
                        backend='redis://localhost:6379/1',
                        broker_connection_retry_on_startup=True)
    background_callback_manager = CeleryManager(celery_app)
    print(f"Using background manager: {background_callback_manager}")
else:
    import diskcache
    diskcache_dir = getenv('DISKCACHE_DIR')
    background_callback_manager = DiskcacheManager(diskcache.Cache(diskcache_dir))


app = Dash(__name__, server=server, use_pages=True,
           background_callback_manager=background_callback_manager)  # , suppress_callback_exceptions=True)
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
