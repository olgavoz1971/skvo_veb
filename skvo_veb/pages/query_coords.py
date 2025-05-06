import logging
import re

from astropy.coordinates import Angle
# noinspection PyUnresolvedReferences
from astropy.units import deg, hourangle
from dash import register_page, html
import dash_bootstrap_components as dbc
from dash.dcc import Markdown
from dash import callback, Input, Output, State
from dash.exceptions import PreventUpdate

from skvo_veb.components.table import table_with_link
from skvo_veb.components import message, table
from skvo_veb.utils import request_gaia, coord
from skvo_veb.utils.my_tools import timeit
import aladin_lite_react_component

register_page(__name__, name='Query by coordinates',
              title='IGEBC: Query by coordinates',
              description='Query by coordinates description',
              in_navbar=False,
              path='/igebc/coo')


def load_cone(coord_str: str, radius: str, catalogue):
    logging.debug('load_cone')
    df, tooltips = request_gaia.request_coord_cone(coord_str, radius, catalogue)
    # f-string doesn't work here. But why???
    df['Identifier'] = ('[' + df["Identifier"] + '](/igebc/star?source_id=' + df["gaia_id"] +
                        # '&catalogue=' + df['catalogue'] +
                        ')')
    # Select and rearrange columns in a user-friendly manner:
    columns = ['Identifier', 'RA DEC', 'Mag G', 'dist']
    return df[columns], tooltips


@timeit
def layout(coords='0 0', radius='10'):
    try:
        df, tooltips = load_cone(coords, radius, catalogue='Gaia')
        table_coord = table_with_link(df=df, tooltips=tooltips, ident='table_query_coords', numeric_columns={'Mag G': 2, 'dist': 1})
        df[['ra', 'dec']] = df['RA DEC'].str.split(' ', expand=True)
        df['ra'] = Angle(df['ra'].values, unit=hourangle).deg
        df['dec'] = Angle(df['dec'].values, unit=deg).deg
        df['name'] = df['Identifier'].str.extract(r'\[([^\]]+)\]')
        stars = df[['name', 'ra', 'dec']].to_dict(orient='records')
        try:
            radius_float = float(radius)
        except ValueError:
            radius_float = 10
        response = dbc.Row([
            dbc.Col([
                table_coord
            ], className="gx-2", lg=7, md=7, sm=12),
            dbc.Col([
                aladin_lite_react_component.AladinLiteReactComponent(
                    id='aladin',
                    width=400,
                    height=400,
                    fov=round(2 * radius_float) / 60,  # in degrees
                    target=coord.skycoord_to_hms_dms(coord.parse_coord_to_skycoord(coords), precision=1),
                    stars=stars,
                ),
                Markdown(children='*Click on a table row to highlight the object on the map, and vice versa*',
                         style={"font-size": 14, 'font-family': 'courier'}),
                # style={"white-space": "pre", "font-size": 14, 'font-family': 'courier'})
                dbc.Label(id='label_aladin_coords', children='')
            ], className="gx-2", lg=5, md=5, sm=12)  # p-0 -- without padding
        ])
        # response = table_coord
    except Exception as e:
        response = message.warning_alert(e)

    return dbc.Container([
        html.H1('Query coordinates', className="text-primary text-left fs-3"),
        dbc.Label(f'Request by coord "{coords}" within the radius {radius} arc min'),
        html.Br(),
        response,
    ], fluid=True)


@callback(Output('aladin', 'selectedStar'),
          Input('table_query_coords', 'active_cell'),
          State('table_query_coords', 'data'),
          prevent_initial_call=True)
def display_row_data(active_cell, data):
    if active_cell:
        row_index = active_cell['row']
        row_data = data[row_index]
        # coord_str = row_data.get('RA DEC', None)
        # match = re.match(r'\[([^\]]+)\]', row_data['Identifier'])
        match = re.match(r'\[([^]]+)\]', row_data['Identifier'])  # extract link_name from [link_name](/link)
        if match:
            name = match.group(1)
        else:
            name = ''
        return {"ra": 0, "dec": 0, "name": name}  # Honestly, ra, dec don't matter.
        # I search the right star by its name
        # return coord_str
    raise PreventUpdate


@callback(Output('label_aladin_coords', 'children'),
          Output('table_query_coords', 'style_data_conditional'),
          Input('aladin', 'selectedStar'),
          State('table_query_coords', 'data'),
          prevent_initial_call=True)
def highlite_yellow(selected_star, table_data):
    name = selected_star['name']
    for i, row in enumerate(table_data):
        match = re.match(r'\[([^]]+)]', row['Identifier'])  # extract link name from [link name](/link) )
        if match and match.group(1) == name:
            styles_conditionals = table.highlite_styles(i)
            return name, styles_conditionals
        else:
            continue
    raise PreventUpdate

