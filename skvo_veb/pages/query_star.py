import logging
import urllib.parse

import dash
import dash_bootstrap_components as dbc
import aladin_lite_react_component
from dash import html, dcc  # , callback, Input, Output, State

import skvo_veb.utils.my_tools
from skvo_veb.components import message
from skvo_veb.utils import coord, veb_parameters, request_gaia
from skvo_veb.utils.my_tools import timeit
from skvo_veb.utils.request_gaia import decipher_source_id

dash.register_page(__name__,
                   path='/igebc/star',
                   title='IGEBC: Source parameters',
                   in_navbar=False,
                   name='Search results')


def create_table(row_list: list) -> dbc.Table:
    # rows = [html.Tr([html.Th(row[0]), html.Td(row[1])]) for row in row_list]
    rows = [html.Tr([html.Th(row[0])] + [html.Td(cell_content) for cell_content in row[1:]]) for row in row_list]
    return dbc.Table(html.Tbody(rows), borderless=True, responsive=True, hover=True)


# @timeit
# def format_photometric_params(jdict_photometric_params: dict):  # todo Change it!
#     #   -------------------------------------------------- Table in the Accordion ----------------------------
#
#     dict_titled_rows = handler.create_prop_tables(jdict_photometric_params)
#     params = [dbc.AccordionItem(create_table(row_list), title=title) for title, row_list in
#               dict_titled_rows.items()]
#
#     return params


@timeit
def accordion_item_from_params(jdict_params: dict, title: str):
    try:
        row_list = veb_parameters.table_from_dict(jdict_params, title)
        if len(row_list) < 1:
            return None
        return dbc.AccordionItem(create_table(row_list), title=title)

    except Exception as e:
        logging.warning(f'accordion_item_from_params {title} {repr(e)}')
        return None


@timeit
def accordion_item_from_photometric_params(predicted_params: dict, fitted_params: dict, title: str):
    if predicted_params is None or fitted_params is None:
        return None
    try:
        row_list = veb_parameters.photometric_param_table(predicted_params, fitted_params)
        return dbc.AccordionItem(create_table(row_list), title=title)
    except Exception as e:
        logging.warning(f'accordion_item_from_photometric_param {title} {repr(e)}')
        return None


def _str_to_float(val_str):
    try:
        return float(val_str)
    except Exception as e:
        logging.warning(f'{val_str} to float: {e}')
        return None


def _safe_round(value: float | None, precision: float | None) -> str:
    if value is None:
        return ''
    try:
        res = round(value, precision)
    except Exception as e:
        logging.warning(f'round({value}, {precision}: {e}')
        res = ''
    return res


def _load_source_params_gaia(source_id: str) -> dict:
    gaia_id = decipher_source_id(source_id)  # M.b. long remote call. Or m.b. not
    # jdict_main, jdict_gaia_params, jdict_photometric_params, jdict_cross_ident = request_gaia.load_source(gaia_id)
    dict_source = request_gaia.load_source_params(gaia_id)
    return dict_source  # jdict_main, jdict_gaia_params, jdict_photometric_params, jdict_cross_ident


@timeit
def summary(jdict_main_data: dict, jdict_cross_ident: dict):  # todo add lamost data
    try:
        name = f'Gaia DR3 {jdict_main_data["gaia_id"]}'
        synonyms = []
        sky_coords = coord.coordequ_to_skycoord(jdict_main_data["coordequ"])
        coord_hms_dms = coord.skycoord_to_hms_dms(sky_coords, precision=1)
        coord_dms_dms = coord.skycoord_to_dms_dms(sky_coords, precision=1)
        # coord_hms_dms, coord_dms = coord.coordequ_to_hms_dms_both_str(jdict_main_data["coordequ"], precision=1)
        g_mag = _str_to_float(jdict_main_data.get("g_mag", None))
        g_mag_err = _str_to_float(jdict_main_data.get("g_mag_err", None))
        bp_mag = _str_to_float(jdict_main_data.get("bp_mag", None))
        bp_mag_err = _str_to_float(jdict_main_data.get("bp_mag_err", None))
        rp_mag = _str_to_float(jdict_main_data.get("rp_mag", None))
        rp_mag_err = _str_to_float(jdict_main_data.get("rp_mag_err", None))
        parallax = _str_to_float(jdict_main_data.get('parallax', None))
        parallax_err = _str_to_float(jdict_main_data.get('parallax_err', None))
        teff = _str_to_float(jdict_main_data.get("teff", None))
        logg = _str_to_float(jdict_main_data.get("logg", None))
        fe2h = _str_to_float(jdict_main_data.get("fe2h", None))

        if jdict_cross_ident["simbad"] is not None and jdict_cross_ident["simbad"] != name:
            synonyms.append(f'{jdict_cross_ident["simbad"]}')

        if jdict_cross_ident["vsx"] is not None and jdict_cross_ident["vsx"] != jdict_cross_ident["simbad"]:
            synonyms.append(f'{jdict_cross_ident["vsx"]}')
                
        text = '|||\n|:---|:---|\n'  # Markdown table

        text += f'|**Names:**|{name}|\n'

        # Add synonyms if they exist
        for syn in synonyms:
            text += f'||{syn}|\n'
        
        text += f'|**ICRS coord:**|**{coord_hms_dms}**|\n'
        text += f'||{coord_dms_dms}|\n'
        
        mag_prec = 3
        if g_mag is not None:
            text += f'|**Fluxes**:|**Gmag**={_safe_round(g_mag, mag_prec)} \[{_safe_round(g_mag_err, mag_prec)}\]|\n'
        else:
            text += f'|**Fluxes**:||\n'
        if bp_mag is not None:
            text += f'||**Bp**={_safe_round(bp_mag, mag_prec)} \[{_safe_round(bp_mag_err, mag_prec)}\]|\n'
        if rp_mag is not None:
            text += f'||**Rp**={_safe_round(rp_mag, mag_prec)} \[{_safe_round(rp_mag_err, mag_prec)}\]|\n'

        for name, val, err, prec in zip(['Parallax(mas)', 'Teff', 'Logg', 'Fe/H'],
                                        [parallax, teff, logg, fe2h],
                                        [parallax_err, None, None, None],
                                        [4, 1, 3, 3]):
            if val is not None:
                text += f'|**{name}**:|{_safe_round(val, prec)}'
                if err is not None:
                    text += f'\[{_safe_round(err, prec)}\]'
                text += '|\n'
    except Exception as e:
        logging.warning(f'load_summary: {repr(e)}')
        text = None
    ret = dcc.Markdown(children=text,
                       style={"white-space": "pre", "font-size": 14, 'font-family': 'courier'},
                       )
    return ret


@timeit
def layout(source_id='AA%20And'):
    try:
        logging.info(f'Load source data from db: {source_id}')
        dict_source = _load_source_params_gaia(source_id)
        jdict_main = dict_source['jdict_main']
        gaia_id = jdict_main['gaia_id']
        jdict_gaia_params = dict_source['jdict_gaia_params']
        # jdict_photometric_params = dict_source['jdict_photometric_params']
        jdict_cross_ident = dict_source['jdict_cross_ident']
        # jdict_lamost = dict_source['jdict_lamost']
        # Retrieve Lamost parameters:
        try:  # It may not exist
            jdict_lamost_info = dict_source['jdict_lamost'].get('info', None)  # It may not exist
            lamost_link_low = dict_source['jdict_lamost'].get('link_to_page_low', None)
            lamost_link_med = dict_source['jdict_lamost'].get('link_to_page_med', None)
        except Exception as e:
            logging.info(f'query star {source_id=} lamost data were not found: {repr(e)}')
            jdict_lamost_info, lamost_link_low, lamost_link_med = None, None, None
        # Retrieve predicted photometric parameters
        # try:
        #     jdict_predicted = jdict_photometric_params['predicted']
        #     jdict_fitted = jdict_photometric_params['fitted']
        # except Exception as e:
        #     logging.info(f'query star gaia_id={source_id}, photometric were not found: {repr(e)}')
        #     jdict_predicted, jdict_fitted = None, None
        # gaia_id = jdict_cross_ident['gaia_id']
        source_name = skvo_veb.utils.my_tools.main_name(jdict_cross_ident)
        sky_coords = coord.coordequ_to_skycoord(jdict_main["coordequ"])
        coord_hms_dms = coord.skycoord_to_hms_dms(sky_coords, precision=1)
        # coord_dms_dms = coord.skycoord_to_dms_dms(sky_coords, precision=1)
        # coord_hms_dms, coord_dms = coord.coordequ_to_hms_dms_both_str(jdict_main["coordequ"], precision=1)
        ra_deg, dec_deg = sky_coords.ra.deg, sky_coords.dec.deg
        print(coord_hms_dms, type(coord_hms_dms))
        # param_items = accordion_item_from_params(jdict_lamost, jdict_gaia_params, jdict_photometric_params)
        param_items = [
            accordion_item_from_params(jdict_main, 'GAIA Main parameters'),
            accordion_item_from_params(jdict_gaia_params, 'GAIA photometric parameters'),
            accordion_item_from_params(jdict_lamost_info, 'LAMOST parameters'),
            # accordion_item_from_photometric_params(jdict_predicted, jdict_fitted, 'Photometric parameters'),
        ]
        # try:
        #     image = request_gaia.request_photometric_params_image(gaia_id)
        #     item = dbc.AccordionItem(html.Img(width=400, src=image, id='img-graph'), title='Lightcurve fit')
        #     param_items.append(item)
        # except DBException as e:
        #     logging.info(f'lightcurve fit for gaia_id={gaia_id}: repr(e)')
        #     pass
        parameters = dbc.Accordion([item for item in param_items if item is not None],
                                   flush=False, start_collapsed=True)
        # links_text = f'[Gaia Lightcurve](/igebc/lc?source_id={gaia_id}&catalogue=Gaia&band=G)'
        # links_text = f'[Gaia Lightcurve](/igebc/lc?source_id={urllib.parse.quote(source_name)}&catalogue=Gaia&band=G)'
        links_text = f'[Gaia Lightcurve](/igebc/gaia?source_id={urllib.parse.quote(source_name)}&band=G)'
        # links_text += f'   [Request ASAS-SN Lightcurve](/igebc/asassn?gaia_id={gaia_id}&band=g)\n'
        links_text += (f'   [Request ASAS-SN Lightcurve](/igebc/asassn?source_id='
                       f'{urllib.parse.quote(source_name)}&band=g)\n')
        if lamost_link_med is not None:
            links_text += f'[LAMOST med-res spectrum]({lamost_link_med})   '
        if lamost_link_low is not None:
            links_text += f'[LAMOST low-res spectrum]({lamost_link_low})'
        content = dbc.Container([
            html.Br(),
            html.H1(f'{source_name}', className="text-primary text-left fs-3"),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Stack([
                        summary(jdict_main, jdict_cross_ident),
                        dcc.Markdown(links_text,
                                     style={"white-space": "pre", "font-size": 14, 'font-family': 'courier'}),
                    ]),
                ], md=6, sm=12),  # Summary
                dbc.Col([
                    dbc.Row([
                        aladin_lite_react_component.AladinLiteReactComponent(
                            id='aladin',
                            width=500,
                            height=300,
                            fov=5 / 60,  # in degrees
                            target=coord_hms_dms,
                            stars=[
                                {"ra": ra_deg, "dec": dec_deg, "name": source_name},
                            ]
                        ),
                    ], justify='center'),
                ], md=6, sm=12),  # Aladin
            ]),
            dbc.Row([
                html.Br(),
            ]),
            dbc.Row([
                parameters,
            ], class_name="g-2"),
            dcc.Location(id='location-query-star'),
            # dcc.Store(data=json.dumps({'source_id': source_name,
            #                            'gaia_id': gaia_id, 'catalogue': catalogue}), id='store-query_star')
        ], fluid=True)
    except Exception as e:
        logging.warning(repr(e))
        content = dbc.Container([
            html.H1('Query result', className="text-primary text-left fs-3"),

            # message.warning_alert(repr(e)),
            message.warning_alert(e),
        ], fluid=True)

    return content

# @callback(Output('location-query-star', 'href'),
#           Input('btn-submit-lc', 'n_clicks'),
#           # State('select-band', 'value'),
#           State('store-query_star', 'data'),
#           prevent_initial_call=True)
# def goto_source_page(_, js):
#     band = 'G'
#     logging.info(f'goto_source_page {band} {js}')
#     try:
#         di_js = json.loads(js)
#         gaia_id = di_js['gaia_id']
#         catalogue = di_js['catalogue']
#     except Exception as e:
#         raise PipeException(e)
#     return f'/lc?source_id={gaia_id}&catalogue={catalogue}&band={band}'
