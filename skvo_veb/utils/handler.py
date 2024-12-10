import json
import logging

from dash import dcc

from skvo_veb.utils.request_gaia import jd0_gaia
from skvo_veb.utils import ask_simbad, request_gaia, kurve
from skvo_veb.utils.request_asassn import load_asassn_lightcurve
from skvo_veb.utils.my_tools import DBException, PipeException, is_like_gaia_id


def serialise(jdict: dict) -> str:
    return json.dumps(jdict)


def deserialise(js: str) -> dict:
    return json.loads(js)


def main_name(cross_ident: dict):
    return cross_ident['vsx'] if cross_ident['vsx'] is not None else cross_ident['simbad'] \
        if cross_ident['simbad'] is not None else f'Gaia DR3 {cross_ident["gaia_id"]}'


def _decipher_source_id(source_id):
    if isinstance(source_id, int) and source_id > 0:  # Is it an integer identifier?
        gaia_id = source_id
        return gaia_id
    if not isinstance(source_id, str):  # Is it a string?
        raise PipeException(f'Unappropriated type of source identification {source_id}')  # Bad for you...
    # So, it is a string
    if is_like_gaia_id(source_id):  # A string with an integer identifier?
        gaia_id = int(source_id)
        return gaia_id
    # m.b. something like ''Gaia DR3 123345':
    if (gaia_id := ask_simbad.extract_gaia_id(source_id)) is not None:  # short call
        return gaia_id

    # M.b. this name is present in tho local crossident:
    if (gaia_id := request_gaia.extract_gaia_id(source_id)) is not None:
        return gaia_id

    # Suppose it is a simbad-resolvable name:
    if (gaia_id := ask_simbad.get_gaia_id_by_simbad_name(source_id)) is not None:  # long remote call
        return gaia_id

    # M.b. at least Vizier will be able to find it in the Gaia VEB table? This happens...
    if (gaia_id := ask_simbad.get_gaia_id_from_gaia_veb_table(source_id)) is None:  # very long remote call
        raise DBException(f'Source {source_id} is not found by Simbad')  # Bad for you...
    return gaia_id


def load_source(source_id: str, catalogue: str) -> dict:
    assert catalogue == 'Gaia'  # todo add other databases
    gaia_id = _decipher_source_id(source_id)  # M.b. long remote call. Or m.b. not
    # jdict_main, jdict_gaia_params, jdict_photometric_params, jdict_cross_ident = request_gaia.load_source(gaia_id)
    dict_source = request_gaia.load_source(gaia_id)
    return dict_source  # jdict_main, jdict_gaia_params, jdict_photometric_params, jdict_cross_ident


def load_lightcurve(source_id: str, band: str, catalogue: str, force_update=False) -> dict:
    assert catalogue in ['Gaia', 'Asassn']  # todo add other databases
    gaia_id = _decipher_source_id(source_id)  # M.b. long remote call. Or m.b. not
    if catalogue == 'Gaia':
        return request_gaia.load_gaia_lightcurve(gaia_id, band)
    else:
        return load_asassn_lightcurve(gaia_id, band, force_update)


def _make_lc_filename(jdict_metadata: dict, table_format: str):
    outfile_base = f'light_curve'
    try:
        # outfile_base = (f'{outfile_base}_{jdict_metadata["cross_ident"]["gaia_id"]}_{jdict_metadata["band"]}'
        #                 .replace(' ', '_'))
        outfile_base = (f'{outfile_base}_{jdict_metadata["gaia_id"]}_{jdict_metadata["band"]}'
                        .replace(' ', '_'))
        # outfile_base = f'{outfile_base}_{meta["source_id"]}_{meta["band"]}'.replace(' ', '_')
    except Exception as e:
        logging.warning(f'kurve.save_lc: {repr(e)}')
    # https://docs.astropy.org/en/stable/io/ascii/index.html#supported-formats
    ext = kurve.get_file_extension(table_format)
    outfile = f'{outfile_base}.{ext}'
    return outfile


def get_format_list():
    return list(kurve.format_dict.keys())


def prepare_download(js_lightcurve: str, js_metadata: str, table_format='ascii.ecsv') -> tuple[str, bytes]:
    """
    Write astropy Table into some string. The io.StringIO or io.BytesIO mimics the output file for Table.write()
    The specific IO-type depends on the desirable output format, i.e. on the writer type:
    astropy.io.ascii.write, astropy.io.fits.write, astropy.io.votable.write
    We use BytesIO for binary format (including xml-type votable, Yes!) and StringIO for the text one.
    If you know the better way, please tell me.

    :param js_metadata: json string contained metadata
    :param js_lightcurve: json string contained the lightcurve
    :param table_format:  one of the maintained astropy Table formats from kurve.format_dict
    :return: filename, constructed from source and band names and lightcurve as a string or bytes
    """
    metadata = deserialise(js_metadata)
    filename = _make_lc_filename(metadata, table_format)    # todo: add gaia_id
    epoch = metadata.get('epoch_gaia', None)
    period = metadata.get('period', None)
    my_weird_string = kurve.prepare_download(deserialise(js_lightcurve), epoch, period, table_format)
    return filename, my_weird_string


# def prepare_plot(js: str, phase_view: bool, period: float, epoch: float):
def prepare_plot(js_lightcurve: str, phase_view: bool, js_metadata: str):
    # if extract_error(js) is not None:
    #     clear the figure
    #     raise PipeException('Error in the data, nothing to plot')
    jdict_lightcurve = deserialise(js_lightcurve)
    metadata = deserialise(js_metadata)
    period = metadata.get('period', None)
    epoch = metadata.get('epoch', None)
    # epoch = metadata.get('epoch_gaia', None)

    # curve_title = (f'{extract_source_id(jdict)} {extract_band(jdict)} '
    #                f'{extract_mag_band(jdict)}={extract_mag(jdict):.3f}')

    if phase_view:
        xaxis_title = 'phase'
        tab = kurve.extract_folded_lightcurve(jdict_lightcurve, period=period, epoch=epoch)
    else:
        jd0 = 245000
        xaxis_title = f'jd-{jd0}'
        tab = kurve.extract_lightcurve(jdict_lightcurve, jd0)
    yaxis_title = f'flux, {tab["flux"].unit}'

    return tab, xaxis_title, yaxis_title
    # return tab, curve_title, xaxis_title, yaxis_title


def mark_for_deletion(js: str, selected_point_indices: list) -> str:
    jdict = kurve.mark_for_deletion(deserialise(js), selected_point_indices)
    return serialise(jdict)


def unmark(js: str) -> str:
    jdict = kurve.unmark(deserialise(js))
    return serialise(jdict)


def delete_selected(js: str) -> str:
    jdict = kurve.delete_selected(deserialise(js))
    return serialise(jdict)


def autoclean(js: str, sigma=5.0, sigma_lower=None, sigma_upper=None) -> str:
    jdict = kurve.autoclean(deserialise(js), sigma=sigma, sigma_lower=sigma_lower, sigma_upper=sigma_upper)
    return serialise(jdict)


def clean_by_flux_err(js: str, flux_err_max: float) -> str:
    jdict = kurve.clean_by_flux_err(deserialise(js), flux_err_max)
    return serialise(jdict)


def suggest_flux_err_max(js: str) -> float | None:
    flux_err_max = kurve.suggest_flux_err_max(deserialise(js))
    return flux_err_max


# -------------------------------- Photometric parameters part ----------------------------------------------------
DEBUG_PARAMS = True

if not DEBUG_PARAMS:
    photometric_parameter_name_dict = request_gaia.request_photometric_params_description()
    photometric_parameter_template = {}  # todo upload it from the db?
else:
    photometric_parameter_template = {
        # {long name: (fitted_key, predicted_key, (unit, precision))}
        # The order does matter
        'Inclination': (('i', 'i'), ('deg', 0)),
        'Mass ratio': (('q', 'q'), (None, 2)),
        'Potential': ((None, 'pot'), (None, 2)),
        'Potential of 1st component': (('pot1', None), (None, 3)),
        'Potential of 2st component': (('pot2', None), (None, 3)),
        'Temperature ratio': (('t1/t2', 't1/t2'), (None, 2)),
        'Temperature of the 1st component': (('t1', None), (None, 0)),
        'Temperature of the 2st component': (('t2', None), (None, 0)),
        'Bolometric luminosity of the 1st component': ((None, 'BL1'), ('solar luminosity', 2)),
        'Bolometric luminosity of the 2st component': ((None, 'BL2'), ('solar luminosity', 2)),
        'Equivalent radius of the 1st component': ((None, 'R1_eq'), ('solar radius', 3)),
        'Equivalent radius of the 2st component': ((None, 'R2_eq'), ('solar radius', 3)),
        'Phase shift': ((None, 'phase_shift'), (None, 1)),
        'Sum of squares of fit': (('sq', 'sq'), (None, 5)),
    }  # todo Use this
    main_parameters_dict = {
        'gaia_id': ('Source identifier', None),
        'parallax': ('Parallax', 'mas', 4),
        'pm': ('Total proper motion', 'mas/yr', 4),
        'pm_ra': ('Proper motion in right ascension direction, pmRA*cosDE', 'mas/yr', 4),
        'pm_de': ('Proper motion in declination direction', 'mas/yr', 4),
        'g_mag': ('G magnitude', 'mag', 3),
        'bp_mag': ('Bp magnitude', 'mag', 3),
        'rp_mag': ('Rp magnitude', 'mag', 3),
        'teff': ('Teff from BP/RP spectra', 'K', 0),
        'teff_low': ('Lower confidence level (16%) of Teff', 'K', 0),
        'teff_up': ('Upper confidence level (84%) of Teff', 'K', 0),
        'logg': ('Logg from BP/RP spectra', None, 3),
        'logg_up': ('Upper confidence level (84%) of logg', None, 3),
        'logg_low': ('Lower confidence level (16%) of logg', None, 3),
        'fe2h': ('[Fe/H] from BP/RP spectra', None, 3),
        'fe2h_up': ('Upper confidence level (84%) of [Fe/H]', None, 3),
        'fe2h_low': ('Lower confidence level (16%) of [Fe/H]', None, 3),
        'rv': ('Radial velocity', 'km/s', 3),
        'vbroad': ('Spectral line broadening parameter', 'km/s', 3),
        'coordequ': None,  # i.e., ignore this parameter
    }
    gaia_photometric_parameter_name_dict = {
        # Gaia parameters: Description, units, precision (digits after the decimal point)
        'gaia_id': ('Source identifier', None),
        'time_ref': ('Estimated reference time', f'jd-{jd0_gaia}', 4),
        'freq': ('Frequency', None, 5),
        'mag_mod': ('Model magnitude reference level', 'mag', 3),
        'phase1': ('Phase of the Gaussian 1 component', None, 3),
        'sig_phase1': ('Standard deviation of Gaussian 1 component', 'phase', 3),
        'depth1': ('Magnitude depth of Gaussian 1 component', 'mag', 3),
        'phase2': ('Phase of the Gaussian 2 component', None, 3),
        'sig_phase2': ('Standard deviation of Gaussian 2 component', 'phase', 3),
        'depth2': ('Magnitude depth of Gaussian 2 component', 'mag', 3),
        'amp_chp': ('Amplitude of the cosine component with half the period of the model', 'mag', 3),
        'phase_chp': ('Reference phase of the cosine component with half the period of the model', None, 3),
        'phase_e1': ('Primary eclipse: phase at geometrically deepest point', None, 3),
        'dur_e1': ('Primary eclipse: duration', 'phase fraction', 4),
        'depth_e1': ('Primary eclipse: depth', 'mag', 3),
        'phase_e2': ('Secondary eclipse: phase at geometrically deepest point', None, 3),
        'dur_e2': ('Secondary eclipse: duration', 'phase fraction', 4),
        'depth_e2': ('Secondary eclipse: depth', 'mag', 3),
        'model_type': ('Type of geometrical model of the light curve', None),
    }
    lamost_parameter_name_dict = {
        # LAMOST parameters
        'Teff(low)': ('Teff from Low-resolution spectrum', 'K', 0),
        'Teff_lasp(med)': ('Teff from Medium-resolution spectrum, LAMOST pipeline', 'K', 0),
        'Teff_cnn(med)': ('Teff from Medium-resolution spectrum, CNN method', 'K', 0),
        'Fe/H(low)': ('[Fe/H] from Low-resolution spectrum', None, 3),
        'Fe/H_lasp(med)': ('[Fe/H] from Medium-resolution spectrum, LAMOST pipeline', None, 3),
        'Fe/H_cnn(med)': ('[Fe/H] from Medium-resolution spectrum, CNN method', None, 3),
        'logg(low)': ('Logg from Low-resolution spectrum', None, 2),
        'logg_lasp(med)': ('Logg from Medium-resolution spectrum, LAMOST pipeline', None, 2),
        'logg_cnn(med)': ('Logg from Medium-resolution spectrum, CNN method', None, 2),
    }


# def create_prop_tables(jdict_params: dict) -> dict:
#     """
#
#     :param jdict_params:
#     :return: dictionary of the form {table_title: list of rows}, where
#     row has a form [formatted value1, formatted value2]
#     """
#
#     def format_field_name(key):
#         try:
#             desc, unit = parameter_name_dict[key]
#             if unit is None:
#                 return f'{desc.capitalize()} ({key})'
#             else:
#                 return f'{desc.capitalize()} ({key}), [{unit}]'
#         except Exception as _e:
#             logging.warning(f'get_field_name: {repr(_e)}')
#             return key
#
#     def format_value(value):
#         try:
#             val, err = value
#             text = f'${val} \pm {err}$'
#         except (TypeError, ValueError):
#             text = f'${value}$'
#         return dcc.Markdown(text, mathjax=True)
#
#     def table_from_dict(di_) -> [[str, str]]:
#         row_list_ = []
#         if di_ is None:
#             di_ = {}
#
#         for key, value in di_.items():
#             row_list_.append([format_field_name(key), format_value(value)])
#         return row_list_
#
#     # jdict_photometric_params = deserialise(jdict_photometric_params)
#     tables_dict = {}
#     try:
#         gaia_id = jdict_params['gaia_id']
#         dict_main = {'source': gaia_id,
#                      'spot': jdict_params['spot'],
#                      'type': jdict_params['type']}
#         for di, title in zip(
#                 [dict_main] + [jdict_params[dict_name] for dict_name in ['absolute', 'fitted', 'predicted']],
#                 ['Main', 'Absolute Parameters', 'Fitted Parameters', 'Predicted Parameters']):
#             row_list = table_from_dict(di)
#             tables_dict[title] = row_list
#         return tables_dict
#
#     except Exception as e:
#         logging.warning(f'create_prop_tables exception: {repr(e)}')
#         return {}


def reformat_dict(di: dict) -> dict:
    """
        interpret *_err keys as errors of *
    :param di:
    :return: improved dict {name: (val,err)}
    """
    #
    new_di = {}
    for key, value in di.items():
        if value is None or (isinstance(value, str) and value.strip() == ''):
            # Ignore empty parameters
            continue
        # if key.startswith('e_') and (key.replace('e_', '') in di.keys()):
        if key.endswith('_err') and (key.replace('_err', '') in di.keys()):
            if value is None:
                continue
            key_new = key.replace('_err', '')
            new_di[key_new] = (di[key_new], value)
        else:
            new_di[key] = value
    return new_di


def format_field_name(description, unit):
    if unit is None:
        # return f'{desc_.capitalize()}'
        return f'{description}'
    # return f'{desc_.capitalize()}, ({unit_})'
    return f'{description}, ({unit})'


def format_value(value, precision: int | None):
    try:
        if isinstance(value, tuple) and len(value) == 2:
            val, err = value
        else:
            val = value
            err = None
        err_str = ''
        val_str = str(val)
        if precision is not None:  # and isinstance(val_, (int, float)):
            try:
                val_str = f'{float(val):.{precision}f}'
                if err is not None:
                    # err = round(float(err), precision)
                    err = float(err)
                    err_str = f'\({err:.{precision}f}\)' if err else ''
            except Exception as e_:
                logging.warning(f'table_from_dict: format_value {value=}, {precision=} {e_}')
                pass
        text = f'{val_str}    {err_str}'
    except (TypeError, ValueError):
        # text = f'${value}$'
        text = f'{value}'
    return dcc.Markdown(text)
    # return dcc.Markdown(text, mathjax=True)


def table_from_dict(di: dict | None, params_catalog: str) -> list:
    """
    Convert a parameter dictionary into formatted table, i.e., a list of pairs of strings in the form "name","value"
    The dictionary keys are converted into a formatted name using a special dictionary
    "parameter_name_dict" loaded from the Database
    :param params_catalog: Gaia, Lamost, Photometric -- dictionary of parameters description: Long name, unit, precision
    :param di: dictionary in the form {parameter_name: (value,error)}
    :return:
    """

    row_list = []
    if di is None:
        di = {}
    di = reformat_dict(di)

    def parse_options(opt: tuple):
        desc_, unit_, prec_ = None, None, None
        try:
            desc_, unit_, prec_ = opt
        except ValueError:
            try:
                desc_, unit_ = opt
            except ValueError:
                desc_ = opt
        except Exception as e:
            logging.warning(f'parse_options {opt}: {repr(e)}')
        return desc_, unit_, prec_

    if ('GAIA' and 'PHOTOMETRIC') in params_catalog.upper():
        parameter_name_dict = gaia_photometric_parameter_name_dict
    elif 'LAMOST' in params_catalog.upper():
        parameter_name_dict = lamost_parameter_name_dict
    elif ('GAIA' and 'MAIN') in params_catalog.upper():
        parameter_name_dict = main_parameters_dict
    else:
        parameter_name_dict = photometric_parameter_name_dict
    # First of all, try to fill a catalog-specific dictionary grom the DB:
    for key, option in parameter_name_dict.items():
        if key not in di:
            continue
        val = di.pop(key)
        if option is None:  # This item has been marked as ignored
            continue
        desc, unit, prec = parse_options(option)
        row_list.append([format_field_name(desc, unit), format_value(val, prec)])
    # Then add the rest:
    for key, val in di.items():
        desc, unit, prec = key, None, None
        logging.warning(f'The description of parameter {key} is missing from the database')
        row_list.append([format_field_name(desc, unit), format_value(val, prec)])
    return row_list


def photometric_param_table(predicted_params: dict, fitted_params: dict) -> list:
    row_list = [['Name', 'Fitted', 'Predicted']]
    predicted_params = reformat_dict(predicted_params)
    fitted_params = reformat_dict(fitted_params)
    for desc, value in photometric_parameter_template.items():
        # {long name: (fitted_key, predicted_key, (unit, precision))}
        key_fitted, key_predicted = value[0]
        unit, prec = value[1]
        val_fitted = fitted_params.get(key_fitted, '') if key_fitted is not None else ''
        val_predicted = predicted_params.get(key_predicted, '') if key_predicted is not None else ''
        if val_fitted == '' and val_predicted == '':
            continue
        row_list.append([format_field_name(desc, unit), format_value(val_fitted, prec),
                         format_value(val_predicted, prec)])
    return row_list


if __name__ == '__main__':
    # gaia_name_test = 541801332594262912
    gaia_name_test = 5284186916701857536
    # gaia_name_test = 3454363379233043968
    band_test_gaia = 'G'
    # band_test_gaia = 'BP'
    # lk_, js_meta_ = request_gaia.get_gaia_lc(gaia_name_test, band_test_gaia)
    js_ = load_source(str(gaia_name_test), 'Gaia')
    # rt = extract_epoch_gaia(js_[0])

    # prepare_plot(js_, True)
    # js__ = mark_for_deletion(js_[0], [1, 2, 3, 4, 5])
    # js___ = autoclean(js_[0])
    # js____ = clean_by_flux_err(js__, 20)
