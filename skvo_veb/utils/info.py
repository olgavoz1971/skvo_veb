# This module knows all about star data structure, and it's a serialisation
# Nobody else knows this!

import json

from astropy import units as u
# noinspection PyUnresolvedReferences
from astropy.units import day

# todo Remove this module
# --------------------------------------- Collect all data in a json dictionary ---------------------------


def collect_dictionaries(source_id, band: str, mag: float | None, mag_band: str,
                         period: float | None, reftime_gaia: float | None, reftime_new: float | None,
                         dict_main_data: dict,
                         dict_prop_gaia: dict,
                         dict_cross_id: dict,
                         lightcurve: dict) -> dict:
    meta = {'source_id': source_id, 'band': band, 'mag': mag, 'mag_band': mag_band,
            'period': None, 'period_unit': None}

    try:
        meta['period'] = period.value
        meta['period_unit'] = str(period.unit)
    except Exception as e:
        print(f'info.collect_dictionaries: {repr(e)}')

    meta['reftime_gaia'] = reftime_gaia
    meta['reftime_new'] = reftime_new

    return {'metadata': meta, 'main_data': dict_main_data, 'prop_gaia': dict_prop_gaia,
            'cross_id': dict_cross_id, 'lightcurve': lightcurve}


# def extract_source_id(jdict: dict):
#     try:
#         return jdict['metadata']['source_id']
#     except Exception as e:
#         print(f'extract_source_id: {repr(e)}')
#         return None


# def extract_band(jdict: dict):
#     try:
#         return jdict['metadata']['band']
#     except Exception as e:
#         print(f'extract_band: {repr(e)}')
#         return None


def extract_mag_band(jdict: dict):
    try:
        return jdict['metadata']['mag_band']
    except Exception as e:
        print(f'extract_mag_band: {repr(e)}')
        return None


def extract_mag(jdict: dict):
    try:
        return jdict['metadata']['mag']
    except Exception as e:
        print(f'extract_mag: {repr(e)}')
        return None


def extract_lightcurve(jdict: dict):
    return jdict['lightcurve']


def push_lightcurve(jdict: dict, lc: dict):
    jdict['lightcurve'] = lc
    return jdict


def push_error(error: str) -> dict:
    return {'error': error}


def extract_error(js: str) -> str | None:
    if js is None:
        return 'Empty string'
    jdict = json.loads(js)
    return jdict.get('error', None)


def serialise(jdict: dict) -> str:
    return json.dumps(jdict)


def deserialise(js: str) -> dict:
    return json.loads(js)


def extract_main_data(js: str) -> dict:
    try:
        jdict = deserialise(js)
        jdict_main_data = jdict['main_data']
    except Exception as e:
        print(f'extract_main_data: something went wrong: {repr(e)}')
        jdict_main_data = {}
    return jdict_main_data


def extract_cross_ident(js: str) -> dict:
    # yes
    try:
        jdict = deserialise(js)
        jdict_cross_ident = jdict['cross_id']
    except Exception as e:
        print(f'extract_cross_ident: something went wrong: {repr(e)}')
        jdict_cross_ident = {}
    return jdict_cross_ident


def extract_period(js: str) -> float | None:
    if extract_error(js) is not None:
        return None
    jdict = deserialise(js)
    try:
        period_unit = jdict['metadata']['period_unit']
        period = jdict['metadata']['period'] * u.Unit(period_unit)
        period_in_days = period.to(day).value
        return period_in_days
    except Exception as e:
        print(f'extract_period: {repr(e)}')
        return None


def push_period(period: float, js: str) -> str:
    if extract_error(js) is not None:
        print('Period_to_json: Error in the data, can\'t store period')
        return js
    try:
        jdict = deserialise(js)
        jdict['metadata']['period'] = period
        return serialise(jdict)
    except Exception as e:
        print(f'push_period: {repr(e)}')
        return js


def extract_epoch_gaia(js: str) -> float | None:
    if extract_error(js) is not None:
        return None
    try:
        jdict = deserialise(js)
        epoch = jdict['metadata']['reftime_gaia']
        return epoch
    except Exception as e:
        print(f'extract_epoch_gaia: {repr(e)}')
        return None


def extract_epoch_new(js: str) -> float | None:
    if extract_error(js) is not None:
        return None
    try:
        jdict = deserialise(js)
        epoch = jdict['metadata']['reftime_new']
        return epoch
    except Exception as e:
        print(f'extract_epoch_new: {repr(e)}')
        return None


# ------------------- VEB new properties storage ----------------
def extract_source_id_prop(js_prop_new: str):
    jdict = deserialise(js_prop_new)
    try:
        return jdict['gaia_id']
    except Exception as e:
        print(f'extract_source_id_prop: {repr(e)}')
        return None
