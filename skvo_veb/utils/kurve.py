import io
import json
import logging
from io import StringIO

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table
from astropy.time import Time
from lightkurve import LightCurve

from skvo_veb.utils.my_tools import PipeException


def cook_lightcurve(df: pd.DataFrame, timescale: str, flux_unit: str, flux_err_unit: str,
                    period_day: float | None = None, epoch_jd: float | None = None) -> dict:
    # Call it immediately after the loading from database
    # Add columns to the lightcurve dealing with selected points:
    # df['selected'] = 0  # work
    # df['perm_index'] = df.index  # keep it forever; protect against reindexing; important when cleaning data
    # Do it explicitly on the slice of df itself
    df.loc[:, 'selected'] = 0  #
    df.loc[:, 'perm_index'] = df.index  # keep it forever; protect against reindexing; important when cleaning data
    if period_day is not None and epoch_jd is not None:
        df.loc[:, 'phase'] = _calc_phase(df['jd'], epoch_jd, period_day)

    lightcurve_data = _df_to_dict(df)
    dict_lightcurve = {'time_scale': timescale,
                       'flux_unit': flux_unit,
                       'flux_err_unit': flux_err_unit,
                       'folded_view': 1,
                       'data': lightcurve_data}

    return dict_lightcurve


_format_dict_bytes = {
    'votable': 'vot',
    'fits': 'fits',
}

_format_dict_text = {
    'ascii.ecsv': 'ecsv',
    'csv': 'csv',
    'ascii': 'dat',
    'ascii.commented_header': 'dat',
    'ascii.fixed_width': 'dat',
    'html': 'html',
    'ascii.html': 'html',
    'pandas.csv': 'csv',
    'pandas.json': 'json'
}

format_dict = _format_dict_text.copy()
format_dict.update(_format_dict_bytes)


def get_file_extension(table_format):
    return format_dict.get(table_format, 'dat')


def _calc_phase(time_arr, epoch_jd: float | None, period_day: float | None):
    phase = ((time_arr - (0 if epoch_jd is None else epoch_jd)) / (1 if period_day is None else period_day)) % 1
    return phase


def prepare_download(jdict_lc: dict, epoch: float | None, period: float | None, table_format='ascii.ecsv') -> bytes:
    """
    Write astropy Table into some string. The io.StringIO or io.BytesIO mimics the output file for Table.write()
    The specific IO-type depends on the desirable output format, i.e. on the writer type:
    astropy.io.ascii.write, astropy.io.fits.write, astropy.io.votable.write
    We use BytesIO for binary format (including xml-type votable, Yes!) and StringIO for the text one.
    If you know the better way, please tell me.

    :param period: for phase calculation
    :param epoch: for phase calculation
    :param jdict_lc: json dict with the lightcurve
    :param table_format:    one of the maintained astropy Table formats from kurve.format_dict
    :return: filename, constructed from source and band names and lightcurve as a string or bytes
    """
    if table_format in _format_dict_text:
        my_weird_io = io.StringIO()
    elif table_format in _format_dict_bytes:
        my_weird_io = io.BytesIO()
    else:
        raise PipeException(f'Unsupported format {table_format}\n Valid formats: {str(format_dict.keys())}')

    tab = _extract_lightcurve_table(jdict_lc)
    # tab['phase'] = ((tab['time'].value - epoch) / period) % 1
    # tab['phase'] = ((tab['time'].value - (0 if epoch is None else epoch)) / (1 if period is None else period)) % 1
    # tab['phase'] = _calc_phase(tab['time'].value, epoch, period)
    # todo: Add an appropriate time serialization method ???:
    # https://docs.astropy.org/en/stable/io/unified.html#id13
    if table_format == 'votable' or table_format == 'pandas.json':
        tab['jd'] = tab['time'].jd
        # tab.remove_columns(['time', 'selected', 'perm_index'])
        selected_columns = ['jd', 'phase', 'flux', 'flux_err']
        # tab = tab['jd', 'phase', 'flux', 'flux_err']
    else:
        # tab.remove_columns(['selected', 'perm_index'])
        selected_columns = ['time', 'phase', 'flux', 'flux_err']
        # tab = tab['time', 'phase', 'flux', 'flux_err']
    tab = tab[[col for col in selected_columns if col in tab.colnames]]

    tab.write(my_weird_io, format=table_format, overwrite=True)
    my_weird_string = my_weird_io.getvalue()
    if isinstance(my_weird_string, str):
        # instead, we could choose  dcc.send_string() or dcc.send_bytes() for text or byte string in Dash application
        # I prefer to place all io-logic in one place, here, and convert all stuff into bytes
        my_weird_string = bytes(my_weird_string, 'utf-8')
    my_weird_io.close()  # todo Needed?
    return my_weird_string


#     todo change it using time serialization method:
#      https://docs.astropy.org/en/stable/io/unified.html#id13


def _tab_to_lightcurve(tab: Table) -> LightCurve:
    lc = LightCurve(time=tab['time'], flux=tab['flux'], flux_err=tab['flux_err'])
    lc.add_column(tab['selected'])  # the only way (
    lc.add_column(tab['perm_index'])
    return lc


def extract_folded_lightcurve(jdict: dict, period: float | None, epoch: float | None) -> Table:
    tab = _extract_lightcurve_table(jdict)
    lc = _tab_to_lightcurve(tab)
    if not period:
        lc_folded = lc
    else:
        lc_folded = lc.fold(period=period, epoch_time=epoch, normalize_phase=True)
    tab = lc_folded
    # I can't just rename the "time" column because of astropy timeseries columns check
    tab['phase'] = tab['time'].value  # It's important to convert timedelta to float, f.e like this
    return tab


def extract_lightcurve(jdict: dict, jd0: float) -> Table:
    tab = _extract_lightcurve_table(jdict)
    tab['jd'] = tab['time'].jd - jd0
    return tab


def mark_for_deletion(jdict_lc: dict, selected_point_indices: list) -> dict:
    tab = _extract_lightcurve_table(jdict_lc)
    if tab is None:
        raise PipeException('Noting to select')
    # todo Could I perform this using astropy Table only?
    df = tab.to_pandas()
    mask = df.perm_index.isin(selected_point_indices)
    tab['selected'][list(mask)] = 1

    return _push_table(tab)


def unmark(jdict_lc: dict) -> dict:
    tab = _extract_lightcurve_table(jdict_lc)
    if tab is None:
        raise PipeException('Noting to unmark')
    tab['selected'] = 0
    return _push_table(tab)


def delete_selected(jdict_lc: dict) -> dict:
    tab = _extract_lightcurve_table(jdict_lc)
    if tab is None:
        raise PipeException('Noting to delete')
    tab_cleaned = tab[tab['selected'] == 0]
    return _push_table(tab_cleaned)


def autoclean(jdict_lc: dict, sigma=5.0, sigma_lower=None, sigma_upper=None) -> dict:
    tab = _extract_lightcurve_table(jdict_lc)
    if tab is None:
        raise PipeException('Nothing to autoclean')
    lc = _tab_to_lightcurve(tab)
    lc_cleaned = lc.remove_outliers(sigma=sigma, sigma_lower=sigma_lower, sigma_upper=sigma_upper)
    return _push_table(Table(lc_cleaned))


def clean_by_flux_err(jdict_lc: dict, flux_err_max: float) -> dict:
    tab = _extract_lightcurve_table(jdict_lc)
    if tab is None:
        raise PipeException('Nothing to clean')
    tab_cleaned = tab[tab['flux_err'] <= flux_err_max]
    return _push_table(tab_cleaned)


def suggest_flux_err_max(jdict: dict) -> float | None:
    try:
        tab = _extract_lightcurve_table(jdict)
        flux_err_max = 2 * np.median(tab['flux_err'])
    except Exception as e:
        logging.warning(f'suggest_flux_err_max: something went wrong: {repr(e)}')
        return None
    return round(flux_err_max, 1)


# We use table as LightKurve with three main obligatory columns: jd (time), flux and flux_err
# All related information, such as period, timescale, units, etc., is stored in json in the 'metadata' part,
# accessible via info module
# def _push_table(tab: Table, jdict: dict) -> dict:
def _push_table(tab: Table) -> dict:
    """
    :param tab: astropy Table with three main obligatory columns: time (astropy.Time), flux and flux_err
    :return: updated JSON dictionary with new lightcurve
    """
    time_scale = tab['time'].scale
    flux_unit = str(tab['flux'].unit)
    flux_err_unit = str(tab['flux_err'].unit)
    tab['jd'] = tab['time'].jd
    tab.remove_column('time')
    df = tab.to_pandas(index=False)
    lightcurve = {
        'time_scale': time_scale,
        'flux_unit': flux_unit,
        'flux_err_unit': flux_err_unit,
        'data': _df_to_dict(df)  # df.to_dict(orient='split', index=False)
    }
    return lightcurve
    # return info.push_lightcurve(jdict, lightcurve)


def _lightcurve_to_df(jdict_lc: dict):
    # df = pd.read_json(StringIO(json.dumps(jdict_lc['data'])), orient='split')
    df = _lc_to_df(jdict_lc['data'])
    return df


def _lc_to_df(lc_data) -> pd.DataFrame:
    df = pd.read_json(StringIO(json.dumps(lc_data)), orient='split')
    return df


def _df_to_dict(df: pd.DataFrame) -> dict:
    return df.to_dict(orient='split', index=False)


def dict_to_df(df_dict: dict) -> pd.DataFrame:
    return pd.DataFrame(data=df_dict['data'], columns=df_dict['columns'])


# def _extract_lightcurve_table(jdict: dict) -> Table:
def _extract_lightcurve_table(jdict_lc: dict) -> Table:
    """
    :param jdict_lc: JSON dictionary with lightcurve
    :return: astropy Table, metadata as JSON formatted string
    """
    # tab = None
    # meta = {}
    try:
        # jdict_lc = info.extract_lightcurve(jdict)
        df = _lightcurve_to_df(jdict_lc)
        # df = pd.read_json(StringIO(json.dumps(jdict['lightcurve']['data'])), orient='split')
        tab = Table.from_pandas(df)
        # tab['flux'].unit = u.Unit(jdict['lightcurve']['flux_unit'])
        tab['flux'].unit = u.Unit(jdict_lc['flux_unit'])
        tab['flux_err'].unit = u.Unit(jdict_lc['flux_err_unit'])
        # tab['flux_err'].unit = u.Unit(jdict['lightcurve']['flux_err_unit'])
        # tab['time'] = Time(tab['jd'], format='jd', scale=jdict['lightcurve']['time_scale'])
        tab['time'] = Time(tab['jd'], format='jd', scale=jdict_lc['time_scale'])
        tab.remove_column('jd')
    except Exception as e:
        logging.warning(f'extract_table exception occurred: {repr(e)}')
        raise
    return tab


# def get_veb_prop(gaia_id: int):  # todo do more...
#     return request_gaia._request_veb_prop_new(gaia_id)


if __name__ == '__main__':
    # gaia_name_test = 541801332594262912
    gaia_name_test = 5284186916701857536
    # gaia_name_test = 3454363379233043968
    band_test_gaia = 'G'
    # band_test_gaia = 'BP'
    # lk_, js_meta_ = request_gaia.get_gaia_lc(gaia_name_test, band_test_gaia)
    # js_ = load_source(str(gaia_name_test), band_test_gaia, 'Gaia')
    # rt = extract_epoch_gaia(js_)

    # prepare_plot(js_, True)
    # js__ = mark_for_deletion(js_, [1, 2, 3, 4, 5])
    # js___ = autoclean(js_)
    # js____ = clean_by_flux_err(js__, 20)
