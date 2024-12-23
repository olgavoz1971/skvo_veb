import logging
import os

import pandas
import pandas as pd
# noinspection PyUnresolvedReferences
from astropy.units import deg, hourangle, day, electron
from numpy import isnan

from pyasassn.client import SkyPatrolClient

from skvo_veb.utils.kurve import cook_lightcurve
from skvo_veb.utils.my_tools import DBException, timeit, PipeException

# http://asas-sn.ifa.hawaii.edu/documentation/getting_started.html
gaia_id_DP_Peg = 1791119426789765632
gaia_id_no_data = 1791119426789765630
gaia_id_no_ephem = 1865212594815600768


def _build_path_to_cache(gaia_id):
    cache_dir = os.getenv('ASASSN_CACHE_DIR')
    if not cache_dir:
        logging.warning('Environmental variable ASASSN_CACHE_DIR is not specified')
        return None
    return os.path.join(cache_dir, f'asassn_lc_{gaia_id}.pkl')


def _load_from_cache(gaia_id) -> (pandas.DataFrame | None, float | None, float | None):
    path_to_cached_data = _build_path_to_cache(gaia_id)
    if not path_to_cached_data or not os.path.exists(path_to_cached_data):
        return None, None, None
    try:
        df = pd.read_pickle(path_to_cached_data)
        epoch = df.attrs.get('epoch', None)
        period = df.attrs.get('period', None)
        df.attrs.clear()
        return df, epoch, period,
    except Exception as e:
        logging.warning(f'request_asassn, bad pickle file {path_to_cached_data}: {e}')
        return None, None, None


def _store_in_cache(gaia_id, df: pandas.DataFrame, epoch: float | None = None, period: float | None = None):
    path_to_cached_data = _build_path_to_cache(gaia_id)
    if not path_to_cached_data:
        return
    # if os.access(cache_dir, os.W_OK)
    try:
        df.attrs['epoch'] = epoch
        df.attrs['period'] = period
        df.to_pickle(path_to_cached_data)
    except Exception as e:
        logging.warning(f'Store Asas-SN lightcurve in cache: {e}')


@timeit
def load_asassn_lightcurve(gaia_id: int, band='g', force_update=False) -> dict:
    epoch = None
    period = None
    lc_df = None

    if os.getenv('DEBUG_ASASSN'):
        gaia_id = gaia_id_DP_Peg
        path_to_test_data = 'test_data/'
        lc_df = pd.read_pickle(path_to_test_data + str(gaia_id) + '.pkl')
        period = 1.661301
        epoch = 2452501.753
        _store_in_cache(gaia_id, lc_df, epoch, period)
        # raise DBException('Test Exception: Star not found, try another object')
    else:
        if not force_update:
            lc_df, epoch, period = _load_from_cache(gaia_id)
            if lc_df is not None and lc_df.empty:
                raise DBException(f'Gaia DR3 {gaia_id} was not found in the cached ASAS-SN database\n'
                                  f'Consider forcing a fetch if the data is really needed')
        if lc_df is None:       # Try to load it from the remote database:
            try:
                client = SkyPatrolClient()
            except Exception as e:
                logging.error('request_asassn SkyPatrolClient exception', e)
                raise e
            try:
                # res = client.query_list(gaia_id, catalog='stellar_main', id_col='gaia_id', download=True)
                res = client.adql_query(f'SELECT asas_sn_id, epoch, period FROM stellar_main '
                                        f'JOIN aavsovsx USING(asas_sn_id) WHERE gaia_id = {gaia_id}',
                                        download=True)
                # lc_df = getattr(res, 'data', [])
                logging.info('Lightcurve is ready. Or not...')
                if hasattr(res, 'data'):
                    lc_df = res.data
                if lc_df is None or lc_df.empty:
                    _store_in_cache(gaia_id, pd.DataFrame())
                    raise DBException(f'The source Gaia DR3 {gaia_id} was not found in the ASAS-SN database')
                if hasattr(res, 'catalog_info'):
                    # res.catalog_info.replace({float('nan'): None}, inplace=True)
                    epoch = getattr(res.catalog_info, 'epoch', [None])[0]
                    period = getattr(res.catalog_info, 'period', [None])[0]
                    # epoch = None if epoch is None or (isinstance(epoch, float) and isnan(epoch)) else epoch
                    epoch = 0 if epoch is None or (isinstance(epoch, float) and isnan(epoch)) else epoch
                    period = None if period is None or (isinstance(period, float) and isnan(period)) else period
            except DBException:
                raise
            except Exception as e:
                logging.warning(f'request_asassn request lightcurve exception {e}')
                raise DBException(f'It seems that the star Gaia DR3 {gaia_id} was not found in the ASAS-SN database')
            # client.catalogs.master_list
            _store_in_cache(gaia_id, lc_df, epoch, period)

    # mask = lc_df['phot_filter']
    try:
        df = lc_df[lc_df['phot_filter'] == band][['jd', 'flux', 'flux_err']]
        if os.getenv('CUT_ASASSN'):
            df = df[:5]
            # df['flux_err'] = 0.5
        lightcurve = cook_lightcurve(df, timescale='tcg',  # todo Check
                                     flux_unit='', flux_err_unit='',
                                     epoch_jd=epoch, period_day=period)
        period_unit = None if not period else 'day'
        metadata = {'gaia_id': gaia_id, 'epoch': epoch, 'period': period, 'period_unit': period_unit, 'band': band}
    except Exception as e:
        logging.error(f'load_asassn_lightcurve exception: {type(e).__name__} {e}')
        raise PipeException(f'GAIA DR3 {gaia_id}: ASAS-SN data structure is invalid')
    return dict(metadata=metadata, lightcurve=lightcurve)


if __name__ == '__main__':
    try:
        jdict = load_asassn_lightcurve(gaia_id_no_ephem, band='V')
        print(jdict['metadata'])
    except Exception as ee:
        print(ee)
    print('AHA!')
