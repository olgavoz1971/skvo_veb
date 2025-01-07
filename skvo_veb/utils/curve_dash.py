import logging

import pandas
import pandas as pd
import json

from astropy.table import Table
from astropy.time import Time

from skvo_veb.utils.my_tools import PipeException
from astropy import units as u


def astropy_init(unit_str: str):
    try:
        return u.Unit(unit_str)
    except ValueError:
        return u.Unit()


# class UnitWrapper:
#     def __init__(self, unit_str: str | None):
#         self.unit_str = unit_str
#
#     @property
#     def astropy_unit(self):
#         try:
#             return u.Unit(self.unit_str)
#         except ValueError:
#             return u.Unit()  # Default unit if invalid


class CurveDash:
    """
    Class deals with lightcurve data. It stores, saves, serializes and restores lightcurves with units
    and related metadata. Lightcurve is stored as pandas.DataFrame
    """

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

    # Combine both dictionaries into a single class-level dictionary
    format_dict = {**_format_dict_text, **_format_dict_bytes}

    def __init__(self, serialized: str | None = None,
                 jd=None, flux=None, flux_err=None,
                 name: str = '', gaia_id=None,
                 band='',
                 time_unit: str = '', flux_unit: str = '',
                 timescale: str | None = None,
                 period: float | None = None, period_unit: str = '',
                 epoch: float | None = None,
                 cross_ident=None):
        """
        Initializes an instance of the class, allowing the creation of a lightcurve from a
        JSON string or directly from lists of time (jd) and flux values. The initialized
        object will have a lightcurve attribute defined as a Pandas DataFrame, either
        deserialized from the provided JSON string or built directly from the input data.

        :param serialized: A JSON string representation of the lightcurve data.
            If provided, the lightcurve attribute is initialized using this data.
        :type serialized: str | None
        :param jd: A column of Julian dates representing time points of the lightcurve.
            Only used if `js_lightcurve` is not provided.
        :param flux: A column of flux values corresponding to the Julian dates in `jd`.
            Only used if `js_lightcurve` is not provided.
        """
        self.lightcurve: pandas.DataFrame | None = None
        if serialized is not None:
            # Restore fro serialized data
            try:
                di = json.loads(serialized)
                # self.lightcurve = pd.DataFrame.from_dict(di.get('lightcurve'))
                lightcurve_dict = di.get('lightcurve')
                self.lightcurve = pd.DataFrame(data=lightcurve_dict['data'], columns=lightcurve_dict['columns'])
                self.metadata = di.get('metadata')
            except Exception as e:
                logging.warning(f'curve_dash.__init__: {e}')
                raise PipeException('CurveDash init: unappropriated serialized data')
        elif (jd is not None) and (flux is not None):
            # Create structures from the scratch
            if flux_err is None:
                flux_err = -1 * flux / flux  # todo: find a better solution
            df = pd.DataFrame({'jd': jd, 'flux': flux, 'flux_err': flux_err})
            df.loc[:, 'selected'] = 0  #
            # create permanent index. Keep it forever, protect against reindexing; important when cleaning data
            df.loc[:, 'perm_index'] = df.index
            if period is not None and epoch is not None:
                df.loc[:, 'phase'] = self.calc_phase(df['jd'], epoch, period, period_unit)
            self.lightcurve = df
            self.metadata: dict = {'name': name, 'gaia_id': gaia_id, 'band': band, 'cross_ident': cross_ident,
                                   'time_unit': time_unit, 'timescale': timescale,
                                   'flux_unit': flux_unit, 'period': period, 'period_unit': period_unit,
                                   'epoch': epoch,
                                   'folded_view': 1}

    def serialize(self):
        """
        Warning! This serialization approach is used by lightcurve_gaia.py and lightcurve_asassn etc.
        in JavaScript clientside callbacks, so I don't recommend changing it unless absolutely necessary
        """
        lc = self.lightcurve.to_dict(orient='split', index=False)
        metadata = self.metadata
        return json.dumps({'lightcurve': lc, 'metadata': metadata})

    def serialize_lightcurve(self):
        return json.dumps(self.lightcurve.to_dict())

    @property
    def folded_view(self):
        if self.metadata is not None:
            return self.metadata.get('folded_view')

    @folded_view.setter
    def folded_view(self, value):
        if self.metadata is not None:
            self.metadata['folded_view'] = value

    @staticmethod
    def calc_phase(time_arr, epoch_jd: float | None, period: float | None, period_unit='d'):
        # noinspection PyUnresolvedReferences
        period_day = (period * astropy_init(period_unit)).to(u.day)
        phase = ((time_arr - (0 if epoch_jd is None else epoch_jd)) / (1 if period_day is None else period_day)) % 1
        return phase

    @staticmethod
    def get_format_list() -> list[str]:
        """
        :return: a list of all supported table formats.
        """
        return list(CurveDash.format_dict.keys())

    @staticmethod
    def get_file_extension(table_format: str) -> str:
        """
        :return:  File extension corresponding to the table_format, or 'dat' if not found
        """
        return CurveDash.format_dict.get(table_format, 'dat')

    @property
    def flux(self):
        return self.lightcurve.get('flux') if self.lightcurve is not None else None

    @property
    def jd(self):
        return self.lightcurve.get('jd') if self.lightcurve is not None else None

    @property
    def flux_unit(self):
        return astropy_init(self.metadata.get('flux_unit'))

    @property
    def time_unit(self):
        return astropy_init(self.metadata.get('time_unit'))

    @property
    def timescale(self):
        return self.metadata.get('timescale')

    @property
    def period(self):
        return self.metadata.get('period')

    @property
    def period_unit(self):
        return astropy_init(self.metadata.get('period_unit'))

    @property
    def epoch(self):
        return self.metadata.get('epoch')

    @property
    def gaia_id(self):
        return self.metadata.get('gaia_id')

    @property
    def band(self):
        return self.metadata.get('band')

    def download(self, table_format='ascii.ecsv') -> bytes:
        """
        Write astropy Table into some string. The io.StringIO or io.BytesIO mimics the output file for Table.write()
        The specific IO-type depends on the desirable output format, i.e. on the writer type:
        astropy.io.ascii.write, astropy.io.fits.write, astropy.io.votable.write
        We use BytesIO for binary format (including xml-type votable, Yes!) and StringIO for the text formats.
        If you know the better way, please tell me
        """
        import io
        if self.lightcurve is None:
            raise PipeException(f'CurveDash.download: Empty lightcurve')
        if table_format in self._format_dict_text:
            my_weird_io = io.StringIO()
        elif table_format in self._format_dict_bytes:
            my_weird_io = io.BytesIO()
        else:
            raise PipeException(f'Unsupported format {table_format}\n Valid formats: {str(self.format_dict.keys())}')
        tab = Table.from_pandas(self.lightcurve)
        tab['flux'].unit = self.flux_unit
        # u.Unit(self.metadata.get('flux_unit'))
        tab['flux_err'].unit = self.flux_unit
        tab['time'] = Time(tab['jd'], format='jd', scale=self.timescale)
        tab.remove_column('jd')

        if table_format == 'votable' or table_format == 'pandas.json':
            tab['jd'] = tab['time'].jd
            selected_columns = ['jd', 'phase', 'flux', 'flux_err']
        else:
            selected_columns = ['time', 'phase', 'flux', 'flux_err']

        tab = tab[[col for col in selected_columns if col in tab.colnames]]
        tab.write(my_weird_io, format=table_format, overwrite=True)
        # self.lightcurve.write(my_weird_io, format=table_format, overwrite=True)
        my_weird_string = my_weird_io.getvalue()
        if isinstance(my_weird_string, str):
            my_weird_string = bytes(my_weird_string, 'utf-8')
        my_weird_io.close()  # todo Needed?

        return my_weird_string
