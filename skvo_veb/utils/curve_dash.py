import logging
import io

import pandas
import pandas as pd
import json

from astropy.table import Table
from astropy.time import Time

from skvo_veb.utils.my_tools import PipeException
from astropy import units as u


def astropy_init(unit_str: str):
    if not unit_str:
        return u.Unit()
    try:
        return u.Unit(unit_str)
    except ValueError:
        return u.Unit()


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
    # format_dict = {**_format_dict_text, **_format_dict_bytes}
    format_dict = _format_dict_text | _format_dict_bytes
    extension_dict = {v: k for k, v in format_dict.items()}

    # def __init__(self, serialized: str | None = None,
    #              jd=None, flux=None, flux_err=None,
    #              flux_correction: str | None = None,
    #              name: str = '', lookup_name: str | None = None,  gaia_id=None,
    #              title: str = '',
    #              band='',
    #              time_unit: str = '', flux_unit: str = '',
    #              timescale: str | None = None,  # one pf astropy.time Scale or 'hjd' for Heliocentric julian
    #              period: float | None = None, period_unit: str = 'd',
    #              epoch: float | None = 0,
    #              cross_ident=None, folded_view=0):
    #     """
    #     Initializes an instance of the class, allowing the creation of a lightcurve from a
    #     JSON string or directly from lists of time (jd) and flux values. The initialized
    #     object will have a lightcurve attribute defined as a Pandas DataFrame, either
    #     deserialized from the provided JSON string or built directly from the input data.
    #
    #     :param serialized: A JSON string representation of the lightcurve data.
    #         If provided, the lightcurve attribute is initialized using this data.
    #     :type serialized: str | None
    #     :param jd: A column of Julian dates representing time points of the lightcurve.
    #         Only used if `js_lightcurve` is not provided.
    #     :param flux: A column of flux values corresponding to the Julian dates in `jd`.
    #         Only used if `js_lightcurve` is not provided.
    #     """
    #     self.lightcurve: pandas.DataFrame | None = None
    #     self.metadata = None
    #     if serialized is not None:
    #         # Restore from serialized data
    #         # try upload from the file object
    #
    #         try:
    #             di = json.loads(serialized)
    #             if not di:  # empty dictionary
    #                 return  # create an empty lcd
    #             # self.lightcurve = pd.DataFrame.from_dict(di.get('lightcurve'))
    #             lightcurve_dict = di.get('lightcurve')
    #             self.lightcurve = pd.DataFrame(data=lightcurve_dict['data'], columns=lightcurve_dict['columns'])
    #             self.metadata = di.get('metadata')
    #         except Exception as e:
    #             logging.warning(f'curve_dash.__init__: {e}')
    #             raise PipeException('CurveDash init: unappropriated serialized data')
    #     elif (jd is not None) and (flux is not None):
    #         # Create structures from the scratch
    #         if flux_err is None:
    #             flux_err = flux / flux  # todo: find a better solution
    #         df = pd.DataFrame({'jd': jd, 'flux': flux, 'flux_err': flux_err})
    #         # Using loc to avoid SettingWithCopyWarning and ensure in -place DataFrame update
    #         df.loc[:, 'selected'] = 0
    #         # create permanent index. Keep it forever, protect against reindexing; important when cleaning data:
    #         df.loc[:, 'perm_index'] = df.index
    #         df.loc[:, 'phase'] = 0.0
    #         self.lightcurve = df
    #         if lookup_name and lookup_name == name:
    #             lookup_name = ''
    #         self.metadata: dict = {'name': name, 'lookup_name': lookup_name, 'gaia_id': gaia_id, 'band': band,
    #                                'cross_ident': cross_ident,
    #                                'time_unit': time_unit, 'timescale': timescale,
    #                                'title': title,
    #                                'flux_correction': flux_correction,
    #                                'flux_unit': flux_unit, 'period': period, 'period_unit': period_unit,
    #                                'epoch': epoch,
    #                                'folded_view': folded_view}
    #         self.recalc_phase()  # recalc phase after period and epoch setting

    # def __init__(self, **kwargs):
    #     self.lightcurve: pandas.DataFrame | None = None
    #     self.metadata = None

    def __init__(self, jd=None, flux=None, flux_err=None,
                 flux_correction: str | None = None,
                 name: str = '', lookup_name: str | None = None, gaia_id=None,
                 title: str = '',
                 band='',
                 time_unit: str = '', flux_unit: str = '',
                 timescale: str | None = None,  # one pf astropy.time Scale or 'hjd' for Heliocentric julian
                 period: float | None = None, period_unit: str = 'd',
                 epoch: float | None = 0,
                 cross_ident=None, folded_view=0):
        """
        Initializes an instance of the class, allowing the creation of a lightcurve
        directly from lists of time (jd) and flux values. The initialized
        object will have a lightcurve attribute defined as a Pandas DataFrame

        :param jd: A column of Julian dates representing time points of the lightcurve.
            Only used if `js_lightcurve` is not provided.
        :param flux: A column of flux values corresponding to the Julian dates in `jd`.
            Only used if `js_lightcurve` is not provided.
        """

        self.lightcurve: pandas.DataFrame | None = None
        self.metadata: dict | None = None
        if (jd is not None) and (flux is not None):
            # Create structures from the scratch
            if flux_err is None:
                flux_err = flux / flux  # todo: find a better solution
            df = pd.DataFrame({'jd': jd, 'flux': flux, 'flux_err': flux_err})
            # Using loc to avoid SettingWithCopyWarning and ensure in -place DataFrame update
            df.loc[:, 'selected'] = 0
            # create permanent index. Keep it forever, protect against reindexing; important when cleaning data:
            df.loc[:, 'perm_index'] = df.index
            df.loc[:, 'phase'] = 0.0
            self.lightcurve = df
            if lookup_name and lookup_name == name:
                lookup_name = ''
            self.metadata: dict = {'name': name, 'lookup_name': lookup_name, 'gaia_id': gaia_id, 'band': band,
                                   'cross_ident': cross_ident,
                                   'time_unit': time_unit, 'timescale': timescale,
                                   'title': title,
                                   'flux_correction': flux_correction,
                                   'flux_unit': flux_unit, 'period': period, 'period_unit': period_unit,
                                   'epoch': epoch,
                                   'folded_view': folded_view}
            self.recalc_phase()  # recalc phase after period and epoch setting

    @classmethod
    def from_serialized(cls, serialized: str):
        """
        Initializes an instance of the class, allowing the recreation of a lightcurve from a
        JSON string. This is useful for restoring an object from dcc.Store data
        :param serialized: A JSON string representation of the lightcurve data.
        :type serialized: str
        """
        try:
            self = cls()
            if not serialized:
                return self     # create an empty lcd
            di = json.loads(serialized)
            if not di:  # empty dictionary
                return self     # create an empty lcd
            lightcurve_dict = di.get('lightcurve')
            self.lightcurve = pd.DataFrame(data=lightcurve_dict['data'], columns=lightcurve_dict['columns'])
            self.metadata = di.get('metadata')
            return self
        except Exception as e:
            logging.warning(f'curve_dash.__init__: {e}')
            raise PipeException('CurveDash init: unappropriated serialized data')

    @classmethod
    def from_file(cls, file_obj: io.BytesIO, extension: str):
        t = Table.read(file_obj, format=CurveDash.get_table_format(extension))
        flux_unit = str(getattr(t['flux'], 'unit', ''))
        metadata = getattr(t, 'meta', None)
        self = cls(jd=t['time'].jd, flux=t['flux'], flux_err=t['flux_err'], flux_unit=flux_unit, time_unit='d')
        if metadata:
            self.metadata = self.metadata | metadata
        return self

    # def __init__(self, serialized: str | None = None,
    #              jd=None, flux=None, flux_err=None,
    #              flux_correction: str | None = None,
    #              name: str = '', lookup_name: str | None = None,  gaia_id=None,
    #              title: str = '',
    #              band='',
    #              time_unit: str = '', flux_unit: str = '',
    #              timescale: str | None = None,  # one pf astropy.time Scale or 'hjd' for Heliocentric julian
    #              period: float | None = None, period_unit: str = 'd',
    #              epoch: float | None = 0,
    #              cross_ident=None, folded_view=0):
    #     """
    #     Initializes an instance of the class, allowing the creation of a lightcurve from a
    #     JSON string or directly from lists of time (jd) and flux values. The initialized
    #     object will have a lightcurve attribute defined as a Pandas DataFrame, either
    #     deserialized from the provided JSON string or built directly from the input data.
    #
    #     :param serialized: A JSON string representation of the lightcurve data.
    #         If provided, the lightcurve attribute is initialized using this data.
    #     :type serialized: str | None
    #     :param jd: A column of Julian dates representing time points of the lightcurve.
    #         Only used if `js_lightcurve` is not provided.
    #     :param flux: A column of flux values corresponding to the Julian dates in `jd`.
    #         Only used if `js_lightcurve` is not provided.
    #     """
    #     self.lightcurve: pandas.DataFrame | None = None
    #     self.metadata = None
    #     if serialized is not None:
    #         # Restore from serialized data
    #         # try upload from the file object
    #
    #         try:
    #             di = json.loads(serialized)
    #             if not di:  # empty dictionary
    #                 return  # create an empty lcd
    #             # self.lightcurve = pd.DataFrame.from_dict(di.get('lightcurve'))
    #             lightcurve_dict = di.get('lightcurve')
    #             self.lightcurve = pd.DataFrame(data=lightcurve_dict['data'], columns=lightcurve_dict['columns'])
    #             self.metadata = di.get('metadata')
    #         except Exception as e:
    #             logging.warning(f'curve_dash.__init__: {e}')
    #             raise PipeException('CurveDash init: unappropriated serialized data')
    #     elif (jd is not None) and (flux is not None):
    #         # Create structures from the scratch
    #         if flux_err is None:
    #             flux_err = flux / flux  # todo: find a better solution
    #         df = pd.DataFrame({'jd': jd, 'flux': flux, 'flux_err': flux_err})
    #         # Using loc to avoid SettingWithCopyWarning and ensure in -place DataFrame update
    #         df.loc[:, 'selected'] = 0
    #         # create permanent index. Keep it forever, protect against reindexing; important when cleaning data:
    #         df.loc[:, 'perm_index'] = df.index
    #         df.loc[:, 'phase'] = 0.0
    #         self.lightcurve = df
    #         if lookup_name and lookup_name == name:
    #             lookup_name = ''
    #         self.metadata: dict = {'name': name, 'lookup_name': lookup_name, 'gaia_id': gaia_id, 'band': band,
    #                                'cross_ident': cross_ident,
    #                                'time_unit': time_unit, 'timescale': timescale,
    #                                'title': title,
    #                                'flux_correction': flux_correction,
    #                                'flux_unit': flux_unit, 'period': period, 'period_unit': period_unit,
    #                                'epoch': epoch,
    #                                'folded_view': folded_view}
    #         self.recalc_phase()  # recalc phase after period and epoch setting

    def serialize(self):
        """
        Warning! This serialization approach is used by lightcurve_gaia.py and lightcurve_asassn etc.
        in JavaScript clientside callbacks, so I don't recommend changing it unless absolutely necessary
        """
        if self.lightcurve is None or self.metadata is None:
            return '{}'
        lc = self.lightcurve.to_dict(orient='split', index=False)
        metadata = self.metadata
        return json.dumps({'lightcurve': lc, 'metadata': metadata})

    @property
    def title(self):
        return self.metadata.get('title') if self.metadata else None

    @title.setter
    def title(self, value: str):
        if self.metadata is not None:
            self.metadata['title'] = value

    @property
    def name(self):
        return self.metadata.get('name') if self.metadata else None

    @property
    def lookup_name(self):
        return self.metadata.get('lookup_name') if self.metadata else None

    @property
    def folded_view(self):
        return self.metadata.get('folded_view') if self.metadata else None

    @folded_view.setter
    def folded_view(self, value):
        if self.metadata is not None:
            self.metadata['folded_view'] = value

    @property
    def flux_correction(self):
        if self.metadata is not None:
            if self.metadata.get('flux_correction') is not None:
                return self.metadata.get('flux_correction')
        return ''

    def recalc_phase(self):
        if self.period is not None and self.epoch is not None:
            df = self.lightcurve
            # Using loc to avoid SettingWithCopyWarning and ensure inplace DataFrame update
            df.loc[:, 'phase'] = self.calc_phase(df['jd'], self.epoch, self.period, self.period_unit)
            self.lightcurve = df

    @staticmethod
    def calc_phase(time_arr, epoch_jd: float | None, period: float | None, period_unit: str):
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

    @staticmethod
    def get_table_format(file_extension: str) -> str:
        """
        :return:  Table format corresponding to the file_extension, or None if not found
        """
        return CurveDash.extension_dict.get(file_extension, None)

    @staticmethod
    def get_extension_list():
        return [CurveDash.get_file_extension(f) for f in CurveDash.get_format_list()]

    @property
    def flux(self):
        # It's important to leave 'is not None' here, because flux is pandas.Series, we can't ask 'if pandas.Series'
        return self.lightcurve.get('flux') if self.lightcurve is not None else None

    @property
    def flux_err(self):
        return self.lightcurve.get('flux_err') if self.lightcurve is not None else None

    @property
    def jd(self):
        return self.lightcurve.get('jd') if self.lightcurve is not None else None

    @property
    def phase(self):
        return self.lightcurve.get('phase') if self.lightcurve is not None else None

    @property
    def perm_index(self):
        """
        Unique identifier of each, protected from cleaning and all kinds of point reordering.
        It is stored in customdata of the plotly figure
        :return:
        """
        return self.lightcurve.get('perm_index') if self.lightcurve is not None else None

    @property
    def flux_unit(self):
        return self.metadata.get('flux_unit') if self.metadata else ''

    @property
    def flux_unit_ap(self):
        # return astropy.unit if it is convertable
        return astropy_init(self.metadata.get('flux_unit')) if self.metadata else None

    @property
    def time_unit(self):
        return self.metadata.get('time_unit') if self.metadata else None

    @property
    def time_unit_ap(self):
        # return astropy.unit if it is convertable
        return astropy_init(self.metadata.get('time_unit')) if self.metadata else None

    @property
    def timescale(self):
        return self.metadata.get('timescale') if self.metadata else None

    @property
    def period(self):
        return self.metadata.get('period') if self.metadata is not None else None

    @period.setter
    def period(self, value):
        if self.metadata is not None:
            self.metadata['period'] = value
            self.recalc_phase()

    @property
    def period_unit(self):
        return self.metadata.get('period_unit') if self.metadata else None

    @period_unit.setter
    def period_unit(self, value):
        if self.metadata is not None:
            self.metadata['period_unit'] = value
            self.recalc_phase()

    @property
    def period_unit_ap(self):
        return astropy_init(self.metadata.get('period_unit')) if self.metadata else None

    @property
    def epoch(self):
        return self.metadata.get('epoch') if self.metadata else None

    @epoch.setter
    def epoch(self, value):
        if self.metadata is not None:
            self.metadata['epoch'] = value
            self.recalc_phase()

    @property
    def gaia_id(self):
        return self.metadata.get('gaia_id') if self.metadata else None

    @property
    def band(self):
        return self.metadata.get('band') if self.metadata else None

    def cut(self, left_border, right_border):
        """
        Remove a piece of lightcurve between  left_border and right_border along the time axis
        :param left_border: start_time
        :param right_border: end_time
        """
        df = self.lightcurve
        self.lightcurve = df[(df['jd'] < left_border) | (df['jd'] > right_border)]

    def keep(self, left_border, right_border):
        """
        Keep only a piece of lightcurve (remove the rest) between left_border and right_border along the time axis
        :param left_border: start_time
        :param right_border: end_time
        """
        df = self.lightcurve
        self.lightcurve = df[(df['jd'] >= left_border) & (df['jd'] <= right_border)]

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
        tab['flux'].unit = self.flux_unit_ap
        # u.Unit(self.metadata.get('flux_unit'))
        tab['flux_err'].unit = self.flux_unit_ap
        timescale = self.timescale if self.timescale != 'hjd' else None
        tab['time'] = Time(tab['jd'], format='jd', scale=timescale)
        tab.remove_column('jd')

        if table_format == 'votable' or table_format == 'pandas.json':
            tab['jd'] = tab['time'].jd
            selected_columns = ['jd', 'phase', 'flux', 'flux_err']
        else:
            selected_columns = ['time', 'phase', 'flux', 'flux_err']

        tab = tab[[col for col in selected_columns if col in tab.colnames]]
        tab.meta = self.metadata
        tab.write(my_weird_io, format=table_format, overwrite=True)
        # self.lightcurve.write(my_weird_io, format=table_format, overwrite=True)
        my_weird_string = my_weird_io.getvalue()
        if isinstance(my_weird_string, str):
            my_weird_string = bytes(my_weird_string, 'utf-8')
        my_weird_io.close()  # todo Needed?

        return my_weird_string

    def append(self, other: "CurveDash") -> None:
        # todo: append title
        if not isinstance(other, CurveDash):
            raise TypeError("The input object must be an instance of CurveDash.")
        if (self.lightcurve is None) or self.lightcurve.empty:
            self.lightcurve = other.lightcurve.copy()
        elif not other.lightcurve.empty:
            self.lightcurve = pd.concat([self.lightcurve, other.lightcurve], ignore_index=True)
