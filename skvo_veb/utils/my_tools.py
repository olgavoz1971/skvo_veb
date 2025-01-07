import logging
import re
import time


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        logging.debug(f'func:{f.__name__} args:{args, kw} took: {(te - ts):2.4f} sec')
        return result
    return timed


class DBException(Exception):
    # print(f'My exception {Exception} occurred')
    pass


class PipeException(Exception):
    # print(f'My exception {Exception} occurred')
    pass


def is_like_gaia_id(value: str):
    return bool(re.fullmatch(r'\d+', value))


def tcb2tdb(jd_tcb):
    """
    Conversion between Barycentric Coordinate Time (TCB) and Barycentric Dynamical Time (TBD)

    Gaia photometric series use TCB, Kepler and TESS - TDB scale.
    The transformation between TCB and TDB time scales is given by Berthier et al. (2021)
    and Klioner et al. (2010) following IAU resolution 2006 B31
    TDB = TCB − L_B(JD_TCB − 2 443 144.500 3725) × 86400 s − 6.55 × 10−5 s,
    where the time is expressed in seconds, and L_B = 1.550 519 768 × 10−8.
    During the period covered by the Gaia DR3, the difference TDB − TCB is ∼ −19 s
    https://arxiv.org/pdf/2206.05561.pdf

    Astropy can also do this conversion:
    https://docs.astropy.org/en/stable/time/#convert-time-scale

    :param jd_tcb: TCB in jd,
    :return: TBD-TCB in seconds
    """
    L_B = 1.550519768E-08
    dt = -L_B * (jd_tcb - 2443144.5003725) * 86400 - 6.55 * 1.0E-05
    return dt


def explain_exception(e):
    # return f'{type(e).__name__}: {e}'
    return repr(e)


def main_name(cross_ident: dict):
    return cross_ident['vsx'] if cross_ident['vsx'] is not None else cross_ident['simbad'] \
        if cross_ident['simbad'] is not None else f'Gaia DR3 {cross_ident["gaia_id"]}'
