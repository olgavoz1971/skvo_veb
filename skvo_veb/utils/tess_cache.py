import dill
import logging
import tarfile
import os
import hashlib

import lightkurve
from lightkurve import TessTargetPixelFile, TessLightCurve


def _get_cache_filename(prefix, extension='dill', **kwargs, ):
    """Create unique filename based on input parameters"""
    print(f'_get_cache_filename: {prefix=} {kwargs=}')
    cache_dir = os.getenv('TESS_CACHE_DIR')
    print(f'_get_cache_filename: {cache_dir=}')
    if not cache_dir:
        logging.warning('Environmental variable TESS_CACHE_DIR is not specified')
        return None
    unique_key = f"{prefix}_{hashlib.md5(str(kwargs).encode()).hexdigest()}.{extension}"
    # unique_key = f"{prefix}_{hashlib.md5(str(kwargs).encode()).hexdigest()}.dill"
    return os.path.join(cache_dir, unique_key)
    # return CACHE_DIR / unique_key


def save(data, prefix, **kwargs):
    """Save results into cache"""
    filename = _get_cache_filename(prefix, **kwargs)
    # joblib.dump(data, filename)
    with open(filename, "wb") as f:
        # pickle.dump(data, f)
        dill.dump(data, f)


def save_tpf_fits(data, **kwargs):
    """Save tpf data into cache as a fits-file"""
    prefix = 'tpf_data'
    filename = _get_cache_filename(prefix, extension='fits', **kwargs)
    data.to_fits(filename, overwrite=True)


def load_tpf_fits(**kwargs):
    prefix = 'tpf_data'
    filename = _get_cache_filename(prefix, extension='fits', **kwargs)
    if os.path.exists(filename):
        return TessTargetPixelFile(str(filename))
    return None


def load_ffi_fits(**kwargs):
    prefix = 'ffi_data'
    filename = _get_cache_filename(prefix, extension='fits', **kwargs)
    if os.path.exists(filename):
        return TessTargetPixelFile(str(filename))
    return None


def save_ffi_fits(data, **kwargs):
    """Save ffi data into cache as a fits-file"""
    prefix = 'ffi_data'
    filename = _get_cache_filename(prefix, extension='fits', **kwargs)
    data.to_fits(filename, overwrite=True)


def save_lc_fits(data, **kwargs):
    """Save ffi data into cache as a fits-file"""
    prefix = 'lc'
    filename = _get_cache_filename(prefix, extension='fits', **kwargs)
    data.to_fits(filename, overwrite=True)


def load_lc_fits(**kwargs):
    prefix = 'lc'
    filename = _get_cache_filename(prefix, extension='fits', **kwargs)
    if os.path.exists(filename):
        return TessLightCurve.read(str(filename))
    return None


def load(prefix, **kwargs):
    """Loads result from cache if exists"""
    filename = _get_cache_filename(prefix, **kwargs)
    if os.path.exists(filename):
        # return joblib.load(filename)
        with open(filename, "rb") as f:
            # return pickle.load(f)
            return dill.load(f)
    return None


def save_lightcurve_collection(lc_collection, tar_filename):
    with tarfile.open(tar_filename, "w") as tar:
        for i, lc in enumerate(lc_collection):
            # Сохраняем каждую кривую как отдельный FITS файл
            lc_filename = f"lightcurve_{i}.fits"
            lc.to_fits().writeto(lc_filename)

            # Добавляем этот файл в архив
            tar.add(lc_filename)

            # Удаляем временный FITS файл
            os.remove(lc_filename)


def load_lightcurve_collection(tar_filename):
    lc_collection = []
    with tarfile.open(tar_filename, "r") as tar:
        for member in tar.getmembers():
            if member.isfile():
                # Извлекаем и читаем FITS файл
                with tar.extractfile(member) as file:
                    lc = lightkurve.LightCurve.read(file)
                    lc_collection.append(lc)
    return lc_collection
