import os
import urllib

import numpy as np
import xarray as xr

import astropy.units as u
from shone.config import shone_dir


def cache_hitran_cia_files(urls=None):
    cia_dir = os.path.abspath(os.path.join(shone_dir, 'cia'))
    os.makedirs(cia_dir, exist_ok=True)

    if urls is None:
        # browse options here: https://hitran.org/cia/
        urls = [
            "https://hitran.org/data/CIA/H2-H2_2011.cia",
            "https://hitran.org/data/CIA/H2-He_2011.cia",
            "https://hitran.org/data/CIA/H2-H_2011.cia"
        ]

    paths = []

    for url in urls:
        filename = url.split('/')[-1]
        cia_path = os.path.join(cia_dir, filename)
        nc_path = os.path.join(cia_dir, filename.replace('.cia', '.nc'))

        if not os.path.exists(nc_path):
            urllib.request.urlretrieve(url, cia_path)

            ds = parse_hitran_cia(cia_path)
            ds.to_netcdf(nc_path)
        os.remove(cia_path)


def parse_hitran_cia(path):
    system = os.path.basename(path).split('_')[0]
    year = os.path.basename(path).split('_')[1].split('.')[0]
    lines = []
    for i, line in enumerate(
            open(path).read().splitlines()
    ):
        # the 5th character is empty only if this line
        # is a header line:
        if line[5] == ' ':
            lines.append(
                [i, line.split()]
            )

    start_line = lines[0][0] + 1
    stop_line = lines[1][0]
    nrows = stop_line - start_line

    freqs = np.loadtxt(
        path, delimiter=None,
        skiprows=start_line,
        max_rows=nrows,
        usecols=[0]
    )

    temperature = np.array([float(line[4]) for (i, line) in lines])

    absorption_coeff = np.loadtxt(
        path, delimiter=None, comments='H', usecols=[1]
    ).reshape((len(lines), len(freqs)))
    wavelength = (freqs / u.cm).to(u.um, u.spectral()).value

    ds = xr.Dataset(
        data_vars=dict(
            absorption_coeff=(
                ["temperature", "wavelength"],
                absorption_coeff[:, ::-1].astype(np.float64)
            )
        ),
        coords=dict(
            temperature=(["temperature"], temperature),
            wavelength=wavelength[::-1]
        ),
        attrs={
            "system": system,
            "source": "HITRAN",
            "year": year,
        }
    )

    return ds