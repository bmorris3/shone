import os
import urllib

import numpy as np
import xarray as xr

import astropy.units as u
from shone.config import shone_dir

from jax import jit, vmap, lax, numpy as jnp
from shone.constants import k_B

__all__ = [
    'h_minus_continuum',
    'download_hitran_cia_files'
]


@jit
def h_minus_continuum(
    wavelength, temperature, number_density_e, number_density_h
):
    """
    Continuum opacity from H-.

    From John (1988) [1]_.

    Parameters
    ----------

    wavelength : array
        Wavelength [µm].
    temperature : array
        Temperature [K].
    number_density_e: array
        Number density of electrons [cm^-3].
    number_density_h: array
        Number density of hydrogen [cm^-3].

    Returns
    -------
    alpha : array
        Absorption coefficient [cm^-1]

    References
    ----------
    .. [1] `John, T. L. 1988, Astronomy and Astrophysics, 193, 189
           <https://ui.adsabs.harvard.edu/abs/1988A%26A...193..189J/abstract>`_
    """

    # first, compute the cross-sections [cm4/dyn]
    kappa_bf_0 = vmap(bound_free_absorption, (None, 0), 0)
    kappa_ff_0 = vmap(free_free_absorption, (None, 0), 0)
    kappa_bf_1 = vmap(kappa_bf_0, (0, None), 0)
    kappa_ff_1 = vmap(kappa_ff_0, (0, None), 0)
    kappa_bf = kappa_bf_1(wavelength, temperature)
    kappa_ff = kappa_ff_1(wavelength, temperature)

    electron_pressure = (
        number_density_e * k_B * temperature
    )  # [dyn/cm2]

    absorption_coeff = (
        (kappa_bf + kappa_ff) *
        electron_pressure * number_density_h
    )  # [cm^-1]

    return absorption_coeff.T


@jit
def bound_free_absorption(wavelength, temperature):
    """
    Bound free absorption of H-.

    Parameters
    ----------
    wavelength : array
        Wavelength [µm].
    temperature : array
        Temperature [K].

    Returns
    -------
    alpha : array
        Absorption coefficient [cm4/dyn]
    """
    # alpha has a value of 1.439e4 micron-1 K-1, the value stated in John (1988) is wrong
    alpha = 1.439e4  # [µm^-1 K^-1]
    lambda_0 = 1.6419  # photo-detachment threshold

    #   //tabulated constant from John (1988)
    def f(wavelength):
        C_n = jnp.vstack(
            [jnp.arange(7),
             [0.0, 152.519, 49.534, -118.858, 92.536, -34.194, 4.982]]
        ).T

        def body_fun(val, x):
            i, C_n_i = x
            return val, val + C_n_i * jnp.power(
                jnp.clip(
                    1.0 / wavelength -
                    1.0 / lambda_0,
                    a_min=0,
                    a_max=None
                ),
                (i - 1) / 2.0,
            )

        return lax.scan(
            body_fun, jnp.zeros_like(wavelength), C_n
        )[-1].sum(0)

    # photo-detachment cross-section:
    kappa_bf = (
        1e-18 * wavelength**3 * jnp.power(
            jnp.clip(1.0 / wavelength - 1.0 / lambda_0, a_min=0, a_max=None), 1.5
        ) * f(wavelength)
    )

    kappa_bf = jnp.where(
        (wavelength <= lambda_0) & (wavelength > 0.125),
        (0.750 * jnp.power(temperature, -2.5) *
            jnp.exp(alpha / lambda_0 / temperature) *
            (1.0 - jnp.exp(-alpha / wavelength / temperature)) * kappa_bf
        ), 0,
    )
    return kappa_bf


@jit
def free_free_absorption(wavelength, temperature):
    """
    Free-free absorption of H-.

    From John (1988).

    Parameters
    ----------
    wavelength : array
        Wavelength [µm].
    temperature : array
        Temperature [K].

    Returns
    -------
    alpha : array
        Absorption coefficient [cm4/dyn]
    """
    # to follow his notation (which starts at an index of 1),
    # the 0-index components are 0 for wavelengths larger than 0.3645 micron
    A_n1 = [0.0, 0.0, 2483.3460, -3449.8890, 2200.0400, -696.2710, 88.2830]
    B_n1 = [0.0, 0.0, 285.8270, -1158.3820, 2427.7190, -1841.4000, 444.5170]
    C_n1 = [0.0, 0.0, -2054.2910, 8746.5230, -13651.1050, 8624.9700, -1863.8650]
    D_n1 = [0.0, 0.0, 2827.7760, -11485.6320, 16755.5240, -10051.5300, 2095.2880]
    E_n1 = [0.0, 0.0, -1341.5370, 5303.6090, -7510.4940, 4400.0670, -901.7880]
    F_n1 = [0.0, 0.0, 208.9520, -812.9390, 1132.7380, -655.0200, 132.9850]

    # for wavelengths between 0.1823 micron and 0.3645 micron
    A_n2 = [0.0, 518.1021, 473.2636, -482.2089, 115.5291, 0.0, 0.0]
    B_n2 = [0.0, -734.8666, 1443.4137, -737.1616, 169.6374, 0.0, 0.0]
    C_n2 = [0.0, 1021.1775, -1977.3395, 1096.8827, -245.6490, 0.0, 0.0]
    D_n2 = [0.0, -479.0721, 922.3575, -521.1341, 114.2430, 0.0, 0.0]
    E_n2 = [0.0, 93.1373, -178.9275, 101.7963, -21.9972, 0.0, 0.0]
    F_n2 = [0.0, -6.4285, 12.3600, -7.0571, 1.5097, 0.0, 0.0]

    def ff(wavelength, A_n, B_n, C_n, D_n, E_n, F_n):
        x = 0

        for i in range(1, 7):
            x += jnp.power(5040.0 / temperature, (i + 1) / 2.0) * (
                wavelength**2 * A_n[i] + B_n[i] + C_n[i] / wavelength +
                D_n[i] / wavelength**2 + E_n[i] / wavelength**3 +
                F_n[i] / wavelength**4
            )

        return x * 1e-29

    kappa_ff = jnp.where(
        wavelength > 0.3645,
        ff(wavelength, A_n1, B_n1, C_n1, D_n1, E_n1, F_n1),
        0
    ) + jnp.where(
        (wavelength >= 0.1823) & (wavelength <= 0.3645),
        ff(wavelength, A_n2, B_n2, C_n2, D_n2, E_n2, F_n2),
        0,
    )

    return kappa_ff


def download_hitran_cia_files(urls=None):
    cia_dir = os.path.abspath(os.path.join(shone_dir, 'cia'))
    os.makedirs(cia_dir, exist_ok=True)

    if urls is None:
        # browse options here: https://hitran.org/cia/
        urls = [
            "https://hitran.org/data/CIA/H2-H2_2011.cia",
            "https://hitran.org/data/CIA/H2-He_2011.cia",
            "https://hitran.org/data/CIA/H2-H_2011.cia"
        ]

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
