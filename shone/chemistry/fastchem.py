import os
import platform
import sys
import warnings
from functools import partial

import numpy as np
from jax import numpy as jnp, jit
from tensorflow_probability.substrates.jax.math import batch_interp_rectilinear_nd_grid as nd_interp
from tqdm.auto import tqdm
import xarray as xr

from astropy.table import Table
from pyfastchem import (
    FastChem, FastChemInput, FastChemOutput
)

from shone.config import shone_dir
from shone.constants import bar_to_dyn_cm2, k_B
from shone.chemistry.translate import species_name_to_fastchem_name

__all__ = [
    'FastchemWrapper',
    'build_fastchem_grid',
    'get_fastchem_interpolator',
    'fastchem_species_table',
    'number_density',
    'mass_density',
    'mean_molecular_weight',
]


fastchem_grid_filename = 'fastchem_grid.nc'
cached_species_table = None


def _check_mac_arm():
    if sys.platform == 'darwin' and platform.processor() == 'arm':
        msg = (
            "It appears that you're using a Mac with one of "
            "Apple's ARM-based processors. Some results from FastChem may "
            "be inaccurate on this processor, especially at lower temperatures "
            "and pressures. For more details, see: "
            "https://github.com/NewStrangeWorlds/FastChem/issues/9"
            ""
        )
        warnings.warn(msg, UserWarning)


class FastchemWrapper:
    """
    Wrapper around pyfastchem.

    FastChem computes mixing ratios for atmospheric
    species assuming equilibrium chemistry [1]_, [2]_, [3]_, [4]_.

    References
    ----------
    .. [1] `Stock, J. W., Kitzmann, D., Patzer, A. B. C., et al. 2018,
       Monthly Notices of the Royal Astronomical Society, 479,
       865. <https://ui.adsabs.harvard.edu/abs/2018MNRAS.479..865S/abstract>`_
       doi:10.1093/mnras/sty1531
    .. [2] `Stock, J. W., Kitzmann, D., & Patzer, A. B. C. 2022,
       Monthly Notices of the Royal Astronomical Society, 517,
       4070. <https://ui.adsabs.harvard.edu/abs/2022MNRAS.517.4070S/abstract>`_
       doi:10.1093/mnras/stac2623
    .. [3] `Kitzmann, D., Stock, J. W., & Patzer, A. B. C. 2024, Monthly
       Notices of the Royal Astronomical Society, 527, 7263.
       <https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.7263K/abstract>`_
       doi:10.1093/mnras/stad3515
    .. [4] `FastChem on Github <https://github.com/exoclime/FastChem>`_.
    """
    def __init__(
        self,
        temperature,
        pressure,
        metallicity=1,
        c_to_o_ratio=0.5888,
        elemental_abundances_path=None,
        fit_coefficients_path=None,
        quiet=False
    ):
        """
        Parameters
        ----------
        temperature : array-like
            Temperature grid [K].
        pressure : array-like
            Pressure grid [bar].
        metallicity : float, optional
            M/H expressed as a linear factor (not log).
        c_to_o_ratio : float, optional
            Carbon to oxygen ratio expressed as a linear
            factor (not log). Default is the C/O ratio of
            the solar abundances from Asplund et al. (2020):
            `0.5888`.
        elemental_abundances_path : str (path-like), optional
            Path to elemental abundances for FastChem.
        fit_coefficients_path : str (path-like), optional
            Path to fit coefficients for FastChem.
        quiet : bool, optional
            Raise warnings for machine configurations that may
            produce inaccurate results. Default is False.
        """

        # Warn the user if FastChem may be inaccurate on this processor:
        if not quiet:
            _check_mac_arm()

        self.temperature = temperature
        self.pressure = pressure
        self.metallicity = metallicity
        self.c_to_o_ratio = c_to_o_ratio

        if elemental_abundances_path is None:
            elemental_abundances_path = os.path.join(
                os.path.dirname(__file__),
                '..', 'data',
                'asplund_2020.dat'
            )

        if fit_coefficients_path is None:
            fit_coefficients_path = os.path.join(
                os.path.dirname(__file__),
                '..', 'data', 'logK.dat'
            )

        fastchem = FastChem(
            elemental_abundances_path,
            fit_coefficients_path, 0
        )
        self.fastchem = fastchem
        self.solar_abundances = np.array(self.fastchem.getElementAbundances())

        # create the input and output structures for FastChem
        self._input_data = FastChemInput()
        self._input_data.temperature = self.temperature
        self._input_data.pressure = self.pressure

    def vmr(self):
        """
        Volume mixing ratio.

        Returns
        -------
        vmr : array-like
            Volume mixing ratio for each species.
        """

        # metallicity does not scale the abundance of H or He:
        skip_indices = [self.fastchem.getElementIndex(element) for element in ['H', 'He']]

        # scale the abundance of all other elements:
        multiplier = self.metallicity * np.ones_like(self.solar_abundances)
        multiplier[skip_indices] = 1

        abundances_with_metallicity = self.solar_abundances * multiplier

        if self.c_to_o_ratio is not None:
            index_C = self.fastchem.getElementIndex('C')
            index_O = self.fastchem.getElementIndex('O')

            abundances_with_metallicity[index_C] = (
                abundances_with_metallicity[index_O] * self.c_to_o_ratio
            )

        self.fastchem.setElementAbundances(abundances_with_metallicity)

        # create the input and output structures for FastChem
        output_data = FastChemOutput()

        # run FastChem on the entire p-T structure
        self.fastchem.calcDensities(self._input_data, output_data)
        n_densities = np.array(output_data.number_densities)  # [cm-3]

        gas_number_density = self.pressure * bar_to_dyn_cm2 / (k_B * self.temperature)  # [cm-3]
        vmr = n_densities / gas_number_density[:, None]

        return vmr

    def get_species(self):
        """
        Return an astropy table with names, symbols, weights and
        indices for each species in this FastChem instance.

        Returns
        -------
        species_table : `~astropy.table.Table`
            Table of FastChem species.
        """
        return _fastchem_species_table(self.fastchem)

    def get_weights(self):
        """
        Get weights for each species [AMU].
        """
        element_weights = []
        gas_species_weights = []
        idx = 0
        while self.fastchem.getElementSymbol(idx) != '':
            element_weights.append(self.fastchem.getElementWeight(idx))
            idx += 1

        while self.fastchem.getGasSpeciesSymbol(idx) != '':
            gas_species_weights.append(self.fastchem.getGasSpeciesWeight(idx))
            idx += 1

        weights = element_weights + gas_species_weights

        return np.array(weights)

    def mmr_mmw(self):
        """
        Mass mixing ratio (MMR) multiplied by the mean molecular
        weight (MMW).

        To convert to MMR, divide this result by MMW.

        Returns
        -------
        mmr_mmw : array-like
            Mass mixing ratio times mean molecular weight
            for each species.
        """
        vmr = self.vmr()
        weights_amu = self.get_weights()
        mmr_mmw = vmr * weights_amu[None, :]

        return mmr_mmw

    def get_column_index(self, fastchem_name=None, species_name=None):
        """
        Return the column of the FastChem result corresponding
        to a given species or list of species.

        Parameters
        ----------
        fastchem_name : str, or list of strings (optional)
            Species name in Hill notation (e.g. water is "H2O1"),
            as it is stored in FastChem.
        species_name : str, or list of strings (optional)
            Common species name (e.g. water is "H2O"),
            which will be converted to FastChem's
            preferred (Hill) notation.

        Returns
        -------
        idx : int, or list of ints
            Column index.
        """

        if species_name is not None and fastchem_name is None:
            if isinstance(species_name, str):
                species_name = [species_name]

            fastchem_name = [
                species_name_to_fastchem_name(name)
                for name in species_name
            ]

        indices = []
        for name in fastchem_name:
            indices.append(
                min(
                    self.fastchem.getElementIndex(name),
                    self.fastchem.getGasSpeciesIndex(name),
                )
            )
        return indices


def round_in_log(x):
    log_floor = np.floor(np.log10(x))
    pre_exponent = np.round(x * 10 ** -log_floor, 1)
    return pre_exponent * 10 ** log_floor


def build_fastchem_grid(
    temperature=None, pressure=None,
    log_m_to_h=None, log_c_to_o=None,
    n_species=523
):
    """
    Pre-compute a grid of equilibrium chemistry solutions from FastChem.

    The grid will be saved as a netCDF file in the default ``shone``
    directory.

    Parameters
    ----------
    temperature : array-like
        Temperature grid. Default is roughly log-spaced from 300 to 6000 K.
    pressure : array-like
        Pressure grid. Default is roughly log-spaced from 1e-8 to 10 bar.
    log_m_to_h : array-like
        Metallicity grid. Default is log-spaced from -1 to 3.
    log_c_to_o : array-like
        C/O grid. Default is log-spaced from -1 to 0.3.
    n_species : int
        Number of species in this FastChem computation.

    Returns
    -------
    ds : `~xarray.Dataset`
        Dataset containing the 5D fastchem grid over temperature,
        pressure, metallicity, C/O, and species.
    """
    if temperature is None:
        temperature = np.round(np.geomspace(300, 6000, 22), -1)
    if pressure is None:
        pressure = round_in_log(np.geomspace(1e-8, 10, 20))
    if log_m_to_h is None:
        log_m_to_h = np.linspace(-1, 3, 11)
    if log_c_to_o is None:
        log_c_to_o = np.linspace(-1, 0.3, 16)

    shape = (
        pressure.size, temperature.size,
        log_m_to_h.size, log_c_to_o.size,
        n_species
    )

    results_mmr = np.empty(shape, dtype=np.float32)
    results_vmr = np.empty(shape, dtype=np.float32)
    temperature2d, pressure2d = np.meshgrid(temperature, pressure)

    for i, log_mh in tqdm(
        enumerate(log_m_to_h),
        total=len(log_m_to_h),
        desc="Pre-computing FastChem grid"
    ):
        for j, log_co in enumerate(log_c_to_o):
            chem2d = FastchemWrapper(
                temperature2d.ravel(), pressure2d.ravel(),
                metallicity=10 ** log_mh,
                c_to_o_ratio=10 ** log_co
            )

            mmr_mmw = chem2d.mmr_mmw().reshape((*pressure2d.shape, n_species))
            vmr = chem2d.vmr().reshape((*pressure2d.shape, n_species))

            results_mmr[:, :, i, j, :] = mmr_mmw
            results_vmr[:, :, i, j, :] = vmr

    species_table = chem2d.get_species()

    coord_names = "pressure temperature log_m_to_h log_c_to_o species".split()

    ds = xr.Dataset(
        data_vars=dict(
            mmr_mmw=(coord_names, results_mmr),
            vmr=(coord_names, results_vmr)
        ),
        coords=dict(
            temperature=temperature,
            pressure=pressure,
            log_m_to_h=log_m_to_h,
            log_c_to_o=log_c_to_o,
            species=list(species_table['symbol']),
        ),
        attrs={str(idx): symbol for idx, symbol in species_table[['index', 'symbol']]}
    )

    ds.to_netcdf(os.path.join(shone_dir, fastchem_grid_filename))
    return ds


def get_fastchem_interpolator(path=None):
    """
    Return a jitted FastChem abundance interpolator.

    Returns
    -------
    interp : function
        A just-in-time compiled opacity interpolator.
    """
    if path is None:
        path = os.path.join(shone_dir, fastchem_grid_filename)

    if not os.path.exists(path):
        raise ValueError(
            f"Expected precomputed FastChem grid at {path}, "
            "but none was found. Run `build_fastchem_grid()` to "
            "create one."
        )

    with xr.open_dataset(path) as ds:
        grid = ds.vmr

    x_grid_points = (
        jnp.float32(grid.pressure.to_numpy()),
        jnp.float32(grid.temperature.to_numpy()),
        jnp.float32(grid.log_m_to_h.to_numpy()),
        jnp.float32(grid.log_c_to_o.to_numpy()),
    )

    @partial(jit, donate_argnames=('grid',))
    def interp(
        temperature, pressure, log_m_to_h, log_c_to_o,
        grid=grid.to_numpy().astype(np.float32)
    ):
        """
        Parameters
        ----------
        temperature : float or array
            Temperature value.
        pressure : float or array
            Pressure value.
        log_m_to_h : float
            [M/H] value.
        log_c_to_o : float
            [C/O] value.
        """
        interp_point = jnp.column_stack([
            pressure,
            temperature,
            jnp.broadcast_to(log_m_to_h, temperature.shape),
            jnp.broadcast_to(log_c_to_o, temperature.shape),
        ]).astype(jnp.float32)

        return nd_interp(
            interp_point,
            x_grid_points,
            grid,
            axis=0
        )

    return interp


def _fastchem_species_table(fastchem=None):
    """
    Return a table of the species included in FastChem.

    Returns
    -------
    table : `~astropy.table.Table`
        Table with columns: index, name, symbol, weight,
        and type (element or molecule).
    """
    if fastchem is None:
        fastchem = FastchemWrapper(
            np.array([2300]), np.array([1]), quiet=True
        ).fastchem

    element_symbols = []
    gas_species_symbols = []
    idx = 0
    while fastchem.getElementSymbol(idx) != '':
        element_symbols.append([
            idx,
            fastchem.getElementName(idx),
            fastchem.getElementSymbol(idx),
            fastchem.getElementWeight(idx),
            'element',
        ])
        idx += 1

    while fastchem.getGasSpeciesSymbol(idx) != '':
        gas_species_symbols.append([
            idx,
            fastchem.getGasSpeciesName(idx),
            fastchem.getGasSpeciesSymbol(idx),
            fastchem.getGasSpeciesWeight(idx),
            'gas species',
        ])
        idx += 1

    symbols = list(element_symbols)
    symbols.extend(gas_species_symbols)

    fastchem_species = Table(
        rows=symbols,
        names='index name symbol weight type'.split()
    )
    fastchem_species.add_index('symbol')

    return fastchem_species


def number_density(temperature, pressure):
    """
    Number density of all atmospheric species.

    Parameters
    ----------
    temperature : array
        Temperature [K].
    pressure : array
        Pressure [bar].

    Returns
    -------
    n : array
        Number density at each pressure and temperature.
    """
    return pressure * bar_to_dyn_cm2 / (k_B * temperature)


def mass_density(temperature, pressure, vmr):
    # [AMU / cm3]
    species_table = fastchem_species_table()
    n_total = number_density(temperature, pressure)
    rho = jnp.sum(
        vmr * n_total[:, None] *
        species_table['weight'],
        axis=1
    )
    return rho


def mean_molecular_weight(temperature, pressure, vmr):
    """
    Mean molecular weight [AMU] at each pressure and
    temperature for an ideal gas.

    Parameters
    ----------
    temperature : array
        Temperature [K].
    pressure : array
        Pressure [bar].
    vmr : array of shape `(M, N)`
        Volume mixing ratio for `M` pressure layers and `N` species.

    Returns
    -------
    mean_molecular_weight_amu : array
        Mean molecular weight [AMU].
    """
    rho = mass_density(temperature, pressure, vmr)
    mean_molecular_weight_amu = rho / (pressure * bar_to_dyn_cm2 / (k_B * temperature))
    return mean_molecular_weight_amu


def fastchem_species_table():
    """
    Return an astropy table with names, symbols, weights and
    indices for each species in a generic instance of FastChem.

    Returns
    -------
    species_table : `~astropy.table.Table`
        Table of FastChem species.
    """
    global cached_species_table
    if cached_species_table is None:
        cached_species_table = _fastchem_species_table()
    return cached_species_table
