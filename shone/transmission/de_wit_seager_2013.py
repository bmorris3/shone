from functools import partial
import numpy as np
from jax import numpy as jnp, jit
from jax.scipy.integrate import trapezoid
from shone.constants import m_p, k_B_over_m_p
from shone.chemistry.fastchem import (
    fastchem_species_table, number_density, mean_molecular_weight
)
from shone.opacity.scattering import (
    rayleigh_cross_section_H2, rayleigh_cross_section_He
)

__all__ = [
    'transmission_radius'
]

# cache these indices on load:
species_table = fastchem_species_table()

scatter_symbols = ['H2', 'He']
scatter_indices = jnp.array([
    np.argwhere(species_table['symbol'] == symbol)[0, 0]
    for symbol in scatter_symbols
])
weights_amu = jnp.array(species_table['weight'])
base_pressure = 1e-4  # [bar]


def _grad_nan_safe_func(func, x):
    safe_x = jnp.where(x != 0, x, 1)
    return jnp.where(x != 0, func(safe_x), 1)


@jit
def scale_height(temperature, g, mmw):
    """
    Pressure scale height.
    """
    return (
        k_B_over_m_p * temperature / mmw *
        _grad_nan_safe_func(lambda g: 1 / g, g)
    )


@jit
def layer_height(temperature, pressure, g, mmw, R_p0):
    """
    Height or altitude in the atmosphere [cm], where the
    minimum pressure corresponds to ``R_p0``.

    Parameters
    ----------
    temperature : array
        Temperature [K].
    pressure : array
        Pressure [bar].
    g : array or float
        Surface gravity [cm/s2].
    mmw : array or float
        Mean molecular weight [AMU].
    R_p0 : float
        Fiducial planet radius [cm].

    Returns
    -------
    dx : array
        Height in the atmosphere [cm]. For a decreasing ``pressure``
        vector, ``dx`` will also be decreasing.
    """
    H = scale_height(temperature[1:], g, mmw[1:])
    z = H * jnp.log(pressure[1:] / pressure[:-1])
    return jnp.concatenate([z, jnp.array([z[-1]])])


@partial(jit, static_argnames=("rayleigh_scattering", "absorption", "vmr_indices", "weights_amu"))
def _transmission_radius(
    wavelength, temperature, pressure,
    g, R_p0, opacity,
    vmr, vmr_indices,
    weights_amu,
    continuum_opacity=0,
    rayleigh_scattering=True,
    absorption=True
):
    """
    Compute the radius spectrum for planet observed in transmission.

    Uses the general formulation for computing transmission spectra in
    de Wit & Seager (2013) [1]_. This function assumes that only H2 and He
    contribute to Rayleigh scattering.

    Parameters
    ----------
    wavelength : array
        Wavelength [µm].
    temperature : array
        Temperature [K].
    pressure : array
        Pressure [bar].
    g : array or float
        Surface gravity [cm/s2].
    R_p0 : float
        Fiducial planet radius [cm].
    opacity : array
        Opacities [cm2/g] of shape (N_species, N_pressure, N_wavelength).
    vmr : array
        Volume mixing ratios for each pressure and each species
    vmr_indices : tuple
        Indices of the columns of the ``vmr`` FastChem output matrix
        corresponding to each opacity in ``opacity``.
    weights_amu : tuple
        Weights of each species in ``vmr`` [AMU].
    rayleigh_scattering : bool
        Include the contribution to the optical depth from
        Rayleigh scattering. Default is True.
    absorption : bool
        Include the contribution to the optical depth from
        absorption by atoms, ions, and molecules. Default is True.

    Returns
    -------
    transmission_radius : array
        Transmission radius [cm] as a function of wavelength.

    References
    ----------
    .. [1] `de Wit, J. & Seager, S. 2013, Science, 342, 1473. doi:10.1126/science.1245450
           <https://ui.adsabs.harvard.edu/abs/2013Sci...342.1473D/abstract>`_
    """
    # validate inputs to avoid nonsense:
    assert isinstance(absorption, bool), (
        f"`{absorption=}` argument must be "
        f"boolean; got type {type(absorption)}."
    )
    assert isinstance(rayleigh_scattering, bool), (
        f"`{rayleigh_scattering=}` argument must be "
        f"boolean; got type {type(rayleigh_scattering)}."
    )
    # compute number densities of all species and for scattering species:
    mmw = mean_molecular_weight(temperature, pressure, vmr, weights_amu)
    n_total = number_density(temperature, pressure)
    n_scatter = vmr[:, scatter_indices] * n_total[:, None]

    # compute scattering cross-sections:
    scatter_H2 = rayleigh_cross_section_H2(wavelength)
    scatter_He = rayleigh_cross_section_He(wavelength)
    sigma_scatter = jnp.vstack([scatter_H2, scatter_He])  # shape (N_scatterers, N_wavelengths)

    # compute the length of a transmission chord through the atmosphere
    # that reaches a minimum altitude at each layer in the pressure grid:
    # dx = transmission_chord_length(temperature, pressure, g, mmw, R_p0)
    # the planet radius at each pressure layer, from top to bottom:
    z = layer_height(temperature, pressure, g, mmw, R_p0)[::-1]
    r = jnp.cumsum(z) + R_p0  # radius [cm] bottom to top of atmos

    inner = (r + z)[:, None] ** 2 - r[None, :] ** 2
    safe_inner = jnp.where(inner > 0, inner, 1)
    x = jnp.where(inner > 0, jnp.sqrt(safe_inner), 0)

    # take elementwise difference along columns to compute `dx`.
    # ensure that this is a lower triangular.
    # flip result's order along the pressure dimension from high-to-low (increasing `r`)
    # to low-to-high (like `pressure`).
    dx = jnp.tril(-jnp.diff(x, axis=1))[::-1]

    # compute the optical depth due to scattering from the absorption coefficient
    # for the scattering species:
    alpha_scatter = n_scatter @ sigma_scatter
    tau_scatter = (alpha_scatter.T @ dx).T  # (N_pressures, N_wavelengths) [unitless]

    # compute the absorption coefficient by taking the
    # sum of the opacity per species weighted by the volume
    # mixing ratio, number density, and molecular weight:
    molecular_weight = jnp.array(weights_amu)[list(vmr_indices), None, None]
    alpha_absorb = continuum_opacity + (
        opacity *                                 # (N_species, N_press, N_wavelength) [cm2/g]
        vmr[:, list(vmr_indices)].T[..., None] *  # (N_species, N_pressures, 1) [unitless]
        n_total[None, :, None] *                  # (1, N_pressures, 1) [1/cm3]
        molecular_weight * m_p                    # (N_species, 1, 1) [g]
    ).sum(0)                                      # (N_pressures, N_wavelengths) [1/cm]

    # the optical depth contributed by absorption in each pressure layer is:
    tau_absorb = (alpha_absorb.T @ dx).T  # (N_pressures, N_wavelengths) [unitless]

    # total optical depth is from absorption and scattering:
    tau = (
        # multiplication here allows for toggling each component:
        int(absorption) * tau_absorb +
        int(rayleigh_scattering) * tau_scatter
    )    # (N_pressures, N_wavelengths) [unitless]

    # The cross-sectional area of a planet as a function of
    # wavelength observed in transmission is given by
    # de Wit & Seager 2013 Equation 3. We integrate from the bottom
    # to the top of the atmosphere, and assume the atmosphere
    # is fully opaque below max(pressure).
    cross_sectional_area = (
        trapezoid(
            2 * np.pi * r[1:, None] * (1 - jnp.exp(-tau)),
            r[1:, None], axis=0
        ) + np.pi * r[0] ** 2
    )
    obs_radius = jnp.clip(cross_sectional_area / np.pi, 0) ** 0.5

    return obs_radius


def transmission_radius(
    wavelength, temperature, pressure,
    g, R_p0, opacity,
    vmr, vmr_indices,
    weights_amu,
    continuum_opacity=0,
    rayleigh_scattering=True,
    absorption=True
):
    """
    Compute the radius spectrum for planet observed in transmission.

    Uses the general formulation for computing transmission spectra in
    de Wit & Seager (2013) [1]_. This function assumes that only H2 and He
    contribute to Rayleigh scattering.

    Parameters
    ----------
    wavelength : array
        Wavelength [µm].
    temperature : array
        Temperature [K].
    pressure : array
        Pressure [bar].
    g : array or float
        Surface gravity [cm/s2].
    R_p0 : float
        Fiducial planet radius [cm].
    opacity : array
        Opacities [cm2/g] of shape (N_species, N_pressure, N_wavelength).
    vmr : array
        Volume mixing ratios for each pressure and each species
    vmr_indices : tuple
        Indices of the columns of the ``vmr`` FastChem output matrix
        corresponding to each opacity in ``opacity``.
    weights_amu : tuple
        Weights of each species in ``vmr`` [AMU].
    rayleigh_scattering : bool
        Include the contribution to the optical depth from
        Rayleigh scattering. Default is True.
    absorption : bool
        Include the contribution to the optical depth from
        absorption by atoms, ions, and molecules. Default is True.

    Returns
    -------
    transmission_radius : array
        Transmission radius [cm] as a function of wavelength.

    References
    ----------
    .. [1] `de Wit, J. & Seager, S. 2013, Science, 342, 1473. doi:10.1126/science.1245450
           <https://ui.adsabs.harvard.edu/abs/2013Sci...342.1473D/abstract>`_
    """

    return _transmission_radius(
        wavelength, temperature, pressure,
        g, R_p0, opacity,
        vmr, tuple(list(vmr_indices)),
        tuple(list(weights_amu)),
        continuum_opacity=continuum_opacity,
        rayleigh_scattering=rayleigh_scattering,
        absorption=absorption
    )
