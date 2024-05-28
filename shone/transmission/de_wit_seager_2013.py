from functools import partial

from jax import config
config.update('jax_debug_nans', True)

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
scatter_indices = np.array([
    np.argwhere(species_table['symbol'] == symbol)[0, 0]
    for symbol in scatter_symbols
])
weights_amu = jnp.array(species_table['weight'])


@jit
def delta_z_i(temperature, pressure, g, mmw):
    """
    Change in height in an atmosphere from bottom to
    top of a pressure layer.

    Malik et al. (2017) Equation 18.

    Parameters
    ----------
    temperature : array
        Temperature [K].
    pressure : array
        Pressure at each layer [bar].
    g : array or float
        Surface gravity [cm/s2]
    mmw : array
        Mean molecular weight [AMU].

    Returns
    -------
    dz : array
        Change in height between adjacent pressure layers.
    """
    pressure_bottom = pressure[1:]  # pressures at bottoms of layers
    pressure_top = pressure[:-1]  # pressures at tops of layers

    return (
        temperature[1:] / (mmw[1:] * g) * k_B_over_m_p *
        jnp.log(pressure_bottom / pressure_top)
    )


@jit
def radius_at_layer(temperature, pressure, g, mmw, R_p0):
    """
    Radius from planet center to each layer, given hydrostatic equilibrium.

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
    radius : array
        Radius [cm] from the planet's center to each
        pressure layer.
    """
    dz = delta_z_i(
        temperature, pressure, g, mmw
    )[::-1]
    # add zeroth height:
    dz = jnp.concatenate([jnp.array([0]), dz])
    return R_p0 + jnp.cumsum(dz)


@jit
def transmission_chord_length(temperature, pressure, g, mmw, R_p0):
    """
    Distance from the entry point to the exit point of a photon
    transmitted through a planetary atmosphere.

    The result is an array of the same shape as `pressure`
    with transmission chord lengths for chords that reach
    minimum altitude at each `pressure`.

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
        Total distance [cm] traveled through an atmosphere
        for rays that transmit through to a minimum
        altitude of `pressure`.
    """
    R_0 = radius_at_layer(temperature, pressure, g, mmw, R_p0)
    dx = 2 * (R_0.max()**2 - R_0**2)**0.5

    return jnp.where(~jnp.isnan(dx), dx, 0)


@jit
def transmission_radius(
    wavelength, temperature, pressure,
    g, R_p0, opacity,
    vmr, vmr_indices
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
    vmr_indices : array

    Returns
    -------
    transmission_radius : array
        Transmission radius [cm] as a function of wavelength.

    References
    ----------
    .. [1] `de Wit, J. & Seager, S. 2013, Science, 342, 1473. doi:10.1126/science.1245450
           <https://ui.adsabs.harvard.edu/abs/2013Sci...342.1473D/abstract>`_
    """
    # compute number densities of all species and for scattering species:
    mmw = mean_molecular_weight(temperature, pressure, vmr)
    n_total = number_density(temperature, pressure)
    n_scatter = vmr[:, scatter_indices] * n_total[:, None]

    # compute scattering cross-sections:
    scatter_H2 = rayleigh_cross_section_H2(wavelength)
    scatter_He = rayleigh_cross_section_He(wavelength)
    weights_scatter = weights_amu[None, scatter_indices]
    sigma_scatter = jnp.vstack([scatter_H2, scatter_He])  # shape (N_scatterers, N_wavelengths)

    # compute the length of a transmission chord through the atmosphere
    # that reaches a minimum altitude at each layer in the pressure grid:
    dx = transmission_chord_length(temperature, pressure, g, mmw, R_p0)

    # compute the optical depth due to scattering:
    tau_scatter = (n_scatter @ sigma_scatter) * dx[:, None]

    # compute the optical_depth due to absorption
    absorption_coeff = (
        opacity *                                   # (N_species, N_pressures, N_wavelengths) [cm2/g]
        vmr[:, vmr_indices].T[..., None] *          # (N_species, N_pressures, 1) [None]
        n_total[None, :, None] *                    # (1, N_pressures, 1) [1/cm3]
        weights_amu[vmr_indices, None, None] * m_p  # (N_species, 1, 1) [g]
    ).sum(0)  # [1/cm]; shape: (N_pressures, N_wavelengths)

    tau_absorb = absorption_coeff * dx[:, None]  # (N_pressures, N_wavelengths)

    # total optical depth is from absorption and scattering:
    tau = tau_absorb + tau_scatter

    # the planet radius at each pressure layer:
    radius = radius_at_layer(temperature, pressure, g, mmw, R_p0)

    # Add two values to the radius vector: zero radius and the bottom of
    # the pressure grid and add corresponding large optical depths to the
    # tau array. This represents the truly opaque deep atmosphere/surface.
    r = jnp.concatenate([jnp.array([0, radius.min()]), radius])[:, None]
    tau_padded = jnp.vstack(
        [tau, jnp.ones((2, tau.shape[1])) * 1e30]
    )

    # The cross-sectional area of a planet as a function of
    # wavelength observed in transmission is given by
    # de Wit & Seager 2013 Equation 3:
    cross_sectional_area = trapezoid(
        2 * np.pi * r * (1 - jnp.exp(-tau_padded[::-1])), r, axis=0
    )

    obs_radius = (cross_sectional_area / np.pi) ** 0.5

    return obs_radius
