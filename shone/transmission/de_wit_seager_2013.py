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


@jit
def scale_height(temperature, g, mmw):
    return k_B_over_m_p * temperature / (mmw * g)


@jit
def layer_height(temperature, pressure, g, mmw, R_p0, P_0=1):
    H = scale_height(temperature, g, mmw)
    z = -H * jnp.log(pressure / P_0) + R_p0
    return z


@jit
def transmission_chord_length(temperature, pressure, g, mmw, R_p0, P_0=1):
    a = layer_height(temperature, pressure, g, mmw, R_p0, P_0)
    z = jnp.diff(a[::-1])
    z = jnp.append(z, z[-1])

    length = 2 * jnp.sqrt(z**2 + 2 * a * z)
    return length[::-1]


@partial(jit, static_argnames=("rayleigh_scattering", "absorption"))
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
    vmr_indices : array
        Indices of the columns of the ``vmr`` FastChem output matrix
        corresponding to each opacity in ``opacity``.
    weights_amu : array
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
    dx = transmission_chord_length(temperature, pressure, g, mmw, R_p0)

    # compute the optical depth due to scattering:
    tau_scatter = (n_scatter @ sigma_scatter) * dx[:, None]

    # compute the optical_depth due to absorption
    absorption_coeff = continuum_opacity + (
        opacity *                                   # (N_species, N_press, N_wavelength) [cm2/g]
        vmr[:, vmr_indices].T[..., None] *          # (N_species, N_pressures, 1) [unitless]
        n_total[None, :, None] *                    # (1, N_pressures, 1) [1/cm3]
        weights_amu[vmr_indices, None, None] * m_p  # (N_species, 1, 1) [g]
    ).sum(0)  # [1/cm]; shape: (N_pressures, N_wavelengths)

    tau_absorb = absorption_coeff * dx[:, None]  # (N_pressures, N_wavelengths)

    # total optical depth is from absorption and scattering:
    tau = (
        # multiplication here allows for toggling each component:
        int(absorption) * tau_absorb +
        int(rayleigh_scattering) * tau_scatter
    )

    # the planet radius at each pressure layer:
    radius = layer_height(temperature, pressure, g, mmw, R_p0)[::-1]

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

    obs_radius = jnp.clip(cross_sectional_area / np.pi, 0) ** 0.5

    return obs_radius
