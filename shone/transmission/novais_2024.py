from jax import numpy as jnp, jit
from shone.constants import m_p, k_B, bar_to_dyn_cm2, k_B_over_m_p
import sys
import numpy as np
jnp.set_printoptions(threshold=sys.maxsize)

__all__ = ['transmission_radius_nonisobaric']


#@jit
def transmission_radius_nonisobaric(integral_opacities_grid, P_cloudtop, R0, P0, T0, m_H20, X_H2O, g, Pmin, n_levels, wavelength):
    """
    Compute the radius spectrum for planet observed in transmission
    with a non-isobaric atmosphere.

    Uses the approximation from Novais et al. (2024) [1]_.

    Parameters
    ----------
    integral_opacities_grid : matrix
        Optical depth as a function of temperature (axis = 0),
        pressure (axis = 1), and wavelength (axis = 2).
    P_cloudtop : array
        Cloud-top pressure [bar] as a function of wavelength.
    R_0 : float
        Reference radius [cm].
    P_min : float
        Mininum pressure [bar].
    P_0 : float
        Reference pressure [bar].
    T_0 : float
        Reference temperature [K].
    m_H2O : float
        Water mass [g / mol].
    X_H2O : float
        Water abundance fraction.
    g : float
        Surface gravity [cm / s^2], assumed to be uniform
        with height.
    num_levels : float
        Number of atmospheric layers + 1.

    Returns
    -------
    transmission_radius : array
        Transmission radius [cm] as a function of wavelength.

    References
    ----------
    .. [1] `Heng, K. & Kitzmann, D. 2017, Monthly Notices of the Royal
           Astronomical Society, 470, 2972. doi:10.1093/mnras/stx1453
           <https://ui.adsabs.harvard.edu/abs/2017MNRAS.470.2972H/abstract>`_. ##### to be modified
    """
    P0_cgs = P0 * bar_to_dyn_cm2  # [dyn / cm^2]

    # Define range of pressure for non-isobaric integration
    pressure_levels_cgs = 10 ** jnp.linspace(jnp.log10(Pmin), jnp.log10(P0), n_levels) * bar_to_dyn_cm2

    # Define cloud opacity from cloud-top pressure
    P_cloudtop_cgs = P_cloudtop * bar_to_dyn_cm2
    cloud_tau = jnp.zeros((len(pressure_levels_cgs), len(wavelength)))
    cloud_tau = cloud_tau.at[pressure_levels_cgs > P_cloudtop_cgs,:].set(jnp.inf)

    # Define mass
    solar_H2, solar_He = 0.5, 0.085114
    background = 1 - X_H2O
    X_H2 = background * solar_H2 / (solar_H2 + solar_He)  # H2 abundance
    X_He = background * solar_He / (solar_H2 + solar_He)  # He abundance

    m = (X_H2 * 2.0 * m_p) + (X_He * 4.0 * m_p) + (X_H2O * m_H20 * m_p)

    scale_height = k_B * T0 / m / g  # pressure scale height

    # Interpolate optical depth in given temperature and find opacity
    integral_opacities = integral_opacities_grid(T0) * X_H2O
    tau_values_int = integral_opacities  # (len(pressure_levels), len(wavelength))

    factor = 2 * jnp.sqrt(2 * scale_height * R0) / k_B / T0  # array of len(pressure_levels)
    tau_val = (tau_values_int.T * factor).T + cloud_tau # array of (len(pressure_levels), len(wavelength))
    h_integrand1 = (R0 + scale_height * jnp.log(P0_cgs / pressure_levels_cgs)) / pressure_levels_cgs  # array of len(pressure_levels)
    h_integrand2 = 1 - jnp.exp(-tau_val)  # array of (len(pressure_levels), len(wavelength))
    h_integrand = (h_integrand2.T * h_integrand1).T  # array of (len(pressure_levels), len(wavelength))

    # Integrate over pressure
    h_integral_values = jnp.trapezoid(h_integrand, pressure_levels_cgs, axis=0)  # array of len(wavelength)

    # Novais et al. (2024, in prep) Equation A8.
    # Only applies to non-isobaric atmospheres:
    h_values = (scale_height / R0) * h_integral_values

    radius = R0 + h_values

    return radius