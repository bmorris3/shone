from jax import numpy as jnp, jit
from shone.constants import bar_to_dyn_cm2, k_B_over_m_p

__all__ = ['transmission_radius_isothermal_isobaric']


@jit
def transmission_radius_isothermal_isobaric(kappa, R_0, P_0, T_0, mmw, g):
    """
    Compute the radius spectrum for planet observed in transmission
    with an isothermal atmosphere.

    Uses the simple approximation from Heng & Kitzmann (2017) [1]_.

    Parameters
    ----------
    kappa : array
        Opacity [cm^2/g] as a function of wavelength.
    R_0 : float
        Reference radius [cm].
    P_0 : float
        Reference pressure [bar].
    T_0 : float
        Reference temperature [K].
    mmw : float
        Mean molecular weight [AMU].
    g : float
        Surface gravity [cm / s^2], assumed to be uniform
        with height.

    Returns
    -------
    transmission_radius : array
        Transmission radius [cm] as a function of wavelength.

    References
    ----------
    .. [1] `Heng, K. & Kitzmann, D. 2017, Monthly Notices of the Royal
           Astronomical Society, 470, 2972. doi:10.1093/mnras/stx1453
           <https://ui.adsabs.harvard.edu/abs/2017MNRAS.470.2972H/abstract>`_.
    """
    gamma = 0.57721  # Euler-Mascheroni constant
    mmw_amu = jnp.clip(mmw, 1, 100)  # [amu]
    T_0 = jnp.clip(T_0, 100, 50_000)  # [K]
    g = jnp.clip(g, 10, 1e10)  # [cm / s^2]
    P_0 = P_0 * bar_to_dyn_cm2  # [dyn / cm^2]

    # store the ratio of the Boltzmann constant to the
    # mass of a proton, both in cgs. Helpful for float precision:
    H = T_0 / (mmw_amu * g) * k_B_over_m_p  # pressure scale height

    # Heng & Kitzmann (2017) Equation 12.
    # Only applies to isothermal atmospheres:
    radius = R_0 + H * (
        gamma +
        jnp.log(
            P_0 * kappa / g *
            jnp.sqrt(2 * jnp.pi * R_0 / H)
        )
    )
    return radius
