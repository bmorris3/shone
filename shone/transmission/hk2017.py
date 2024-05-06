from jax import numpy as jnp
from astropy.constants import m_p, k_B

__all__ = ['transmission_radius_isothermal']


# constants in cgs:
m_p = m_p.cgs.value
k_B = k_B.cgs.value


def transmission_radius_isothermal(kappa, R_0, P_0, T_0, mmw, g):
    """
    Compute the radius spectrum for planet observed in transmission
    with an isothermal atmosphere.

    Uses the simple approximation from Heng & Kitzmann (2017) [1]_.

    Parameters
    ----------
    kappa : array-like
        Opacity [cm^2/g] as a function of wavelength.
    R_0 : float
        Reference radius [cm].
    P_0 : float
        Reference pressure [dyn].
    T_0 : float
        Reference temperature [K].
    mmw : float
        Mean molecular weight [g].
    g : float
        Surface gravity [cm/s^2], assumed to be uniform
        with height.

    Returns
    -------
    transmission_radius : array-like
        Transmission radius [cm] as a function of wavelength.

    References
    ----------
    .. [1] `Heng & Kitzmann (2017)
            <https://ui.adsabs.harvard.edu/abs/2017MNRAS.470.2972H/abstract>`_.
    """
    gamma = 0.57721  # Euler-Mascheroni constant
    H = k_B * T_0 / (mmw * g)  # pressure scale height

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
