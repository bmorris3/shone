import numpy as np
from jax import numpy as jnp, jit
from shone.constants import m_p, k_B, bar_to_dyn_cm2, k_B_over_m_p


def delta_z_i(temperature_i, pressure_bottom, pressure_top, g, mmw):
    """
    Change in height in an atmosphere from bottom to
    top of a pressure layer.

    Malik et al. (2017) Equation 18.

    Parameters
    ----------
    temperature_i : array
        Temperature [K].
    pressure_top : array
        Pressure at the upper layer [bar].
    pressure_bottom : array
        Pressure at the lower layer [bar].
    g : array or float
        Surface gravity [cm/s2]
    mmw : array or float
        Mean molecular weight [AMU].

    Returns
    -------
    dz : array
        Change in height between adjacent pressure layers.
    """
    return (
        temperature_i / (mmw * g) * k_B_over_m_p *
        jnp.log(pressure_bottom / pressure_top)
    )


def radius(temperature, pressure, g, mmw, R_p0):
    """
    Altitude of each layer given hydrostatic equilibrium.

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
        temperature, pressure[1:], pressure[:-1], g, mmw
    )
    # add zeroth height:
    dz = jnp.concatenate([jnp.array([0]), dz])
    return R_p0 + jnp.cumsum(dz)


def transmission_chord_length(temperature, pressure, g, mmw, R_p0):
    """
    Distance from the entry point to the exit point of a photon
    transmitted through a planetary atmosphere.

    The result is an array of the same shape as `pressure`
    with transmission chord lengths for chords that reach
    minimum altitude at each `pressure`.

    Returns
    -------
    dx : array
        Total distance [cm] traveled through an atmosphere
        for rays that transmit through to a minimum
        altitude of `pressure`.
    """
    R_0 = radius(temperature, pressure, g, mmw, R_p0)
    dx = 2 * (R_0.max()**2 - R_0**2)**0.5
    return dx



