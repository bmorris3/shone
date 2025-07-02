def v_orb(P,M,e=0,theta=0,m=0):
    """
    This function calculates the orbital velocity in km/s for a planet in 
    an elliptical orbit using the vis viva quation. 
    
    Input is provided in terms of the orbital period and the stellar mass. 
    Stellar mass is assumed to be known to better precision than semi-major
    axis a. If this is not the case, you need to calculate M from a using 
    Kepler III before proceeding with this function.

    If the mass of the companion is non-negligible, then its mass can
    also be set.


    Parameters
    ----------
    P : float
        Orbital period in days.

    M : float
        Stellar mass in solar masses.

    e : float
        eccentricity.

    theta : float
        true anomaly (radians)

    m : float
        mass of the secondary in solar masses.
    Returns
    -------
    v_orb : float
        The planet's orbital velocity in km/s.

    """
    import numpy as np
    from jax import numpy as jnp
    import astropy.units as u
    import astropy.constants as const

    #The vis-viva equation reads:
    #v(r)^2 = G(M+m) (2/r - 1/a)

    #r in an elliptical orbit:
    #r(theta) = a(1-e^2) / (1+ e cos(theta))

    #We need Kepler's third law to convert M's to a's or P's:
    # a^3 / P^2 = G(M+m)/(4pi^2)

    GM = const.G * (M+m)*u.M_sun

    a = (GM * (P*u.d)**2 / 4 / jnp.pi**2)**(1/3)

    r = a*(1-e**2) / (1+e*jnp.cos(theta))

    v = (GM * ( 2/r - 1/a))**(1/2)

    return(v.to('km/s').value)