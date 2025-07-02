def v_orb(P,M,e=0,theta=0,m=0):

    from jax import numpy as jnp
    from shone.constants import G,M_sun,d_in_seconds

    #The vis-viva equation reads:
    #v(r)^2 = G(M+m) (2/r - 1/a)

    #r in an elliptical orbit:
    #r(theta) = a(1-e^2) / (1+ e cos(theta))

    #We need Kepler's third law to convert M's to a's or P's:
    # a^3 / P^2 = G(M+m)/(4pi^2)

    GM = G * (M+m)*M_sun #in cgs

    a = (GM * (P*d_in_seconds)**2 / 4 / jnp.pi**2)**(1/3) # in cm

    r = a*(1-e**2) / (1+e*jnp.cos(theta)) # in cm

    v = (GM * ( 2/r - 1/a))**(1/2)  / 1e5 # in cm, convert to km/s

    return(v)


from jax import jit
def doppler_factor(v):
    """
    This calculates the relativistic doppler factor given a radial velocity 
    in km/s.


    Parameters
    ----------
    v : float, array-like
        Radial velocity in km/s.

    Returns
    -------
    f : float, array-like, same as v
        The doppler factor to be multiplying the wavelength with.

    """
    from jax import numpy as jnp
    from shone.constants import c
    beta = v * 1e5 / c # c is in cgs so convert v to km/s.
    f = jnp.sqrt((1 + beta)/(1 - beta))
    return(f)



@jit
def RV_eccentric(phase, M = 1.0, m = 0.0, P = 365.0, e = 0.0, omega = 0.0,
                i=90.0):
    """
    This function calculates the radial velocity in km/s for a planet in 
    an elliptical orbit using jaxoplanet. 
    
    Input is provided in terms of the orbital phases at which the radial
    velocity is required, and the system parameters, including the 
    stellar mass. Stellar mass is assumed to be known to better precision 
    than the semi-major axis a. If this is not the case, you need to 
    proceed by calculating M from a, using Kepler III.

    If the mass of the companion is non-negligible, then its mass can
    also be set.

    The coordinate system follows that of the exoplanet package:
    https://docs.exoplanet.codes/en/latest/tutorials/data-and-models/


    Parameters
    ----------
    phase : float, array-like
        Orbital phase, typically between 0 and 1.0. 0.0 is mid-transit.
        Equivalent to the Mean Anomaly divided by 2 pi.

    M : float
        Stellar mass in solar masses.

    m : float
        Planet mass in solar masses.

    P : float
        Orbital period in days.

    e : float
        eccentricity.

    omega : float
        argument of peri-apsis, following the exoplanet package coordinate system. 

    i : float
        Orbital inclination in degrees. 90 is transiting.

    Returns
    -------
    rv : float, array-like, same as phase
        The planet's radial velocity in km/s.

    """
    from jaxoplanet.orbits import keplerian
    from shone.constants import rad_in_deg, R_sun, d_in_seconds

    star = keplerian.Central(mass=M, radius=1.0)

    system = keplerian.System(star).add_body(mass = m, period = P, 
                        eccentricity = e, omega_peri = omega/rad_in_deg,
                        inclination = i/rad_in_deg,time_transit=0.0
                        )

    planet = system.bodies[0]

    vz = planet.velocity(phase * P)[2] # along the z axis in Rsol per day.

    # Convert to km/s, note that the positive z axis is the negative RV axis.
    rv = vz * R_sun / 1e5 / d_in_seconds * -1 
   
    return(rv)

@jit
def RV_circular(phase, M = 1.0, m = 0.0, P = 365.0, i=90.0):
    """
    This function calculates the radial velocity in km/s for a planet in 
    a circular orbit using standard analytical mechanics, in an attempt
    to be faster than jaxoplanet. 
    
    Input is provided in terms of the orbital phases at which the radial
    velocity is required, and the system parameters, including the 
    stellar mass. Stellar mass is assumed to be known to better precision 
    than the semi-major axis a. If this is not the case, you need to 
    proceed by calculating M from a, using Kepler III.

    If the mass of the companion is non-negligible, then its mass can
    also be set.



    Parameters
    ----------
    phase : float, array-like
        Orbital phase, typically between 0 and 1.0. 0.0 is mid-transit. 
        Equivalent to the Mean Anomaly divided by 2 pi.

    M : float
        Stellar mass in solar masses.

    m : float
        Planet mass in solar masses.

    P : float
        Orbital period in days.

    i : float
        Orbital inclination in degrees. 90 is transiting.

    Returns
    -------
    rv : float, array-like
        The planet's radial velocity in km/s.

    """
    from jax import numpy as jnp
    import numpy as np
    from shone.constants import G,M_sun,d_in_seconds,rad_in_deg

    GM = G * (M+m)*M_sun #in cgs
    a = (GM * (P*d_in_seconds)**2 / 4 / jnp.pi**2)**(1/3) # in cm

    v_orb = jnp.sqrt(GM / a) # in cgs

    rv = v_orb * jnp.sin(phase * 2 * np.pi) * jnp.sin(i/rad_in_deg) # in cgs
    return(rv / 1e5) # Convert to km/s.

 


