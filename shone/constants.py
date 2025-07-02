"""
Astronomical constants from astropy in handy units.
"""
import astropy.units as u
from astropy.constants import (m_p as quantity_m_p, k_B as quantity_k_B, G as quantity_G, 
                               c as quantity_c, M_sun as quantity_M_sun,
                               R_sun as quantity_R_sun
                            )

__all__ = [
    'm_p',
    'k_B',
    'bar_to_dyn_cm2',
    'k_B_over_m_p',
]

# constants in cgs:
m_p = quantity_m_p.cgs.value
k_B = quantity_k_B.cgs.value
G = quantity_G.cgs.value
c = quantity_c.cgs.value
M_sun = quantity_M_sun.cgs.value
R_sun = quantity_R_sun.cgs.value
bar_to_dyn_cm2 = (1 * u.bar).cgs.value
k_B_over_m_p = k_B / m_p
d_in_seconds =  (1 * u.d).cgs.value
au_in_cm = (1 * u.au).cgs.value
rad_in_deg = (1 * u.rad).to('degree').value
