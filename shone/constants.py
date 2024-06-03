"""
Astronomical constants from astropy in handy units.
"""
import astropy.units as u
from astropy.constants import m_p as quantity_m_p, k_B as quantity_k_B

__all__ = [
    'm_p',
    'k_B',
    'bar_to_dyn_cm2',
    'k_B_over_m_p',
]

# constants in cgs:
m_p = quantity_m_p.cgs.value
k_B = quantity_k_B.cgs.value
bar_to_dyn_cm2 = (1 * u.bar).cgs.value
k_B_over_m_p = k_B / m_p
