import numpy as np
from jax import jit

# number densities at reference p-T
n_ref_H2 = 26867800  # [um^-3]
n_ref_He = 25468990  # [um^-3]

# King factor
K_lambda = 1

__all__ = [
    'rayleigh_cross_section_H2',
    'rayleigh_cross_section_He'
]


@jit
def refractive_index_H2(wavelength):
    """
    Real part of the refractive index of H2.

    From Sneep & Ubachs (2005).

    Parameters
    ----------
    wavelength : array
        Wavelength [µm].
    """
    # Malik 2017 Eqn 17
    return 13.58e-5 * (
        1 + 0.00752 * wavelength**-2
    ) + 1


@jit
def refractive_index_He(wavelength):
    """
    Real part of the refractive index of He.

    From Sneep & Ubachs (2005).

    Parameters
    ----------
    wavelength : array
        Wavelength [µm].
    """
    # Deitrick 2020 Eqn C3
    return 1e-8 * (2283 +
        (1.8102e13 / (1.5342e10 - wavelength**-2))
    ) + 1


@jit
def rayleigh_cross_section_H2(wavelength):
    """
    Rayleigh scattering cross-section of H2.

    From Sneep & Ubachs (2005).

    Parameters
    ----------
    wavelength : array
        Wavelength [µm].

    Returns
    -------
    cross_section: array
        Scattering cross-section [cm^-2].
    """
    # Malik 2017 Eqn 16
    return (
        24e-8 * np.pi**3 / n_ref_H2**2 / wavelength**4 *
        ((refractive_index_H2(wavelength)**2 - 1) /
         (refractive_index_H2(wavelength)**2 + 2))**2 * K_lambda
    )


@jit
def rayleigh_cross_section_He(wavelength):
    """
    Rayleigh scattering cross-section of He.

    From Sneep & Ubachs (2005).

    Parameters
    ----------
    wavelength : array
        Wavelength [µm].

    Returns
    -------
    cross_section: array
        Scattering cross-section [cm^-2].
    """
    # Malik 2017 Eqn 16
    return (
        24e-8 * np.pi**3 / n_ref_He**2 / wavelength**4 *
        ((refractive_index_He(wavelength)**2 - 1) /
         (refractive_index_He(wavelength)**2 + 2))**2 * K_lambda
    )
