import numpy as np
from jax import numpy as jnp, jit


def solid_body_rotation_kernel(velocity, vsini, u1=0, u2=0):
    """
    Solid body rotation kernel from Gray (2005).

    Parameters
    ----------
    velocity : array
        Velocity axis of the convolution.
    vsini : float
        Rotational velocity of the target in the
        same units as ``velocity``.
    u1, u2 : float
        Quadratic limb-darkening parameters.

    Returns
    -------
    rotation_kernel : array
        Rotation kernel.
    """
    velocity_ratio = velocity / vsini

    rotation_kernel = (
        -2/3 * jnp.sqrt(1 - velocity_ratio**2) *
        (3 * u1 + 2 * u2 * velocity_ratio**2 + u2 - 3) +
        0.5 * np.pi * u1 * (1 - velocity_ratio**2)
    ) / (np.pi * (1 - u1 / 3 - u2 / 6))

    rotation_kernel = jnp.where(
        (velocity_ratio < 1) & (velocity_ratio > -1),
        rotation_kernel,
        0
    )
    return rotation_kernel


@jit
def gaussian_kernel(x, sigma):
    """
    Gaussian kernel.

    Parameters
    ----------
    x : array
        Dispersion axis.
    sigma : float
        Standard deviation of the kernel in the
        same units as ``x``.

    Returns
    -------
    kernel : array
        Gaussian kernel.
    """
    return (
        1 / jnp.sqrt(2 * np.pi * sigma**2) *
        jnp.exp(-0.5 * x**2 / sigma ** 2)
    )


@jit
def fft_convolve_gaussian(wavelength, flux, sigma):
    """
    Convolution with a Gaussian kernel via FFT.

    Parameters
    ----------
    wavelength : array
        Wavelength of observations.
    flux : array
        Flux at each wavelength.
    sigma : float
        Standard deviation of the kernel in the
        same units as ``wavelength``.

    Returns
    -------
    flux_conv : array
        ``flux`` convolved with a Gaussian kernel.
    """
    n = len(wavelength)
    f = jnp.fft.fft(flux)
    x = jnp.arange(n)
    x0 = x.mean()
    k = gaussian_kernel(x, x0=x0, sigma=sigma)
    f_k = jnp.fft.fft(k / jnp.sum(k))
    c_2 = jnp.fft.ifft(f * f_k.conjugate())
    return jnp.fft.ifftshift(c_2.real)


@jit
def fft_convolve_rotation(velocity, vsini, flux, u1=0, u2=0):
    """
    Convolution with a solid body rotation kernel via FFT.

    Parameters
    ----------
    velocity : array
        Velocity axis of the convolution.
    vsini : float
        Rotational velocity of the target in the
        same units as ``velocity``.
    flux : array
        Flux at each ``velocity``.
    u1, u2 : float
        Quadratic limb-darkening parameters.

    Returns
    -------
    flux_conv : array
        ``flux`` convolved with a rotation kernel.
    """
    f = jnp.fft.fft(flux)
    k = solid_body_rotation_kernel(velocity, vsini, u1, u2)
    f_k = jnp.fft.fft(k / jnp.sum(k))
    c_2 = jnp.fft.ifft(f * f_k.conjugate())
    return jnp.fft.ifftshift(c_2.real)
