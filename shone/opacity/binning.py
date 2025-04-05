from jax import numpy as jnp, jit, lax
from scipy.special import roots_legendre

__all__ = ['bin_opacity']

roots_10, weights_10 = roots_legendre(n=10)


@jit
def bin_centers_to_edges(wavelength):
    """
    For a vector of bin centers, estimate the bin edges and
    bin widths.

    Parameters
    ----------
    wavelength : array-like
        Wavelengths at bin centers for a spectrum.

    Returns
    -------
    edges : array-like
        Wavelength bin edges.
    widths : array-like
        Wavelength bin widths.
    """
    edges_0 = jnp.array([wavelength[0] - (wavelength[1] - wavelength[0]) / 2])
    widths_m1 = jnp.array([wavelength[-1] - wavelength[-2]])
    edges_m1 = jnp.array([wavelength[-1] + (wavelength[-1] - wavelength[-2]) / 2])
    edges_middle = (wavelength[1:] + wavelength[:-1]) / 2
    edges = jnp.concatenate([edges_0, edges_middle, edges_m1])
    widths = jnp.concatenate([edges[1:-1] - edges[:-2], widths_m1])

    return edges, widths


n_interp = 50


@jit
def gauss_points(kappa_sliced):
    y = jnp.linspace(-1, 1, kappa_sliced.shape[0])
    kappa_y = jnp.cumsum(
        jnp.sort(
            kappa_sliced
        )
    )

    kappa_y_gauss_points = jnp.interp(roots_10, y, kappa_y)
    return kappa_y_gauss_points


@jit
def bin_opacity(output_wavelength, input_wavelength, input_flux):
    """
    Bin an opacity spectrum from one wavelength grid to another.

    Parameters
    ----------
    output_wavelength : array−like
        The target output grid of central wavelengths.
    input_wavelength : array-like
        The input grid of central wavelengths.
    input_flux : array-like
        The fluxes at each input wavelength.

    Returns
    -------
    binned_spectrum : array-like
        Spectrum binned to a different resolution.
    """
    old_edges, old_widths = bin_centers_to_edges(input_wavelength)
    new_edges, new_widths = bin_centers_to_edges(output_wavelength)

    starts_stops = jnp.column_stack([new_edges[:-1], new_edges[1:]])
    indices = jnp.searchsorted(old_edges, starts_stops) - 1

    kappa = jnp.array(input_flux)

    def slicer(start_idx, stop_idx):
        wl_interp_grid = jnp.linspace(
            input_wavelength[start_idx], input_wavelength[stop_idx], n_interp
        )
        return jnp.interp(wl_interp_grid, input_wavelength, kappa)

    def body_fun(i, inds):
        start_idx, stop_idx = inds

        kappa_sliced = slicer(start_idx, stop_idx) / n_interp

        kappa_y_gauss_points = gauss_points(kappa_sliced)
        new_opacity = weights_10 @ kappa_y_gauss_points

        return i + 1, new_opacity

    _, binned_spectrum = lax.scan(
        body_fun, 0, indices
    )

    return binned_spectrum
