from jax import numpy as jnp, jit, lax

__all__ = ['bin_spectrum']


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


@jit
def bin_spectrum(output_wavelength, input_wavelength, input_flux):
    """
    Bin a spectrum from one wavelength grid to another.

    This function is a JAX implementation of SpectRes [1]_
    from A. C. Carnall [2]_.

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

    References
    ----------
    .. [1] `SpectRes on GitHub <https://github.com/ACCarnall/SpectRes>`_.
    .. [2]  Carnall, A. C. 2017, `arXiv:1705.05165 <https://arxiv.org/abs/1705.05165>`_.
            doi:10.48550/arXiv.1705.05165
    """
    old_edges, old_widths = bin_centers_to_edges(input_wavelength)
    new_edges, new_widths = bin_centers_to_edges(output_wavelength)

    starts_stops = jnp.column_stack([new_edges[:-1], new_edges[1:]])
    indices = jnp.searchsorted(old_edges, starts_stops) - 1
    in_idx = jnp.arange(len(old_edges) - 1)

    _, binned_spectrum = lax.scan(
        lambda carry, x: body_fun(
            carry, old_widths, old_edges, new_edges, input_flux, in_idx, x
        ), 0, indices
    )

    return binned_spectrum


def body_fun(i, old_widths, old_edges, new_edges, in_flux, in_idx, indices):
    start_idx = indices[0]
    stop_idx = indices[1]

    start_factor = ((old_edges[start_idx + 1] - new_edges[i]) /
                    (old_edges[start_idx + 1] - old_edges[start_idx]))

    end_factor = ((new_edges[i + 1] - old_edges[stop_idx]) /
                  (old_edges[stop_idx + 1] - old_edges[stop_idx]))

    start_width = old_widths[start_idx] * start_factor
    stop_width = old_widths[stop_idx] * end_factor

    old_widths_weighted = jnp.where(
        (start_idx <= in_idx) & (in_idx <= stop_idx),
        old_widths,
        0
    )
    old_widths_weighted = jnp.where(
        in_idx == start_idx,
        start_width,
        old_widths_weighted
    )

    old_widths_weighted = jnp.where(
        in_idx == stop_idx,
        stop_width,
        old_widths_weighted
    )

    nonzero = old_widths_weighted > 0
    new_flux = jnp.sum(
        old_widths_weighted * in_flux,
        where=nonzero
    ) / jnp.sum(
        old_widths_weighted, where=nonzero
    )

    return i + 1, new_flux
