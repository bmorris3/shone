import numpy as np
from jax import numpy as jnp, random

import astropy.units as u
from specutils import SpectralAxis
from specutils.manipulation import FluxConservingResampler

from shone.spectrum import bin_spectrum


def test_spectral_binning():
    """
    Compare our spectral binning method to the
    specutils implementation for consistency.
    """
    key = random.PRNGKey(0)
    input_wavelength = jnp.geomspace(1, 5, 5_000)
    input_flux = 1 + 0.01 * random.normal(key, shape=input_wavelength.shape)
    output_wavelength = jnp.geomspace(1.1, 4.9, 1_321)

    shone_binned = bin_spectrum(output_wavelength, input_wavelength, input_flux)

    input_spec_axis = SpectralAxis(input_wavelength * u.um)
    output_spec_axis = SpectralAxis(output_wavelength * u.um)

    fcr = FluxConservingResampler()
    specutils_binned = fcr._fluxc_resample(
        input_spec_axis, output_spec_axis, input_flux * u.ct, None
    )[0].value

    # check for 10 ppm agreement:
    np.testing.assert_allclose(
        shone_binned, specutils_binned, rtol=1e-5
    )
