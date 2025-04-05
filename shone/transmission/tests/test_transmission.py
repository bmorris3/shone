import numpy as np
import astropy.units as u

from jax import numpy as jnp

from shone.chemistry import FastchemWrapper
from shone.opacity import generate_synthetic_opacity
from shone.transmission import de_wit_seager_2013
from shone.transmission import heng_kitzmann_2017


def test_hk_vs_dws():
    opacity = generate_synthetic_opacity(filename=None)
    interp_opacity = opacity.get_interpolator()

    wavelength = jnp.geomspace(0.5, 5, 500)
    pressure = jnp.geomspace(1e-10, 1e4)
    temperature = 1000 * jnp.ones_like(pressure)

    P_0 = 10
    T_0 = temperature.mean()
    R_0 = (1 * u.R_jup).cgs.value
    mmw = 2.328
    g = 3000
    weights_amu = jnp.array([3])
    synth_vmr = 1e-10

    # this "lower cloud deck" or minimum opacity has the same
    # effect as the optically thick assumption that we assert
    # below the pressure grid in the de Wit & Seager implementation:
    kappa_cloud = 9e-9
    kappa = (
        synth_vmr * weights_amu / mmw *
        interp_opacity(wavelength, T_0, P_0) + kappa_cloud
    )

    R_p_isothermal = heng_kitzmann_2017.transmission_radius_isothermal(
        kappa, R_0, P_0, T_0, mmw, g
    )

    opacity_samples = interp_opacity(wavelength, temperature, pressure)
    chem = FastchemWrapper(temperature, pressure)

    vmr = chem.vmr()
    vmr = np.hstack([vmr[:, :-1], synth_vmr * np.ones((pressure.size, 1))])
    vmr_indices = [vmr.shape[1] - 1]

    R_p = de_wit_seager_2013.transmission_radius(
        wavelength, temperature, pressure, g, R_0,
        opacity_samples[None, ...], vmr, vmr_indices, weights_amu,
        rayleigh_scattering=False
    )

    # renormalize the isotheraml transmission spectrum for
    # measuring the differences between spectra without considering
    # baseline offsets:
    R_p_isothermal_renorm = R_p_isothermal * (R_p / R_p_isothermal).mean()

    # the median absolute deviation between the isothermal/isobaric
    # approximation from Heng & Kitzmann (2017) and the more general
    # formulation from de Wit & Seager (2013) should agree within
    # 100 ppm for these parameters:
    assert np.median(np.abs(R_p_isothermal_renorm - R_p) / R_0) < 0.005
