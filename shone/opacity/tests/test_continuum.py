from shone.constants import k_B
from shone.opacity.continuum import h_minus_continuum
import jax.numpy as jnp


def test_h_minus_continuum(mihalas_h_minus_continuum):
    # number densities and volume mixing ratios were computed
    # with fastchem using:
    #    temperature = 6300 K
    #    pressure = 1 bar
    number_density_e = 188856770172237.6
    number_density_H = 1.0586659319005289e+18
    vmr_H = 0.920837017933896
    vmr_h1minus = 1.445164249467722e-07
    temperature = 6300  # [K]

    electron_pressure = (
        number_density_e * k_B * temperature
    )  # [dyn/cm2]

    # these values come from digitizing Figure 7.6 on Page 207 in
    # the book by Hubený & Mihalas (2015). Wavelength array has units
    # of 1e-3 Angstrom, alpha has units of cm4/dyn per
    # electron pressure per H atom.
    mihalas_wl_1000AA, mihalas_alpha = mihalas_h_minus_continuum
    mihalas_wl_um = mihalas_wl_1000AA * 0.1  # [µm]
    mihalas_absorption_coeff = mihalas_alpha * electron_pressure * number_density_H  # [cm-1]

    shone_absorption_coeff = h_minus_continuum(
        mihalas_wl_um, temperature,
        number_density_e, number_density_H,
        vmr_H, vmr_h1minus
    ).squeeze()  # [cm-1]

    # median absolute deviation should be <10%:
    assert jnp.median(jnp.abs(shone_absorption_coeff / mihalas_absorption_coeff - 1)) < 0.10
