from jax import jit, lax, numpy as jnp
from shone.constants import k_B

__all__ = [
    'h_minus_continuum',
]


@jit
def h_minus_continuum(
        wavelength, temperature, number_density_e, number_density_h, vmr_H, vmr_h1minus
):
    """
    Continuum opacity from H-.

    From John (1988) [1]_.

    Parameters
    ----------

    wavelength : array
        Wavelength [µm].
    temperature : array
        Temperature [K].
    number_density_e: array
        Number density of electrons [cm^-3].
    number_density_h: array
        Number density of hydrogen [cm^-3].

    Returns
    -------
    alpha : array
        Absorption coefficient [cm^-1]

    References
    ----------
    .. [1] `John, T. L. 1988, Astronomy and Astrophysics, 193, 189
           <https://ui.adsabs.harvard.edu/abs/1988A%26A...193..189J/abstract>`_
    """
    temperature = jnp.atleast_1d(temperature)

    electron_pressure = (
        number_density_e * k_B * temperature
    )  # [dyn/cm2]

    log_cross_section = jnp.log10(
        electron_pressure *  # [dyn/cm2]
        (
            free_free_absorption(wavelength, temperature) +  # [cm4/dyn]
            bound_free_absorption(wavelength) * vmr_h1minus  # [cm4/dyn]
        )
    )  # [cm2]

    log_absorption_coeff = (
        26 + log_cross_section +
        jnp.log10(number_density_h)
    )

    return 10 ** log_absorption_coeff.T  # [1/cm]


@jit
def free_free_absorption(wavelength, temperature):
    """
    Free-free absorption of H-.

    From John (1988).

    Parameters
    ----------
    wavelength : array
        Wavelength [µm].
    temperature : array
        Temperature [K].

    Returns
    -------
    alpha : array
        Absorption cross section per electron pressure per H atom [cm4/dyn]
    """
    A_n1 = jnp.array([0.0, 2483.3460, -3449.8890, 2200.0400, -696.2710, 88.2830])
    B_n1 = jnp.array([0.0, 285.8270, -1158.3820, 2427.7190, -1841.4000, 444.5170])
    C_n1 = jnp.array([0.0, -2054.2910, 8746.5230, -13651.1050, 8624.9700, -1863.8650])
    D_n1 = jnp.array([0.0, 2827.7760, -11485.6320, 16755.5240, -10051.5300, 2095.2880])
    E_n1 = jnp.array([0.0, -1341.5370, 5303.6090, -7510.4940, 4400.0670, -901.7880])
    F_n1 = jnp.array([0.0, 208.9520, -812.9390, 1132.7380, -655.0200, 132.9850])

    # for wavelengths between 0.1823 micron and 0.3645 micron
    A_n2 = jnp.array([518.1021, 473.2636, -482.2089, 115.5291, 0.0, 0.0])
    B_n2 = jnp.array([-734.8666, 1443.4137, -737.1616, 169.6374, 0.0, 0.0])
    C_n2 = jnp.array([1021.1775, -1977.3395, 1096.8827, -245.6490, 0.0, 0.0])
    D_n2 = jnp.array([-479.0721, 922.3575, -521.1341, 114.2430, 0.0, 0.0])
    E_n2 = jnp.array([93.1373, -178.9275, 101.7963, -21.9972, 0.0, 0.0])
    F_n2 = jnp.array([-6.4285, 12.3600, -7.0571, 1.5097, 0.0, 0.0])

    @jit
    def ff(A_n, B_n, C_n, D_n, E_n, F_n, temperature=temperature, wavelength=wavelength):
        temperature = temperature[:, None, None]
        indices = jnp.arange(1, 7)[None, :, None]
        wavelength = wavelength[None, None, :]

        x = (
            jnp.power(5040.0 / temperature, (indices + 1) / 2.0) *
            (
                A_n[None, :, None] * wavelength ** 2 +
                B_n[None, :, None] +
                C_n[None, :, None] / wavelength +
                D_n[None, :, None] / wavelength ** 2 +
                E_n[None, :, None] / wavelength ** 3 +
                F_n[None, :, None] / wavelength ** 4
            )
        ).sum(1)

        return x

    kappa_ff = jnp.where(
        wavelength[None, None, :] > 0.3645,
        ff(A_n1, B_n1, C_n1, D_n1, E_n1, F_n1),
        0
    ) + jnp.where(
        (wavelength[None, None, :] >= 0.1823) &
        (wavelength[None, None, :] <= 0.3645),
        ff(A_n2, B_n2, C_n2, D_n2, E_n2, F_n2),
        0,
    )
    sigma = 1e-29 * kappa_ff
    return sigma.squeeze()


@jit
def bound_free_absorption(wavelength):
    """
    Bound free absorption of H-.

    Parameters
    ----------
    wavelength : array
        Wavelength [µm].

    Returns
    -------
    alpha : array
        Absorption cross section per electron pressure per H- ion [cm2]
    """
    lambda_0 = 1.6419  # photo-detachment threshold

    @jit
    def f():
        C_n = jnp.vstack(
            [jnp.arange(1, 7),
             jnp.array([152.519, 49.534, -118.858, 92.536, -34.194, 4.982])]
        ).T

        def body_fun(val, x):
            i, C_n_i = x
            return val, C_n_i * jnp.power(
                jnp.clip(1.0 / wavelength - 1.0 / lambda_0, 0),
                (i - 1) / 2.0,
            )

        return lax.scan(
            body_fun, jnp.zeros_like(wavelength), C_n
        )[-1].sum(0)

    # photo-detachment cross-section:
    sigma_lambda = (
        wavelength ** 3 *
        jnp.power(jnp.clip(1.0 / wavelength - 1.0 / lambda_0, 0), 1.5)
    )  # [cm2]
    return 1e-18 * sigma_lambda  # [cm4/dyn]
