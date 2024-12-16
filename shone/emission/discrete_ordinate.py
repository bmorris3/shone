from functools import partial

import numpy as np
from scipy.special import roots_legendre

from jax import numpy as jnp, jit
from jax.tree_util import register_pytree_node_class

from shone.constants import bar_to_dyn_cm2, k_B_over_m_p, k_B

__all__ = [
    'Atmosphere'
]


def gauss_quad_mu(degree=2):
    if degree == 1:
        # approximation for Elsasser 1942 for
        # atmospheric slabs of finite thickness,
        # see page 42
        weights = [1]
        mus = [3 / 5]
    else:
        # otherwise, compute Gauss-Legendre
        # roots the usual way:
        x, w = roots_legendre(degree)

        # todo: maybe eventually replace with:
        # https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.4598

        # scale from range (-1, 1) to (0, 1)
        mus = (x + 1) / 2

        # normalize to sum to unity:
        weights = w / np.sum(w)

    return (
        jnp.array(weights)[:, None],
        jnp.array(mus)[:, None]
    )


@register_pytree_node_class
class Atmosphere:
    def __init__(
        self,
        wavelength,
        pressure,
        temperature,
        g,
        mmw
    ):
        """
        Parameters
        ----------
        wavelength : array
            Wavelength [µm].
        pressure : array
            Pressure [bar].
        temperature : array
            Temperature [K].
        g : float
            Surface gravity [cm/s^2].
        mmw : float
            Mean molecular weight [AMU].
        """
        self.wavelength = jnp.array(wavelength)
        self.pressure = jnp.array(pressure)
        self.temperature = jnp.array(temperature)
        self.g = g
        self.sigma_scattering = jnp.zeros_like(self.wavelength)
        H = k_B_over_m_p * self.temperature / (mmw * self.g)  # scale height

        dz = (
            H[1:] * jnp.diff(bar_to_dyn_cm2 * self.pressure) /
            (bar_to_dyn_cm2 * self.pressure[1:])
        )  # delta altitude
        self.delta_z = jnp.concatenate([dz, jnp.array([dz[-1]])])
        self.number_density = bar_to_dyn_cm2 * self.pressure / (k_B * self.temperature)

    def tree_flatten(self):
        children = (
            self.wavelength,
            self.pressure,
            self.temperature,
            self.g,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @jit
    def blackbody(self, temperature):
        h = 6.62607015e-34  # J s
        c = 299792458.0  # m/s
        k_B = 1.380649e-23  # J/K

        return (
            2 * h * c ** 2 /
            jnp.power(self.wavelength, 5) /
            jnp.expm1(h * c / (self.wavelength * k_B * temperature))
        )

    @jit
    def B_i(self, i):
        return self.blackbody(self.temperature[i])

    @jit
    def epsilon(self, i):
        self.alpha_scattering = self.sigma_scattering * self.number_density[i]

        omega_0 = (
            self.alpha_scattering /
            (self.delta_z[i] * self.alpha(i) + self.alpha_scattering)
        )
        return 1 - omega_0

    @jit
    def I_up_1_carry(self, i, mu, I_up_2):
        # Lee 2024 Eqn 7
        tau_a = self.transmission_function(i, mu)
        epsilon = self.epsilon(i)

        I_up_1 = (
            I_up_2 * tau_a +
            epsilon / (mu * self.beta(i) + epsilon) *
            (self.B_i(i - 1) - self.B_i(i) * tau_a)
        )

        return I_up_1, self.delta_tau(i)

    @jit
    def I_down_2_carry(self, i, mu, I_down_1):
        # Lee 2024 Eqn 8
        tau_a = self.transmission_function(i, mu)
        epsilon = self.epsilon(i)
        return (
            I_down_1 * tau_a +
            epsilon / (mu * self.beta(i) - epsilon) *
            (self.B_i(i - 1) * tau_a - self.B_i(i))
        )

    @jit
    def delta_tau(self, i):
        return (
            self.delta_z[i] * self.alpha(i) -
            self.delta_z[i - 1] * self.alpha(i - 1)
        )

    @partial(jit, static_argnames=('alpha_interpolators',))
    def alpha(self, i, alpha_interpolators=None):
        raise NotImplementedError()

    @jit
    def transmission_function(self, i, mu):
        # Lee 2024 Eqn 6
        return jnp.exp(-self.epsilon(i) * self.delta_tau(i) / mu)

    @jit
    def beta(self, i):
        # use `clip` here to dodge nans:
        delta_tau = jnp.clip(self.delta_tau(i), a_min=0)
        log_diff = jnp.clip(
            jnp.log(self.B_i(i)) - jnp.log(self.B_i(i - 1)),
            a_min=0
        )
        return -log_diff / delta_tau
