import pytest
import numpy as np
import astropy.units as u
from astropy.constants import G

from jax import numpy as jnp

from shone.chemistry.fastchem import FastchemWrapper
from shone.opacity import Opacity
from shone.transmission import de_wit_seager_2013, heng_kitzmann_2017


def add_absorption_band(opacity, delta_opacity, wavelength, band_min, band_max):
    return opacity + jnp.where(
        (band_min < wavelength) & (wavelength < band_max),
        delta_opacity, 0
    )


class TestTransmission:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.n_layers = 10
        self.pressure = np.geomspace(1e-8, 5, self.n_layers)
        self.temperature = np.ones_like(self.pressure) * 3000
        self.wavelength = np.linspace(1, 5, 500)
        self.R_p0 = (1 * u.R_jup).cgs.value
        self.g = (G * u.M_jup / self.R_p0 ** 2).cgs.value

    def test_hk_vs_dws_isothermal(self):
        vmr = np.ones((self.n_layers, 1))
        vmr_indices = np.arange(vmr.shape[1])
        delta_opacity = 10
        opacity = add_absorption_band(
            opacity=10 ** -5 + jnp.zeros((self.n_layers, 1)),
            delta_opacity=delta_opacity,
            wavelength=self.wavelength,
            band_min=2, band_max=3
        )

        weights_amu = [30.0]

        Rp = de_wit_seager_2013.transmission_radius(
            self.wavelength, self.temperature, self.pressure,
            self.g, self.R_p0, opacity, vmr, vmr_indices,
            weights_amu, rayleigh_scattering=False
        )

        P_0 = 40
        T_0 = jnp.mean(self.temperature)
        mmw = weights_amu[0]
        Rp_hk = heng_kitzmann_2017.transmission_radius_isothermal_isobaric(
            opacity[0], self.R_p0, P_0, T_0, mmw, self.g
        )
        chi2 = jnp.sum((Rp_hk / self.R_p0 - Rp / self.R_p0) ** 2)

        # expect better than 10 ppm agreement
        assert chi2 < 10e-6

    @pytest.mark.parametrize(
        ("filt", "off_filter"),
        (
            ([1.5, 1.8],   # H band
             [1.8, 2.0]),  # between H and K bands
            ([2.0, 2.5],   # K band
             [2.5, 3.0])   # just beyond K band
        )
    )
    def test_h2o_atmospheric_windows(self, filt, off_filter):
        """
        check that a planet with a steam atmosphere produces atmospheric windows
        between the water bands using the "demo" opacities.
        """
        op = Opacity.load_demo_species("H2O")
        interp_opacity = op.get_binned_interpolator(
            self.wavelength, self.temperature, self.pressure
        )

        water_opacity = interp_opacity(self.temperature, self.pressure)
        vmr = np.ones((self.n_layers, 1))
        vmr_indices = [0]
        weights_amu = [18.0]
        Rp = de_wit_seager_2013.transmission_radius(
            self.wavelength, self.temperature, self.pressure,
            self.g, self.R_p0, water_opacity, vmr, vmr_indices,
            weights_amu, rayleigh_scattering=False
        )

        in_window = (filt[0] < self.wavelength) & (self.wavelength < filt[1])
        off_window = (off_filter[0] < self.wavelength) & (self.wavelength < off_filter[1])
        assert np.mean(Rp, where=in_window) < np.mean(Rp, where=off_window)

    @pytest.mark.parametrize(
        ("filt", "off_filter"),
        (
            ([3.5, 4.0],   # low opacity band
             [1.8, 2.0]),  # high opacity CO2 band
            ([2.3, 2.5],   # low opacity band
             [2.6, 2.9])   # high opacity band
        )
    )
    def test_co2_atmospheric_windows(self, filt, off_filter):
        """
        check that a planet a pure-CO2 atmosphere produces atmospheric windows
        between the CO2 bands using the "demo" opacities.
        """
        op = Opacity.load_demo_species("CO2")
        interp_opacity = op.get_binned_interpolator(
            self.wavelength, self.temperature, self.pressure
        )

        water_opacity = interp_opacity(self.temperature, self.pressure)
        vmr = np.ones((self.n_layers, 1))
        vmr_indices = [0]
        weights_amu = [18.0]
        Rp = de_wit_seager_2013.transmission_radius(
            self.wavelength, self.temperature, self.pressure,
            self.g, self.R_p0, water_opacity, vmr, vmr_indices,
            weights_amu, rayleigh_scattering=False
        )

        in_window = (filt[0] < self.wavelength) & (self.wavelength < filt[1])
        off_window = (off_filter[0] < self.wavelength) & (self.wavelength < off_filter[1])
        assert np.mean(Rp, where=in_window) < np.mean(Rp, where=off_window)

    def test_co2_sanity_check(self):
        op = Opacity.load_demo_species("CO2")
        interp_opacity = op.get_binned_interpolator(
            self.wavelength, self.temperature, self.pressure
        )

        water_opacity = interp_opacity(self.temperature, self.pressure)
        vmr = np.ones((self.n_layers, 1))
        vmr_indices = [0]
        weights_amu = [18.0]
        Rp = de_wit_seager_2013.transmission_radius(
            self.wavelength, self.temperature, self.pressure,
            self.g, self.R_p0, water_opacity, vmr, vmr_indices,
            weights_amu, rayleigh_scattering=False
        )

        # check that the highest-opacity CO2 band is near 4.29 µm:
        np.testing.assert_allclose(self.wavelength[np.argmax(Rp)], 4.29, rtol=0.02)

        # check that the maximum radius over the minimum radius
        # is close to an expected ratio:
        np.testing.assert_allclose(np.ptp(Rp / self.R_p0), 0.013846, rtol=1e-3)

    @pytest.mark.parametrize(
        ("extremum", "expected_value"),
        (
            ("argmin", 1.059118),
            ("argmax", 0.617234)
        )
    )
    def test_H2O_TiO(self, extremum, expected_value):
        """
        Construct an atmosphere in chemical equilibrium with water, titanium oxide,
        and rayleigh scattering. Check that the max/min radii occur at the
        expected wavelengths.
        """
        molecules = ['H2O', 'TiO']
        opacity = []
        wavelength = np.linspace(0.5, 5, 500)
        temperature = np.linspace(2000, 3000, self.pressure.size)

        for molecule in molecules:
            op = Opacity.load_demo_species(molecule)
            interp_op = op.get_binned_interpolator(wavelength, temperature, self.pressure)
            opacity.append(
                interp_op(temperature, self.pressure)
            )

        opacity = jnp.array(opacity)

        chem = FastchemWrapper(temperature, self.pressure)
        vmr = chem.vmr()
        vmr_indices = chem.get_column_index(species_name=molecules)
        weights_amu = chem.get_weights()

        Rp = de_wit_seager_2013.transmission_radius(
            wavelength, temperature, self.pressure,
            self.g, self.R_p0, jnp.array(opacity), vmr, vmr_indices,
            weights_amu
        )

        wl_extremum = wavelength[getattr(np, extremum)(Rp)]
        np.testing.assert_allclose(wl_extremum, expected_value, rtol=1e-3)

    def test_below_bottom_of_atmosphere(self):
        """
        We assume that the atmosphere is fully opaque at pressures
        higher than the maximum pressure in the grid. Check that
        an atmosphere with no opacity above P_0 = 1 bar produces Rp == R_p0.
        """
        n_layers = 100
        pressure = np.geomspace(1e-8, 1, n_layers)
        temperature = np.ones_like(pressure) * 3000

        opacity = np.zeros((n_layers, 1))
        weights_amu = [30]

        vmr = np.ones((n_layers, 1))
        vmr_indices = np.arange(vmr.shape[1])

        Rp = de_wit_seager_2013.transmission_radius(
            self.wavelength, temperature, pressure,
            self.g, self.R_p0, opacity, vmr, vmr_indices,
            weights_amu, rayleigh_scattering=False
        )

        np.testing.assert_allclose(Rp / self.R_p0, 1, rtol=1e-4)
