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
        # these parameters were found to produce similar transmission spectra
        # using both H&K 2017 and dw&S 2013:
        log_kappa_min, hk_kappa_factor = [-5.6054654, 1.8900791]
        delta_opacity = 10
        opacity = add_absorption_band(
            opacity=10 ** log_kappa_min + jnp.zeros((self.n_layers, 1)),
            delta_opacity=delta_opacity,
            wavelength=self.wavelength,
            band_min=2, band_max=3
        )

        weights_amu = jnp.array([30.0])

        Rp = de_wit_seager_2013.transmission_radius(
            self.wavelength, self.temperature, self.pressure,
            self.g, self.R_p0, opacity, vmr, vmr_indices,
            weights_amu, rayleigh_scattering=False
        )

        P_0 = 3.0
        T_0 = jnp.mean(self.temperature)
        mmw = jnp.mean(weights_amu)
        Rp_hk = heng_kitzmann_2017.transmission_radius_isothermal_isobaric(
            opacity[0] * hk_kappa_factor, self.R_p0, P_0, T_0, mmw, self.g
        )
        chi2 = jnp.sum((Rp_hk / self.R_p0 - Rp / self.R_p0) ** 2)

        # expect better than 100 ppm agreement
        assert chi2 < 100e-6

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
        vmr_indices = np.array([0])
        weights_amu = np.array([18.0])
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
        vmr_indices = np.array([0])
        weights_amu = np.array([18.0])
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
        vmr = np.ones((self.n_layers, 1)) * 1e-2
        vmr_indices = np.array([0])
        weights_amu = np.array([18.0])
        Rp = de_wit_seager_2013.transmission_radius(
            self.wavelength, self.temperature, self.pressure,
            self.g, self.R_p0, water_opacity, vmr, vmr_indices,
            weights_amu, rayleigh_scattering=False
        )

        # check that the highest-opacity CO2 band is near 4.29 µm:
        np.testing.assert_allclose(self.wavelength[np.argmax(Rp)], 4.294589178356713, rtol=1e-3)

        # check that the minimum radius between CO2 bands is near an expected
        # fraction of the fiducial radius
        np.testing.assert_allclose(np.min(Rp / self.R_p0), 0.8750502, rtol=1e-3)

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
        pressure = np.geomspace(1e-8, 1, self.n_layers)
        opacity = np.zeros((self.n_layers, 1))
        weights_amu = np.array([30])

        vmr = np.ones((self.n_layers, 1))
        vmr_indices = np.arange(vmr.shape[1])

        Rp = de_wit_seager_2013.transmission_radius(
            self.wavelength, self.temperature, pressure,
            self.g, self.R_p0, opacity, vmr, vmr_indices,
            weights_amu, rayleigh_scattering=False
        )

        np.testing.assert_allclose(Rp / self.R_p0, 1)
