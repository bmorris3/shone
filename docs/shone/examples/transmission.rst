.. _transmission_spectrum:

********************
Transmission spectra
********************

.. contents:: Contents
   :depth: 2
   :local:
   :backlinks: none


.. _transmission_heng_kitzmann:

Isothermal/isobaric transmission spectra
========================================

Let's compute the transmission spectrum for an Earth-like planet with
a single-species atmosphere using the isothermal and isobaric
approximations from `Heng & Kitzmann (2017)
<https://ui.adsabs.harvard.edu/abs/2017MNRAS.470.2972H/abstract>`_.
The full transmission model is demonstrated below in :ref:`transmission_de_wit`.

We'll load an opacity grid and interpolate for the opacity at several temperatures,
add a gray cloud opacity, and compute a transmission spectrum.

Load opacity
------------

.. note::

    This example uses a synthetic opacity file that is totally made up.
    To download real opacity grids, see :ref:`opacities`.


Let’s synthesize a transmission spectrum for an Earth-sized planet with
one atmospheric species in the near-infrared.

.. code-block:: python

    import numpy as np
    from jax import numpy as jnp, jit
    import matplotlib.pyplot as plt
    
    import astropy.units as u
    from astropy.constants import m_p
    
    from shone.opacity import Opacity, generate_synthetic_opacity
    from shone.transmission import heng_kitzmann_2017

For each species to include in the atmosphere, you need to download an
opacity grid for that species. We load and interpolate opacity grids using
the `~shone.opacity.Opacity` class. For this example, we’ll use a synthetic
opacity grid, generated with by function:

.. code-block:: python

    generate_synthetic_opacity()

We can check which species are already chached and available on your
machine using `~shone.opacity.Opacity.get_available_species()`:

.. code-block:: python

    Opacity.get_available_species()


.. raw:: html

    <br /><div><i>Table length=9</i>
    <table id="table11585383056" class="table-striped table-bordered table-condensed">
    <thead><tr><th>name</th><th>species</th><th>charge</th><th>line_list</th><th>path</th><th>index</th></tr></thead>
    <tr><td>synthetic</td><td>synthetic</td><td>--</td><td>example</td><td>/Users/bmmorris/.shone/synthetic__example.nc</td><td>0</td></tr>
    </table></div><br /><br />


Let’s load the opacity file named "synthetic" that we created above:

.. code-block:: python

    # load the synthetic opacity file:
    opacity = Opacity.load_species_from_name('synthetic')


Interpolating opacities
-----------------------

Now we will create a just-in-time compiled opacity interpolator.
`~shone.opacity.Opacity.get_interpolator` returns a *function* that takes three
arguments – a wavelength array [µm], a temperature [K], and a pressure
[bar] – and returns an array of opacities for each wavelength:

.. code-block:: python

    # get a jitted 3D interpolator over wavelength, temperature, pressure:
    interp_opacity = opacity.get_interpolator()


Let's compute the opacity at one temperature and pressure:

.. code-block:: python

    wavelength = np.linspace(1, 5, 500)  # [µm]
    pressure = 1  # [bar]
    temperature = 200  # [K]

    example_opacity = interp_opacity(wavelength, temperature, pressure)
    
    plt.semilogy(wavelength, example_opacity, label=f"T={temperature} K")
    plt.legend()
    plt.gca().set(
        xlabel='Wavelength [µm]',
        ylabel='Opacity, $\kappa$ [cm$^2$ g$^{-1}$]'
    )

.. plot::

    import matplotlib.pyplot as plt
    from shone.opacity import Opacity, generate_synthetic_opacity

    generate_synthetic_opacity()

    # load the one opacity file:
    opacity = Opacity.load_species_from_name('synthetic')

    # get a jitted 3D interpolator over wavelength, temperature, pressure:
    interp_opacity = opacity.get_interpolator()

    wavelength = np.linspace(1, 5, 500)  # [µm]
    pressure = 1  # [bar]
    temperature = 200  # [K]

    example_opacity = interp_opacity(wavelength, temperature, pressure)

    plt.semilogy(wavelength, example_opacity, label=f"T={temperature} K")
    plt.legend()
    plt.gca().set(
        xlabel='Wavelength [µm]',
        ylabel='Opacity, $\kappa$ [cm$^2$ g$^{-1}$]'
    )

Now let’s specify an opacity for a gray cloud:

.. code-block:: python

    kappa_cloud = 5e-2  # [cm2/g]


Suppose we want to compute transmission spectra for several atmospheric
temperatures:

.. code-block:: python

    # interpolate for a range of wavelengths at one pressure and temperature:
    temperature = np.array([200, 400, 600, 800])  # [K]
    pressure = np.ones_like(temperature)  # [bar]

    example_opacity = interp_opacity(wavelength, temperature, pressure)

and now let's plot the result:

.. code-block:: python

    label = [f"{t} K" for t in temperature]
    
    plt.figure()
    plt.semilogy(wavelength, example_opacity.T, label=label)
    plt.semilogy(wavelength, kappa_cloud * np.ones_like(wavelength), ls='--', label="Cloud")
    plt.legend()
    plt.gca().set(
        xlabel='Wavelength [µm]',
        ylabel='Opacity, $\kappa$ [cm$^2$ g$^{-1}$]'
    )

.. plot::

    import matplotlib.pyplot as plt
    from shone.opacity import Opacity, generate_synthetic_opacity

    generate_synthetic_opacity()

    # load the one opacity file:
    opacity = Opacity.load_species_from_name('synthetic')

    # get a jitted 3D interpolator over wavelength, temperature, pressure:
    interp_opacity = opacity.get_interpolator()

    wavelength = np.linspace(1, 5, 500)  # [µm]
    temperature = np.array([200, 400, 600, 800])  # [K]
    pressure = np.ones_like(temperature)  # [bar]

    # interpolate for a range of wavelengths at one pressure and temperature:
    temperature = np.array([200, 400, 600, 800])
    label = [f"{t} K" for t in temperature]
    example_opacity = interp_opacity(wavelength, temperature, pressure)

    kappa_cloud = 5e-2  # [cm2/g]

    plt.figure()
    plt.semilogy(wavelength, example_opacity.T, label=label)
    plt.semilogy(wavelength, kappa_cloud * np.ones_like(wavelength), ls='--', label="Cloud")
    plt.legend()
    plt.gca().set(
        xlabel='Wavelength [µm]',
        ylabel='Opacity, $\kappa$ [cm$^2$ g$^{-1}$]'
    )

Compute a transmission spectrum
-------------------------------

We can compute a transmission spectrum for an Earth-sized planet
transiting a Sun-like star using
`~shone.transmission.heng_kitzmann_2017.transmission_radius_isothermal`:

.. code-block:: python

    R_0 = 1 * u.R_earth  # reference radius
    P_0 = 1 * u.bar  # reference pressure
    T_0 = 290 * u.K  # reference temperature
    mmw = 28 * m_p  # mean molecular weight (AMU)
    g = 9.8 * u.m / u.s**2  # surface gravity
    
    # convert the arguments from astropy `Quantity`s to 
    # floats in cgs units:
    args = (R_0, P_0, T_0, mmw, g)
    cgs_args = (arg.cgs.value for arg in args)
    
    # compute the planetary radius as a function of wavelength:
    Rp = heng_kitzmann_2017.transmission_radius_isothermal(
        example_opacity + kappa_cloud, *cgs_args
    )
    
    # convert to transit depth:
    Rstar = (1 * u.R_sun).cgs.value
    transit_depth_ppm = 1e6 * (Rp / Rstar) ** 2

Now let's plot the result:

.. code-block:: python

    label = [f"{t} K" for t in temperature]
    plt.plot(wavelength, transit_depth_ppm.T, label=label)
    plt.legend()
    plt.gca().set(
        xlabel='Wavelength [µm]',
        ylabel='Transit depth [ppm]'
    )

.. plot::

    import matplotlib.pyplot as plt

    import astropy.units as u
    from astropy.constants import m_p

    from shone.transmission import heng_kitzmann_2017
    from shone.opacity import Opacity, generate_synthetic_opacity

    generate_synthetic_opacity()

    # load the one opacity file:
    opacity = Opacity.load_species_from_name('synthetic')

    # get a jitted 3D interpolator over wavelength, temperature, pressure:
    interp_opacity = opacity.get_interpolator()

    wavelength = np.linspace(1, 5, 500)  # [µm]

    temperature = np.array([200, 400, 600, 800])  # [K]
    pressure = np.ones_like(temperature)  # [bar]


    temperature = np.array([200, 400, 600, 800])
    label = [f"{t} K" for t in temperature]

    example_opacity = interp_opacity(wavelength, temperature, pressure)

    kappa_cloud = 5e-2  # [cm2/g]

    R_0 = 1 * u.R_earth  # reference radius
    P_0 = 1 * u.bar  # reference pressure
    T_0 = 290 * u.K  # reference temperature
    mmw = 28 * m_p  # mean molecular weight (AMU)
    g = 9.8 * u.m / u.s**2  # surface gravity

    # convert the arguments from astropy `Quantity`s to
    # floats in cgs units:
    args = (R_0, P_0, T_0, mmw, g)
    cgs_args = (arg.cgs.value for arg in args)

    # compute the planetary radius as a function of wavelength:
    Rp = heng_kitzmann_2017.transmission_radius_isothermal(example_opacity + kappa_cloud, *cgs_args)

    # convert to transit depth:
    Rstar = (1 * u.R_sun).cgs.value
    transit_depth_ppm = 1e6 * (Rp / Rstar) ** 2

    label = [f"{t} K" for t in temperature]
    plt.plot(wavelength, transit_depth_ppm.T, label=label)
    plt.legend()
    plt.gca().set(
        xlabel='Wavelength [µm]',
        ylabel='Transit depth [ppm]'
    )

.. _transmission_de_wit:

General transmission spectra
============================

Above we demonstrated :ref:`transmission_heng_kitzmann` with a semi-analytic approach.
In this section we'll use the general transmission spectrum model from
`de Wit & Seager (2013) <https://ui.adsabs.harvard.edu/abs/2013Sci...342.1473D/abstract>`_.


.. plot::

    import numpy as np
    import astropy.units as u

    from jax import numpy as jnp

    from shone.chemistry import FastchemWrapper
    from shone.opacity import generate_synthetic_opacity
    from shone.transmission import de_wit_seager_2013

    opacity = generate_synthetic_opacity()
    interp_opacity = opacity.get_interpolator()

    wavelength = np.geomspace(0.5, 5, 500)
    pressure = np.geomspace(1e-6, 1e3)  # [bar]
    temperature = 500 * np.ones_like(pressure)  # [K]

    P_0 = pressure[30]
    T_0 = temperature[30]
    R_0 = (1 * u.R_jup).cgs.value
    mmw = 2.328  # [AMU]
    g = 3000  # [cm/s2]
    weights_amu = jnp.array([3])  # [AMU]
    synth_vmr = 1e-8

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