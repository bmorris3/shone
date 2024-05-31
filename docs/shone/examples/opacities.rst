.. _opacities:

*********
Opacities
*********

Download opacities
------------------

Several helper methods are included in ``shone`` for downloading and archiving
local copies of opacities stored on `DACE <https://dace.unige.ch/>`_ via
the package ``dace-query``. Users must register for an API key to use ``dace-query``,
see `their documentation for instructions
<https://dace-query.readthedocs.io/en/latest/dace_introduction.html#authentication>`_.

To download the opacities for an atom with :func:`~shone.opacity.download_atom`, specify
the atom's name and charge, and optionally limit the pressure range (in log10 bars)
and temperature range (in Kelvin):

.. code-block:: python

    from shone.opacity.dace import download_atom

    download_atom(
        atom='Na',
        charge=0,
        temperature_range=[2500, 2500]
    )

You may query for molecules by their common names (for the most common
isotopologue) with :func:`~shone.opacity.download_molecule` like so:

.. code-block:: python

    from shone.opacity.dace import download_molecule

    download_molecule(
        molecule_name='H2O',
        temperature_range=[2500, 2500],
        pressure_range=[-6, -6]
    )

or for a specific isotopologue:

.. code-block:: python

    from shone.opacity.dace import download_molecule

    download_molecule(
        isotopologue='1H2-16O',
        temperature_range=[2500, 2500],
        pressure_range=[-6, -6]
    )

These methods download the opacity grids from DACE, and store them in a netCDF file
in your home directory with the name ``.shone/``.

Example opacity file
--------------------

It is often useful to have a small, synthetic opacity file for simple demos and testing.
You can generate one with `~shone.opacity.generate_synthetic_opacity()`:

.. code-block:: python

    from shone.opacity import generate_synthetic_opacity

    opacity = generate_synthetic_opacity()

You can also lookup and load opacities within the cache like so:

.. code-block:: python

    from shone.opacity import Opacity

    # load the one opacity file:
    opacity = Opacity.load_species_from_name('synthetic')

Opening opacities
-----------------

Opacities are loaded and interpolated with the `~shone.opacity.Opacity` class.

.. code-block:: python

    from shone.opacity import Opacity

We can check which species are already chached and available on your
machine using `~shone.opacity.Opacity.get_available_species()`:

.. code-block:: python

    Opacity.get_available_species()

This will return a table of available opacity grids on disk.

Let's load the opacity created by `~shone.opacity.generate_synthetic_opacity()`
(see the step above):

.. code-block:: python

    opacity = Opacity.load_species_from_name('synthetic')

The `~shone.opacity.Opacity` object contains the opacity grid as a `~xarray.DataArray`
in its `grid` attribute. You can see the dimensions of the grid with:

.. code-block:: python

    >>> print(opacity.grid.coords)
    Coordinates:
      * wavelength   (wavelength) float64 0.5 0.5012 0.5023 ... 4.977 4.988 5.0
      * temperature  (temperature) int32 200 400 600 800 1000
      * pressure     (pressure) float64 1e-06 10.0

The coordinates in the `~xarray.DataArray` are wavelength in microns,
temperature in K, and pressure in bar. To learn to use the xarray API
directly on the grid attribute, refer to the xarray docs on `indexing
and selecting data <https://docs.xarray.dev/en/stable/user-guide/indexing.html>`_
and `interpolating
<https://docs.xarray.dev/en/stable/user-guide/interpolation.html>`_.

You can inspect the opacities from one temperature and pressure slice like so:

.. code-block:: python

    import matplotlib.pyplot as plt

    opacity_sample = opacity.grid.sel(
        dict(
            pressure=10,  # [bar]
            temperature=200  # [K]
        )
    )

    plt.semilogy(
        opacity_sample.wavelength, opacity_sample,
        label=f"T={opacity_sample.temperature} K"
    )
    plt.gca().set(
        xlabel='Wavelength [µm]',
        ylabel='Opacity, $\kappa$ [cm$^2$ g$^{-1}$]'
    )

.. plot::

    import matplotlib.pyplot as plt
    from shone.opacity import Opacity, generate_synthetic_opacity

    opacity = generate_synthetic_opacity()
    opacity_sample = opacity.grid.sel(
        dict(
            pressure=10,  # [bar]
            temperature=200  # [K]
        )
    )

    plt.semilogy(
        opacity_sample.wavelength, opacity_sample,
        label=f"T={opacity_sample.temperature} K"
    )
    plt.gca().set(
        xlabel='Wavelength [µm]',
        ylabel='Opacity, $\kappa$ [cm$^2$ g$^{-1}$]'
    )

Interpolating opacities
-----------------------

Often in ``shone`` we will need to interpolate over the opacity grid within
compiled code, so we will use a just-in-time compiled interpolator on the
opacity grid. You can produce a function to do these compiled interpolations
with `~shone.opacity.Opacity.get_interpolator`:

.. code-block:: python

    # get a jitted 3D interpolator over wavelength, temperature, pressure:
    interp_opacity = opacity.get_interpolator()

Now you can get the opacity at wavelengths, temperatures, and pressures that weren't on
the grid:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    wavelength = np.linspace(1, 5, 500)  # [µm]
    pressure = 0.3  # [bar]
    temperature = 555  # [K]

    example_opacity = interp_opacity(wavelength, temperature, pressure)

    plt.semilogy(wavelength, example_opacity, label=f"T={temperature} K")
    plt.legend()
    plt.gca().set(
        xlabel='Wavelength [µm]',
        ylabel='Opacity, $\kappa$ [cm$^2$ g$^{-1}$]'
    )

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from shone.opacity import Opacity, generate_synthetic_opacity

    opacity = generate_synthetic_opacity()

    # get a jitted 3D interpolator over wavelength, temperature, pressure:
    interp_opacity = opacity.get_interpolator()

    wavelength = np.linspace(1, 5, 500)  # [µm]
    pressure = 0.3  # [bar]
    temperature = 555  # [K]

    example_opacity = interp_opacity(wavelength, temperature, pressure)

    plt.semilogy(wavelength, example_opacity, label=f"T={temperature} K")
    plt.legend()
    plt.gca().set(
        xlabel='Wavelength [µm]',
        ylabel='Opacity, $\kappa$ [cm$^2$ g$^{-1}$]'
    )

We can compute opacities over a series of temperatures and pressures:

.. code-block:: python

    from jax import numpy as jnp

    temperatures = jnp.array([222, 333, 444])
    pressures = jnp.array([0.1, 0.5, 0.9])

    example_opacity = interp_opacity(wavelength, temperatures, pressures)

For M wavelengths and N samples in pressure and temperature, the
output will have the shape (N, M).

Crop an opacity grid
--------------------

Suppose the full opacity grid covers a wider wavelength range than you
need for your calculation. You can limit the size of the array that
gets read into JAX by cropping the opacity grid to the relevant limits
in wavelength, pressure, and temperature.

The example opacity file above is small compared to real ones, and contains
this many opacity entries:

.. code-block:: python

    >>> print(opacity.grid.size)
    10000

To reduce the size of the opacity grid, we crop the opacity grid on
the wavelength range :math:`1.5 < \lambda < 2.5` µm:

.. code-block:: python

    crop = ((1.5 < opacity.grid.wavelength) & (opacity.grid.wavelength < 2.5))
    opacity.grid = opacity.grid.isel(wavelength=crop)

and we can see the reduction in size:

.. code-block:: python

    print(opacity.grid.size)
    2220


.. _tiny_opacity_archive:

Tiny opacity archives
---------------------

It can be cumbersome to work with opacity grids, given that they
may be tens of GB in size. For simple examples in the documentation
and tests, ``shone`` has very lightweight representations of the full
opacity grids for several molecules, which we call "tiny opacity
archives".

To load one of these example opacities, run:

.. code-block:: python

    from shone.opacity import Opacity

    # load the tiny opacity archive:
    tiny_opacity = Opacity.load_demo_species('H2O')

After you load them, these opacity files work just like
the real ones. We can interpolate the grid at several
temperatures and plot the results like this:

.. code-block:: python

    # get a jitted interpolator:
    interp_opacity = tiny_opacity.get_interpolator()

    # get opacity at several temperatures, all at 1 bar:
    wavelength = np.geomspace(0.6, 5, 500)
    temperature = np.geomspace(100, 3000, 5)
    pressure = np.ones_like(temperature)  # [bar]

    kappa = interp_opacity(wavelength, temperature, pressure)

    # plot the opacities:
    n = len(temperature)
    ax = plt.gca()

    for i in range(n):
        color = plt.cm.plasma(i / n)
        label = f"{temperature[i]:.0f} K"
        ax.semilogy(wavelength, kappa[i], label=label, color=color)

    ax.legend(title='Temperature', loc='lower right', framealpha=1)
    ax.set(
        xlabel='Wavelength [µm]',
        ylabel='Opacity [cm$^2$ g$^{-1}$]',
        title="Demo opacity: H$_2$O"
    )


.. plot::

    import numpy as np
    import matplotlib.pyplot as plt

    from shone.opacity import Opacity

    # load the tiny opacity archive:
    tiny_opacity = Opacity.load_demo_species('H2O')

    # get a jitted interpolator:
    interp_opacity = tiny_opacity.get_interpolator()

    # get opacity at several temperatures:
    wavelength = np.geomspace(0.6, 5, 500)
    temperature = np.geomspace(100, 3000, 5)
    pressure = np.ones_like(temperature)  # [bar]

    kappa = interp_opacity(wavelength, temperature, pressure)

    n = len(temperature)
    ax = plt.gca()

    for i in range(n):
        color = plt.cm.plasma(i / n)
        label = f"{temperature[i]:.0f} K"
        ax.semilogy(wavelength, kappa[i], label=label, color=color)

    ax.legend(title='Temperature', loc='lower right', framealpha=1)
    ax.set(
        xlabel='Wavelength [µm]',
        ylabel='Opacity [cm$^2$ g$^{-1}$]',
        title="Demo opacity: H$_2$O"
    )
