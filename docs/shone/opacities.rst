.. _opacities:

*********
Opacities
*********

Downloading opacities
---------------------

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
