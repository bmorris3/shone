import os
from functools import partial
from glob import glob

import numpy as np
import xarray as xr
from astropy.table import Table

from jax import numpy as jnp, jit
from tensorflow_probability.substrates.jax.math import batch_interp_rectilinear_nd_grid as nd_interp

from shone.chemistry import isotopologue_to_species

__all__ = ['Opacity', 'generate_synthetic_opacity']


on_rtd = os.getenv('READTHEDOCS', False)

shone_dir = os.path.expanduser(os.path.join("~", ".shone"))

if on_rtd:
    # use a temporary directory on readthedocs:
    shone_dir = os.path.abspath('./.')


class Opacity:
    """
    Load and interpolate from a pre-computed opacity grid.
    """
    def __init__(self, path=None, grid=None):
        """
        Parameters
        ----------
        path : str, path-like
            File path for an opacity file.
        grid : `~xarray.Dataset`
            The Dataset of an already-loaded opacity grid.
        """
        if path is not None:
            with xr.open_dataset(path) as ds:
                attrs = ds.attrs
                grid = ds.opacity.assign_attrs(attrs)

        if isinstance(grid, xr.Dataset):
            grid = grid.opacity.assign_attrs(grid.attrs)

        self.grid = grid

    def get_interpolator(self):
        """
        Return a jitted opacity interpolator for any number of
        wavelength points, and one temperature and pressure.

        Returns
        -------
        interp : function
            A just-in-time compiled opacity interpolator.
        """
        x_grid_points = (
            jnp.float32(self.grid.temperature.to_numpy()),
            jnp.float32(self.grid.pressure.to_numpy()),
            jnp.float32(self.grid.wavelength.to_numpy()),
        )

        @partial(jit, donate_argnames=('grid',))
        def interp(
                interp_wavelength, interp_temperature, interp_pressure,
                grid=self.grid.to_numpy()
        ):
            interp_point = jnp.column_stack([
                jnp.broadcast_to(interp_temperature, interp_wavelength.shape),
                jnp.broadcast_to(interp_pressure, interp_wavelength.shape),
                interp_wavelength,
            ])

            return nd_interp(
                interp_point,
                x_grid_points,
                grid,
                axis=0
            )

        return interp

    @classmethod
    def get_available_species(self, shone_directory=None):
        """
        Get a table of available opacity files.

        Parameters
        ----------
        shone_directory : str, path-like (optional)
            Directory containing opacity files.

        Returns
        -------
        table : `~astropy.table.Table`
            Table containing entries for every opacity file
            available on disk.
        """
        # prevent circular import:
        from shone.opacity.dace import parse_nc_path_molecule, parse_nc_path_atom

        if shone_directory is None:
            shone_directory = shone_dir
        nc_paths = glob(os.path.join(shone_directory, "*.nc"))
        table_contents = {
            "name": [],
            "species": [],
            "charge": [],
            "line_list": [],
            "version": [],
            "size_gb": [],
            "path": [],
        }
        for i, path in enumerate(nc_paths):
            file_name = os.path.basename(path).split('.nc')[0]

            # set defaults to overwrite if available:
            charge = np.ma.masked
            atom = None
            isotopologue = None

            if len(file_name.split('_')) == 8:
                (
                    atom, charge, line_list,
                    temperature_range, pressure_range,
                    version
                ) = parse_nc_path_atom(file_name)
            elif len(file_name.split('_')) == 7:
                (
                    isotopologue, line_list,
                    temperature_range, pressure_range,
                    version
                ) = parse_nc_path_molecule(file_name)
            else:
                continue
            species = isotopologue or atom

            table_contents["name"].append(isotopologue_to_species(species))
            table_contents["species"].append(species)
            table_contents["charge"].append(charge)
            table_contents["line_list"].append(line_list)
            table_contents["version"].append(version)
            table_contents["size_gb"].append(
                round(os.stat(path).st_size / 1e9, 3)
            )
            table_contents["path"].append(path)

        table = Table(table_contents)
        table.sort('name')
        table['index'] = np.arange(len(table))
        table.add_index("index")
        return table

    @classmethod
    def load_species_from_index(cls, idx):
        """
        Load an opacity file from its index listed in the output
        of `~shone.opacity.Opacity.get_available_species`.

        Parameters
        ----------
        idx : int
            Index of the opacity file to load.
        """
        species = cls.get_available_species()
        path = species.loc[idx]['path']
        return cls(path=path)

    @classmethod
    def load_species_from_name(cls, name):
        """
        Load an opacity file from its name listed in the output
        of `~shone.opacity.Opacity.get_available_species`.

        Parameters
        ----------
        name : str
            Name of the opacity file to load. Since the "name" entry
            isn't guaranteed to be unique, an error is raised
            if more than one file is available by this name.
        """
        species = cls.get_available_species()
        table_row = species[species['name'] == name]

        if len(table_row) > 1:
            raise ValueError(f"More than one of the available files has "
                             f"a species named {name}.")
        return cls(path=table_row['path'][0])


def generate_synthetic_opacity(filename="synthetic_example_0_0_0_0_0.nc"):
    """
    Construct a netCDF file containing a synthetic opacity grid.
    Useful for examples in the tests and documentation.

    Parameters
    ----------
    filename : str, path-like (optional)
        File name.
    """
    output_path = os.path.join(shone_dir, filename)
    if os.path.exists(output_path):
        return Opacity(path=output_path)

    np.random.seed(42)

    temperature = np.arange(200, 1200, 200)
    pressure = np.geomspace(1e-6, 10, 2)
    wavelength = np.geomspace(0.5, 5, 1000)

    kappa = np.random.uniform(size=wavelength.size)
    x = np.arange(len(kappa))
    kernel = np.exp(-0.5 * (x - x.size / 2) ** 2 / 15 ** 2)
    kappa = np.convolve(kappa - kappa.mean(), kernel, mode='same')
    pressure_dim = np.ones((1, pressure.size, 1))
    kappa = np.power(10, (temperature[:, None, None] / 200) ** 0.7 *
                     kappa[None, None, :]) * pressure_dim

    description = "Example opacity grid for shone demos."

    example_opacity = xr.Dataset(
        data_vars=dict(
            opacity=(["temperature", "pressure", "wavelength"],
                     kappa.astype(np.float32))
        ),
        coords=dict(
            temperature=temperature,
            pressure=pressure,
            wavelength=wavelength
        ),
        attrs=dict(description=description)
    )

    if not filename.endswith('.nc'):
        filename += '.nc'

    example_opacity.to_netcdf(output_path)

    return Opacity(grid=example_opacity)
