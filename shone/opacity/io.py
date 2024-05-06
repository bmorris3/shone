import os
from tempfile import TemporaryDirectory
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
    shone_dir = TemporaryDirectory().name


class Opacity:
    def __init__(self, path=None, grid=None):
        if path is not None:
            grid = xr.load_dataarray(path)
        self.grid = grid

    def get_interpolator(self):
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
        if shone_directory is None:
            shone_directory = shone_dir
        nc_paths = glob(os.path.join(shone_directory, "*.nc"))
        table_contents = {
            "name": [],
            "species": [],
            "charge": [],
            "line_list": [],
            "path": [],
        }
        for i, path in enumerate(nc_paths):
            file_name = os.path.basename(path).split('.nc')[0]
            species, line_list = file_name.split('__')
            table_contents["line_list"].append(line_list)

            if '_' in species:
                species, charge = species.split('_')
                table_contents["charge"].append(int(charge))
            else:
                table_contents["charge"].append(np.ma.masked)

            table_contents["species"].append(species)
            table_contents["name"].append(isotopologue_to_species(species))
            table_contents["path"].append(path)
        table = Table(table_contents)
        table.sort('name')
        table['index'] = np.arange(len(table))
        table.add_index("index")
        return table

    @classmethod
    def load_species_from_index(cls, idx):
        species = cls.get_available_species()
        path = species.loc[idx]['path']
        return cls(path=path)

    @classmethod
    def load_species_from_name(cls, name):
        species = cls.get_available_species()
        table_row = species[species['name'] == name]
        print(table_row)

        if len(table_row) > 1:
            raise ValueError(f"More than one of the available files has "
                             f"a species named {name}.")
        return cls(path=table_row['path'][0])


def generate_synthetic_opacity(filename='synthetic__example.nc'):
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

    example_opacity = xr.Dataset(
        data_vars=dict(
            opacity=(["temperature", "pressure", "wavelength"],
                     kappa.astype(np.float32))
        ),
        coords=dict(
            temperature=temperature,
            pressure=pressure,
            wavelength=wavelength
        )
    )

    description = "Example opacity grid for shone demos."
    example_opacity.attrs['description'] = description

    if not filename.endswith('.nc'):
        filename += '.nc'

    example_opacity.to_netcdf(os.path.join(shone_dir, filename))
