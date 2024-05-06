import logging
import warnings
from functools import cached_property
import os
import tarfile
import shutil
from glob import glob

import numpy as np
import xarray as xr
from astropy.table import Table
from dace_query.opacity import Molecule, Atom

from ..chemistry import species_name_to_common_isotopologue_name

interp_kwargs = dict(
    method='nearest',
    kwargs=dict(fill_value="extrapolate")
)

__all__ = [
    'download_molecule',
    'download_atom'
]


class AvailableOpacities:
    @cached_property
    def atomic(self):
        return get_atomic_database()

    @cached_property
    def molecular(self):
        return get_molecular_database()

    def get_atomic_database_entry(self, atom, charge, line_list):
        table = self.atomic
        return table[(
            (table['atom'] == atom) &
            (table['line_list'] == line_list) &
            (table['charge'] == charge)
        )]

    def get_molecular_database_entry(self, isotopologue, line_list):
        table = self.molecular
        return table[(
            (table['isotopologue'] == isotopologue) &
            (table['line_list'] == line_list)
        )]

    def get_molecular_line_lists(self, isotopologue):
        table = self.molecular
        return set(table[table['isotopologue'] == isotopologue]['line_list'])

    def get_atomic_line_lists(self, atom):
        table = self.atomic
        return set(table[table['atom'] == atom]['line_list'])

    def get_atomic_latest_version(self, atom, charge, line_list):
        table = self.atomic
        matches = table[(
            (table['atom'] == atom) &
            (table['line_list'] == line_list) &
            (table['charge'] == charge)
        )]
        return max(set(matches['version']))

    def get_molecular_latest_version(self, isotopologue, line_list):
        table = self.molecular
        matches = table[(
            (table['isotopologue'] == isotopologue) &
            (table['line_list'] == line_list)
        )]
        return max(set(matches['version']))

    def get_atomic_pT_range(self, atom, charge, line_list):
        table = self.get_atomic_database_entry(atom, charge, line_list)
        temperature_range = (
            int(table['temp_min_k'][0]),
            int(table['temp_max_k'][0])
        )
        pressure_range = (
            float(table['pressure_min_exponent_b'][0]),
            float(table['pressure_max_exponent_b'][0])
        )
        return temperature_range, pressure_range

    def get_molecular_pT_range(self, isotopologue, line_list):
        table = self.get_molecular_database_entry(isotopologue, line_list)
        temperature_range = (
            int(table['temp_min_k'][0]),
            int(table['temp_max_k'][0])
        )
        pressure_range = (
            float(table['pressure_min_exponent_b'][0]),
            float(table['pressure_max_exponent_b'][0])
        )
        return temperature_range, pressure_range


available_opacities = AvailableOpacities()


def get_atomic_database():
    db = Atom.query_database()
    table = Table(db)
    table.add_index('atom')
    return table


def get_molecular_database():
    db = Molecule.query_database()
    table = Table(db)
    table.add_index('isotopologue')
    return table


def dace_download_molecule(
    isotopologue='48Ti-16O',
    line_list='Toto',
    temperature_range=None,
    pressure_range=None,
    version=1
):
    os.makedirs('tmp', exist_ok=True)
    archive_name = isotopologue + '__' + line_list + '.tar.gz'
    Molecule.download(
        isotopologue,
        line_list,
        float(version),
        temperature_range,
        pressure_range,
        output_directory='tmp',
        output_filename=archive_name
    )

    return os.path.join('tmp', archive_name)


def dace_download_atom(
    element='Na',
    charge=0,
    line_list='Kurucz',
    temperature_range=None,
    pressure_range=None,
    version=1
):
    os.makedirs('tmp', exist_ok=True)
    archive_name = element + '__' + line_list + '.tar.gz'
    Atom.download(
        element, charge,
        line_list, float(version),
        temperature_range,
        pressure_range,
        output_directory='tmp',
        output_filename=archive_name
    )
    return os.path.join('tmp', archive_name)


def untar_bin_files(archive_name):
    def bin_files(members):
        for tarinfo in members:
            if os.path.splitext(tarinfo.name)[1] == ".bin":
                yield tarinfo

    with tarfile.open(archive_name, 'r') as tar:
        tar.extractall(path='tmp/.', members=bin_files(tar))


def get_opacity_dir_path_molecule(isotopologue, linelist):
    find_e2b = glob(os.path.join('tmp', isotopologue + '__' + linelist + "*e2b"))
    if len(find_e2b):
        return find_e2b[0]
    return glob(os.path.join('tmp', isotopologue + '__' + linelist + "*"))[0]


def get_opacity_dir_path_atom(linelist):
    return glob(os.path.join('tmp', linelist + "*e2b"))[0]


def opacity_dir_to_netcdf(opacity_dir, outpath):
    temperature_grid = []
    pressure_grid = []

    for dirpath, dirnames, filenames in os.walk(opacity_dir):
        for filename in filenames:
            if not filename.endswith('.bin'):
                continue

            # Wavenumber points from range given in the file names
            temperature = int(filename.split('_')[3])
            sign = 1 if filename.split('_')[4][0] == 'p' else -1
            pressure = 10 ** (sign * float(filename.split('_')[4][1:].split('.')[0]) / 100)

            wl_start = int(filename.split('_')[1])
            wl_end = int(filename.split('_')[2])
            wlen = np.arange(wl_start, wl_end, 0.01)

            # catch divide by zero warning here:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)

                # Convert to micron
                wavelength = 1 / wlen / 1e-4

            unique_wavelengths = wavelength[1:][::-1]
            temperature_grid.append(temperature)
            pressure_grid.append(pressure)

    tgrid = np.sort(list(set(temperature_grid)))
    pgrid = np.sort(list(set(pressure_grid)))

    if len(pgrid) == 1:
        extrapolate_pgrid = True
        pgrid = np.concatenate([pgrid, 10 ** (-1 * np.log10(pgrid))])
    else:
        extrapolate_pgrid = False

    opacity_grid = np.zeros(
        (len(tgrid), len(pgrid), len(unique_wavelengths)), dtype='float32'
    )

    for dirpath, dirnames, filenames in os.walk(opacity_dir):
        for filename in filenames:
            if not filename.endswith('.bin'):
                continue

            opacity = np.fromfile(
                os.path.join(dirpath, filename), dtype=np.float32
            )[1:][::-1]

            # Wavenumber points from range given in the file names
            temperature = int(filename.split('_')[3])
            sign = 1 if filename.split('_')[4][0] == 'p' else -1
            pressure = 10 ** (sign * float(filename.split('_')[4][1:].split('.')[0]) / 100)

            temperature_ind = np.argmin(np.abs(tgrid - temperature))
            pressure_ind = np.argmin(np.abs(pgrid - pressure))

            opacity_grid[temperature_ind, pressure_ind, :] = opacity

    if extrapolate_pgrid:
        for dirpath, dirnames, filenames in os.walk(opacity_dir):
            for filename in filenames:
                opacity = np.fromfile(
                    os.path.join(dirpath, filename), dtype=np.float32
                )[1:][::-1]

                # Wavenumber points from range given in the file names
                temperature = int(filename.split('_')[3])
                # *Flip the sign for the extrapolated grid point in pressure*
                sign = -1 if filename.split('_')[4][0] == 'p' else 1
                pressure = 10 ** (sign * float(filename.split('_')[4][1:].split('.')[0]) / 100)

                temperature_ind = np.argmin(np.abs(tgrid - temperature))
                pressure_ind = np.argmin(np.abs(pgrid - pressure))

                opacity_grid[temperature_ind, pressure_ind, :] = opacity

    ds = xr.Dataset(
        data_vars=dict(
            opacity=(["temperature", "pressure", "wavelength"],
                     opacity_grid)
        ),
        coords=dict(
            temperature=(["temperature"], tgrid),
            pressure=(["pressure"], pgrid),
            wavelength=unique_wavelengths
        )
    )

    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

    ds.to_netcdf(outpath if outpath.endswith(".nc") else outpath + '.nc',
                 encoding={'opacity': {'dtype': 'float32'}})


def clean_up(bin_dir, archive_name):
    os.remove(archive_name)
    shutil.rmtree(bin_dir)


def download_molecule(
    isotopologue=None,
    molecule_name=None,
    line_list='first-found',
    temperature_range=None,
    pressure_range=None,
    version=None,
):
    """
    Download molecular opacity data from DACE.

    .. warning::
        This generates *very* large files. Only run this
        method if you have ~6 GB available per molecule.

    Parameters
    ----------
    isotopologue : str
        For example, "1H2-16O" for water.
    molecule_name : str
        Common name for the molecule, for example: "H2O"
    line_list : str, default is ``'first-found'``, optional
        For example, "POKAZATEL" for water. By default, the first available
        line list for this isotopologue is chosen.
    temperature_range : tuple, optional
        Tuple of integers specifying the min and max
        temperature requested. Defaults to the full
        range of available temperatures.
    pressure_range : tuple, optional
        Tuple of floags specifying the log base 10 of the
        min and max pressure [bar] requested. Defaults to the full
        range of available pressures.
    version : float, optional
        Version number of the line list in DACE. Defaults to the
        latest version.
    """
    if molecule_name is not None:
        isotopologue = species_name_to_common_isotopologue_name(molecule_name)

    available_line_lists = available_opacities.get_molecular_line_lists(isotopologue)

    if line_list == 'first-found':
        line_list = sorted(list(available_line_lists)).pop()
        logging.warning(f"Using first-found line list for {isotopologue}: '{line_list}'")

    elif line_list not in available_line_lists:
        raise ValueError(f"The requested '{line_list}' is not in the set of "
                         f"available line lists {available_line_lists}.")

    if version is None:
        version = available_opacities.get_molecular_latest_version(isotopologue, line_list)
        logging.warning(f"Using latest version of the line "
                        f"list '{line_list}' for {isotopologue}: {version}")

    if temperature_range is None or pressure_range is None:
        dace_temp_range, dace_press_range = available_opacities.get_molecular_pT_range(
            isotopologue, line_list
        )

    if temperature_range is None:
        temperature_range = dace_temp_range

    if pressure_range is None:
        pressure_range = dace_press_range

    archive_name = dace_download_molecule(
        isotopologue, line_list, temperature_range, pressure_range, version=version
    )
    untar_bin_files(archive_name)
    bin_dir = get_opacity_dir_path_molecule(
        isotopologue, line_list
    )

    nc_path = os.path.join(
        os.path.expanduser('~'),
        '.shone',
        isotopologue + '__' + line_list + '.nc'
    )
    opacity_dir_to_netcdf(bin_dir, nc_path)
    clean_up(bin_dir, archive_name)


def download_atom(atom, charge, line_list='first-found',
                  temperature_range=None, pressure_range=None, version=None):
    """
    Download atomic opacity data from DACE.

    .. warning::
        This generates *very* large files. Only run this
        method if you have ~6 GB available per molecule.

    Parameters
    ----------
    atom : str
        For example, "Na" for sodium.
    charge : int
        For example, 0 for neutral.
    line_list : str, default is ``'first-found'``, optional
        For example, "Kurucz". By default, the first available
        line list for this atom/charge is chosen.
    temperature_range : tuple, optional
        Tuple of integers specifying the min and max
        temperature requested. Defaults to the full
        range of available temperatures.
    pressure_range : tuple, optional
        Tuple of floags specifying the log base 10 of the
        min and max pressure [bar] requested. Defaults to the full
        range of available pressures.
    version : float, optional
        Version number of the line list in DACE. Defaults to the
        latest version.
    """
    available_line_lists = available_opacities.get_atomic_line_lists(atom)

    if line_list == 'first-found':
        line_list = sorted(list(available_line_lists)).pop()
        logging.warning(f"Using first-found line list for {atom}: '{line_list}'")

    elif line_list not in available_line_lists:
        raise ValueError(f"The requested '{line_list}' is not in the set of "
                         f"available line lists {available_line_lists}.")

    if temperature_range is None or pressure_range is None:
        dace_temp_range, dace_press_range = available_opacities.get_atomic_pT_range(
            atom, charge, line_list
        )

    if version is None:
        version = available_opacities.get_atomic_latest_version(atom, charge, line_list)
        logging.warning(f"Using latest version of the line "
                        f"list '{line_list}' for {atom}: {version}")

    if temperature_range is None:
        temperature_range = dace_temp_range

    if pressure_range is None:
        pressure_range = dace_press_range

    archive_name = dace_download_atom(
        atom, charge, line_list, temperature_range, pressure_range, version=version
    )
    untar_bin_files(archive_name)
    bin_dir = get_opacity_dir_path_atom(line_list)

    nc_path = os.path.join(
        os.path.expanduser('~'),
        '.shone',
        atom + '_' + str(int(charge)) + '__' + line_list + '.nc'
    )
    opacity_dir_to_netcdf(bin_dir, nc_path)
    clean_up(bin_dir, archive_name)
