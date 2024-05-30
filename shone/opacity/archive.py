"""
Generate and/or reconstruct "tiny opacity archives" from
real opacity grids. This is useful for docs and tests.

The real opacity grids for TiO, H2O,
and CO2 are 12, 21, and 5 GB respectively. The tiny
representations are all 100 KB or less.

The algorithm is an adaptation of the `tynt` filter
approximation methods in:
https://github.com/bmorris3/tynt
"""

import os
import numpy as np
from tqdm.auto import tqdm
import xarray as xr

from shone.opacity import Opacity
from shone.config import tiny_archives_dir
from shone.spectrum import bin_spectrum


pkg_data_directory = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data'
)


def opacity_to_fft_archive(opacity, path, n_terms=25):
    wavelength = np.geomspace(0.4, 5, 300)

    log_wavelength = np.log10(wavelength)
    wavelength_min = wavelength.min()
    wavelength_max = wavelength.max()
    log_lam_min = log_wavelength.min()
    log_lam_max = log_wavelength.max()
    n_lambda = len(wavelength)
    grid_wavelength = opacity.grid.wavelength.to_numpy()

    # Create a simplified wavelength grid:
    simplified_wavelength = np.linspace(log_lam_min, log_lam_max, n_lambda)

    skip_pt_samples = 6
    dimensions = opacity.grid.sizes
    pressure_temperature_shape = np.array(
        list(dimensions.values())[:-1]
    ) // skip_pt_samples + 1

    fft_results = np.zeros(
        pressure_temperature_shape.tolist() + [n_terms + 7],
        dtype=np.complex64
    )

    for temperature_idx in tqdm(range(0, dimensions['temperature'], skip_pt_samples)):
        for pressure_idx in tqdm(range(0, dimensions['pressure'], skip_pt_samples)):
            wavelength_mask = (
                (wavelength_min <= opacity.grid.wavelength) &
                (opacity.grid.wavelength <= wavelength_max)
            )
            op = opacity.grid.isel(
                temperature=temperature_idx,
                pressure=pressure_idx,
                wavelength=wavelength_mask
            ).to_numpy()

            op_binned = bin_spectrum(
                wavelength, grid_wavelength[wavelength_mask], op
            )

            log_spectrum = np.log10(op_binned)
            log_spectrum_min = log_spectrum.min()
            log_spectrum_max = log_spectrum.max()
            renormed_spectrum = (
                (log_spectrum - log_spectrum_min) /
                (log_spectrum_max - log_spectrum_min)
            )
            tr_interp = np.interp(
                simplified_wavelength, log_wavelength, renormed_spectrum
            )

            # Take the DFT of the interpolated spectrum
            fft = np.fft.rfft(tr_interp, norm="ortho")[:n_terms]
            result = np.concatenate([
                [
                    float(opacity.grid.temperature[temperature_idx]),
                    float(opacity.grid.pressure[pressure_idx]),
                    log_lam_min, log_lam_max, n_lambda,
                    log_spectrum_min, log_spectrum_max
                ], fft
            ])

            fft_results[
                temperature_idx // skip_pt_samples,
                pressure_idx // skip_pt_samples
            ] = result

    np.save(path, fft_results)


def reconstruct_fft_archive(result):
    [
        temperature, pressure, log_lam_min, log_lam_max,
        n_lambda, log_spectrum_min, log_spectrum_max
    ] = result[:7]
    fft = result[7:]

    reconstructed_wavelength = 10 ** np.linspace(
        log_lam_min.real, log_lam_max.real, int(n_lambda.real)
    )

    irfft = np.fft.irfft(
        fft, n=len(reconstructed_wavelength), norm="ortho"
    )

    log_opacity = (
        irfft * (log_spectrum_max - log_spectrum_min)
    ) + log_spectrum_min
    reconstructed_opacity = 10 ** log_opacity.real

    return (
        float(temperature.real), float(pressure.real),
        reconstructed_wavelength, reconstructed_opacity
    )


def generate_tiny_opacity_archives(species=['TiO', 'CO2', 'H2O']):
    for molecule in species:
        print(f'Archiving: {molecule}')
        opacity = Opacity.load_species_from_name(molecule)
        path = os.path.join(pkg_data_directory, f'{molecule}_opacity.npy')
        opacity_to_fft_archive(opacity, path)


def unpack_tiny_opacity_archives(species=['TiO', 'CO2', 'H2O']):

    os.makedirs(tiny_archives_dir, exist_ok=True)

    for molecule in species:
        path = os.path.join(pkg_data_directory, f'{molecule}_opacity.npy')
        archive = np.load(path)
        temperature = np.unique(archive[:, :, 0].real)
        pressure = np.unique(archive[:, :, 1].real)
        (_, _, wavelength, _) = reconstruct_fft_archive(archive[0, 0])

        reconstructed_opacity = np.zeros(
            (temperature.size, pressure.size, wavelength.size)
        )
        for i, temp in enumerate(temperature):
            for j, press in enumerate(pressure):
                op_ij = reconstruct_fft_archive(archive[i, j])[-1]
                reconstructed_opacity[i, j] = op_ij

        info_msg = (
            "This is a tiny opacity archive. It has been made for "
            "use in the documentation and tests. It's not the real "
            f"opacity for {molecule}. Don't use it for science."
        )

        ds = xr.Dataset(
            data_vars=dict(
                opacity=(["temperature", "pressure", "wavelength"],
                         reconstructed_opacity)
            ),
            coords=dict(
                temperature=(["temperature"], temperature),
                pressure=(["pressure"], pressure),
                wavelength=wavelength
            ),
            attrs={"info": info_msg}
        )

        nc_path = os.path.join(
            tiny_archives_dir, f'{molecule}_reconstructed.nc'
        )
        ds.to_netcdf(nc_path)
