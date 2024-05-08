import os
import numpy as np
from pyfastchem import (
    FastChem, FastChemInput, FastChemOutput
)
from astropy.constants import m_p, k_B
from astropy.table import Table

# constants in cgs:
m_p = m_p.cgs.value
k_B = k_B.cgs.value


def run_fastchem(
    temperature, pressure,
    metallicity=None,
    c_to_o_ratio=None,
    elemental_abundances_path=None,
    fit_coefficients_path=None
):
    if elemental_abundances_path is None:
        elemental_abundances_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'data',
            'asplund_2020.dat'
        )

    if fit_coefficients_path is None:
        fit_coefficients_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'data', 'logK.dat'
        )

    if metallicity is None:
        metallicity = 1

    fastchem = FastChem(
        elemental_abundances_path,
        fit_coefficients_path, 0
    )

    solar_abundances = np.array(fastchem.getElementAbundances())

    fastchem_species = get_fastchem_species(fastchem)
    is_element = fastchem_species['type'] == 'element'
    multiplier = np.where(
        ~np.isin(fastchem_species[is_element]['symbol'], ['H', 'He']),
        metallicity,
        1
    )

    abundances_with_metallicity = solar_abundances * multiplier

    if c_to_o_ratio is not None:
        index_C = fastchem.getElementIndex('C')
        index_O = fastchem.getElementIndex('O')

        abundances_with_metallicity[index_C] = (
            abundances_with_metallicity[index_O] * c_to_o_ratio
        )

    fastchem.setElementAbundances(abundances_with_metallicity)

    # create the input and output structures for FastChem
    input_data = FastChemInput()
    output_data = FastChemOutput()

    input_data.temperature = temperature
    input_data.pressure = pressure

    # run FastChem on the entire p-T structure
    fastchem.calcDensities(input_data, output_data)

    n_densities = np.array(output_data.number_densities)  # [cm-3]
    gas_number_density = pressure / (k_B * temperature)  # [cm-3]

    vmr = n_densities / gas_number_density[:, None]

    return vmr, fastchem_species


def get_fastchem_species(fastchem):
    element_symbols = []
    gas_species_symbols = []
    idx = 0
    while fastchem.getElementSymbol(idx) != '':
        element_symbols.append(
            [idx, fastchem.getElementName(idx),
             fastchem.getElementSymbol(idx),
             fastchem.getElementWeight(idx),
             'element',
            ]
        )
        idx += 1

    while fastchem.getGasSpeciesSymbol(idx) != '':
        gas_species_symbols.append(
            [idx, fastchem.getGasSpeciesName(idx),
             fastchem.getGasSpeciesSymbol(idx),
             fastchem.getGasSpeciesWeight(idx),
             'gas species',
            ]
        )
        idx += 1

    symbols = list(element_symbols)
    symbols.extend(gas_species_symbols)

    fastchem_species = Table(
        rows=symbols,
        names='index name symbol weight type'.split()
    )
    fastchem_species.add_index('symbol')

    return fastchem_species
