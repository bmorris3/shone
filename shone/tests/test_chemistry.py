import pytest

from ..chemistry import (
    species_name_to_fastchem_name, isotopologue_to_species,
    species_name_to_common_isotopologue_name
)


@pytest.mark.parametrize("isotopologue_name, species_name", (
    zip(['1H2-16O', 'Na', 'K', '48Ti-16O'], ["H2O", "Na", "K", "TiO"])
),)
def test_chemical_names_manipulation_0(isotopologue_name, species_name):
    # Test conversion of isotopologue name, like an opacity file with "1H2-16O"
    # to what I'll call the common "species name" like H2O
    assert isotopologue_to_species(isotopologue_name) == species_name


@pytest.mark.parametrize("species_name, fastchem_name", (
    zip(['H2O', 'TiO', 'VO', 'Na', 'K', 'CO', 'CrH',
         'CF4O', 'Al2Cl6', 'AlNaF4', 'ClAlF2'],
        ['H2O1', 'O1Ti1', 'O1V1', 'Na', 'K', 'C1O1', 'Cr1H1',
         'C1F4O1', 'Al2Cl6', 'Al1F4Na1', 'Al1Cl1F2'])
),)
def test_chemical_names_manipulation_1(species_name, fastchem_name):
    # Test conversion of common species name to a fastchem name which can
    # be called in fastchem.getSpeciesIndex(fastchem_name)
    assert species_name_to_fastchem_name(species_name) == fastchem_name


@pytest.mark.parametrize("species_name, iso_name", (
    zip(['H2O', 'TiO', 'VO', 'Na', 'K', 'CO', 'CrH',
         'CF4O', 'Al2Cl6', 'AlClF2'],
        ['1H2-16O', '48Ti-16O', '51V-16O', 'Na', 'K', '12C-16O', '52Cr-1H',
         '12C-19F4-16O', '27Al2-35Cl6', '27Al-35Cl-19F2'])
),)
def test_chemical_names_manipulation_2(species_name, iso_name):
    # Test conversion of common species name to a isotopologue name which can
    # be used as a key in the opacities dictionary
    assert species_name_to_common_isotopologue_name(species_name) == iso_name


@pytest.mark.parametrize("iso_name", (
    ['1H2-16O', '48Ti-16O', '51V-16O', 'Na', 'K', '12C-16O', '52Cr-1H',
     '12C-19F4-16O', '27Al2-35Cl6', '27Al-35Cl-19F2']
),)
def test_chemical_names_manipulation_3(iso_name):
    # Test round-trip conversion of isotopologue name, to species name, and back
    assert species_name_to_common_isotopologue_name(isotopologue_to_species(iso_name)) == iso_name
