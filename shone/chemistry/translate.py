import re
import numpy as np
import astropy.units as u
from periodictable import elements


__all__ = [
    'isotopologue_to_species',
    'isotopologue_to_mass',
    'species_name_to_fastchem_name',
    'species_name_to_common_isotopologue_name',
]


def isotopologue_to_species(isotopologue):
    """
    Convert isotopologue name to common species name.

    Example: Take 1H2-16O and turn it to H2O, or take 48Ti-16O and turn it to TiO.

    Parameters
    ----------
    isotopologue : str
        Isotopologue name, like "1H2-16O".

    Returns
    -------
    common_name : str
        Common name, like "H2O".
    """
    species = ""
    for element in isotopologue.split('-'):
        for s in re.findall(r'\D+\d*', element):
            species += ''.join(s)
    return species if len(species) > 0 else isotopologue


def isotopologue_to_mass(isotopologue):
    """
    Find the total atomic mass for ``isotopologue``.

    Example: take 1H2-16O and turn it to 18, or take 48Ti-16O and turn it to 64.

    Parameters
    ----------
    isotopologue : str
        Isotopologue name, like "1H2-16O".

    Returns
    -------
    total_mass : astropy.units.Quantity
        Total atomic mass, like 18 AMU.
    """
    mass = 0
    for element in isotopologue.split('-'):
        multiples = list(filter(lambda x: len(x) > 0, re.split(r'\D', element)))
        if len(multiples) > 1:
            species_mass, multiplier = multiples
            mass += float(multiplier) * float(species_mass)
        elif len(multiples) == 1:
            mass += float(multiples[0])
    return (mass if mass != 0 else getattr(elements, isotopologue).mass) * u.u


def species_name_to_fastchem_name(species, return_mass=False):
    """
    Convert generic species name, like "H2O" or "ClAlF2", to
    Hill notation for FastChem, like "H2O1" or "Al1Cl1F2".

    Optionally, return the total mass of the species by summing the masses of its components.

    Parameters
    ----------
    species : str
        Generic name, like "H2O".

    Returns
    -------
    hill_name : str
        Name in Hill notation, like "H2O1".
    """
    atoms = np.array(list(filter(
        lambda x: len(x) > 0, re.split(r"(?<=[a-z])|(?=[A-Z])|\d", species)
    )))

    multipliers = np.array([
        int(x) if len(x) > 0 else 1 for x in re.split(r'\D', species)
    ])
    lens = [len(''.join(atom)) for atom in atoms]
    multipliers_skipped = np.array([multipliers[cs] for cs in np.cumsum(lens)])

    order = np.argsort(atoms)

    correct_notation = ''.join([
        a + str(m) for a, m in zip(atoms[order], multipliers_skipped[order])
    ])

    # If single atom, give only the name of the atom:
    if len(correct_notation) == 2 and correct_notation.endswith('1'):
        correct_notation = correct_notation[0]
    elif len(correct_notation) == 3 and correct_notation.endswith('1'):
        correct_notation = correct_notation[:2]

    if return_mass:
        # Optionally return mass of species
        mass = 0
        for atom, mult in zip(atoms, multipliers_skipped):
            mass += getattr(elements, atom).mass * mult

        return correct_notation, mass
    return correct_notation


def species_name_to_common_isotopologue_name(species):
    """
    Convert generic species name, like "H2O", to isotopologue name like "1H2-16O".

    Parameters
    ----------
    species : str
        Generic name, like "H2O".

    Returns
    -------
    isotopologue_name : str
        Isotopologue name, like "1H2-16O".
    """
    atoms = np.array(list(filter(
        lambda x: len(x) > 0, re.split(r"(?<=[a-z])|(?=[A-Z])|\d", species)
    )))

    multipliers = np.array([
        int(x) if len(x) > 0 else 1 for x in re.split(r'\D', species)
    ])
    lens = [len(''.join(atom)) for atom in atoms]
    multipliers_skipped = np.array([multipliers[cs] for cs in np.cumsum(lens)])

    masses = np.array([
        round(getattr(elements, atom).mass) for atom, mult in zip(atoms, multipliers_skipped)
    ])

    if len(atoms) > 1:
        correct_notation = '-'.join([
            str(mass) + a + (str(mult) if mult > 1 else '')
            for a, mult, mass in zip(atoms, multipliers_skipped, masses)
        ])

    # If single atom, give only the name of the atom:
    else:
        correct_notation = atoms[0]

    return correct_notation
