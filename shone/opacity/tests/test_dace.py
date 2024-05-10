from shone.opacity.dace import (
    create_nc_path_molecule, parse_nc_path_molecule,
    create_nc_path_atom, parse_nc_path_atom
)


def test_molecule_filename_roundtrip():
    isotopologue = '1H2-16O1'
    line_list = 'LineList'
    temperature_range = [50, 5000]
    pressure_range = [-6.0, 0.0]
    version = 2

    input_params = [
        isotopologue, line_list,
        temperature_range,
        pressure_range, version
    ]
    path = create_nc_path_molecule(*input_params)
    recovered_params = parse_nc_path_molecule(path)
    assert input_params == list(recovered_params)


def test_atom_filename_roundtrip():
    atom = 'Na'
    charge = 1
    line_list = 'LineList'
    temperature_range = [50, 5000]
    pressure_range = [-6.0, 0.0]
    version = 2

    input_params = [
        atom, charge, line_list,
        temperature_range,
        pressure_range, version
    ]
    path = create_nc_path_atom(*input_params)
    recovered_params = parse_nc_path_atom(path)
    assert input_params == list(recovered_params)
