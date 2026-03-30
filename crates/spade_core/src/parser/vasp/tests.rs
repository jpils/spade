use crate::types::Vectors;

use super::*;
use tempfile::NamedTempFile;
use std::{io::{ Seek, Write }, usize};

#[test]
fn test_read_config_header() {
    let poscar = create_poscar();
    let poscar = File::open(poscar.path()).unwrap();
    let mut lines = BufReader::new(poscar).lines();

    let _comment_line = lines.next().unwrap();
    let header = read_config_header(&mut lines).unwrap();

    let control_n_atoms_per_element: Vec<(Element, usize)> = vec![
        (Element::Sr, 1), 
        (Element::Ti, 1), 
        (Element::O, 3)
    ];

    let control_scaling_factor = 1.0;
    let contol_atom_count: usize = 5;
    let control_cell_matrix = Matrix::new(3.9, 0.0, 0.0, 0.0, 3.9, 0.0, 0.0, 0.0, 3.9);
    let control_coordinate_type = Coordinates::Direct;

    assert_eq!(header.n_atoms_per_element, control_n_atoms_per_element);
    assert_eq!(header.atom_count, contol_atom_count);
    assert_eq!(header._scaling_factor, control_scaling_factor);
    assert_eq!(header.cell_matrix, control_cell_matrix);
    assert_eq!(header.coordinate_type, control_coordinate_type);
}

#[test]
fn test_read_positions() {
    let poscar = create_poscar();
    let poscar = File::open(poscar.path()).unwrap();
    let mut lines = BufReader::new(poscar).lines();

    for _ in 0..8 {  // skip header
        lines.next();
    }

    let positions = read_positions(5, &mut lines).unwrap();

    let control_positions: Vectors = vec![
        Vector::new(0.0, 0.0, 0.0),
        Vector::new(0.5, 0.5, 0.5),
        Vector::new(0.5, 0.5, 0.0),
        Vector::new(0.5, 0.0, 0.5),
        Vector::new(0.0, 0.5, 0.5),
    ];

    assert_eq!(positions, control_positions);
}

#[test]
fn test_read_poscar() {
    let poscar = create_poscar();
    let full_filepath = poscar.path();
    let res = read_poscar(full_filepath).unwrap();

    let positions = vec![
            Vector::new(0.0, 0.0, 0.0),
            Vector::new(0.5, 0.5, 0.5),
            Vector::new(0.5, 0.5, 0.0),
            Vector::new(0.5, 0.0, 0.5),
            Vector::new(0.0, 0.5, 0.5)
    ];

    let element_count: Vec<(Element, usize)> = vec![
        (Element::Sr, 1), 
        (Element::Ti, 1), 
        (Element::O, 3)
    ];

    let cell_matrix = Matrix::new(3.9, 0.0, 0.0, 0.0, 3.9, 0.0, 0.0, 0.0, 3.9);

    let control_res = ReadResult { 
        positions,
        element_count,
        cell_matrix,
        coordinate_type: Coordinates::Direct
    };

    let control_res_variant = ReadResultVariant::ReadResult { res: control_res };

    assert_eq!(res, control_res_variant);
}

#[test]
fn test_read_xdatcar() {
    let xdatcar = create_xdatcar();
    let full_filepath = xdatcar.path();
    let res = read_xdatcar(full_filepath).unwrap();

    let positions = vec![
            Vector::new(0.0, 0.0, 0.0),
            Vector::new(0.5, 0.5, 0.5),
            Vector::new(0.5, 0.5, 0.0),
            Vector::new(0.5, 0.0, 0.5),
            Vector::new(0.0, 0.5, 0.5)
    ];

    let element_count: Vec<(Element, usize)> = vec![
        (Element::Sr, 1), 
        (Element::Ti, 1), 
        (Element::O, 3)
    ];

    let cell_matrix = Matrix::new(3.9, 0.0, 0.0, 0.0, 3.9, 0.0, 0.0, 0.0, 3.9);

    let control_res = ReadResult { 
        positions,
        element_count,
        cell_matrix,
        coordinate_type: Coordinates::Direct
    };

    let control_read_results: Vec<ReadResult> = std::iter::repeat(control_res)
        .take(3)
        .collect();

    let control_res_variant = ReadResultVariant::ReadResults { res: control_read_results };

    assert_eq!(res, control_res_variant)
}

fn create_poscar() -> NamedTempFile {
    let mut temp_poscar = NamedTempFile::new().unwrap();

    writeln!(temp_poscar, "Temp Poscar").unwrap();

    writeln!(temp_poscar, "1.0").unwrap();
    writeln!(temp_poscar, "3.9 0.0 0.0").unwrap();
    writeln!(temp_poscar, "0.0 3.9 0.0").unwrap();
    writeln!(temp_poscar, "0.0 0.0 3.9").unwrap();

    writeln!(temp_poscar, "Sr Ti O").unwrap();
    writeln!(temp_poscar, "1 1 3").unwrap();
    writeln!(temp_poscar, "Direct").unwrap(); // TODO add test for poscar with selective
                                              // dynamics enabled
    writeln!(temp_poscar, "0.000000000  0.000000000  0.000000000").unwrap(); 
    writeln!(temp_poscar, "0.500000000  0.500000000  0.500000000").unwrap();
    writeln!(temp_poscar, "0.500000000  0.500000000  0.000000000").unwrap();
    writeln!(temp_poscar, "0.500000000  0.000000000  0.500000000").unwrap();
    writeln!(temp_poscar, "0.000000000  0.500000000  0.500000000").unwrap();

    temp_poscar.seek(std::io::SeekFrom::Start(0)).unwrap(); // file pointer to the beginning
    temp_poscar
}
 
fn create_xdatcar() -> NamedTempFile {
    let mut temp_xdatcar = NamedTempFile::new().unwrap();

    writeln!(temp_xdatcar, "Temp Poscar").unwrap();

    writeln!(temp_xdatcar, "1.0").unwrap();
    writeln!(temp_xdatcar, "3.9 0.0 0.0").unwrap();
    writeln!(temp_xdatcar, "0.0 3.9 0.0").unwrap();
    writeln!(temp_xdatcar, "0.0 0.0 3.9").unwrap();

    writeln!(temp_xdatcar, "Sr Ti O").unwrap();
    writeln!(temp_xdatcar, "1 1 3").unwrap();

    writeln!(temp_xdatcar, "Direct").unwrap(); 
    writeln!(temp_xdatcar, "0.000000000  0.000000000  0.000000000").unwrap(); 
    writeln!(temp_xdatcar, "0.500000000  0.500000000  0.500000000").unwrap();
    writeln!(temp_xdatcar, "0.500000000  0.500000000  0.000000000").unwrap();
    writeln!(temp_xdatcar, "0.500000000  0.000000000  0.500000000").unwrap();
    writeln!(temp_xdatcar, "0.000000000  0.500000000  0.500000000").unwrap();

    writeln!(temp_xdatcar, "Direct").unwrap(); 
    writeln!(temp_xdatcar, "0.000000000  0.000000000  0.000000000").unwrap(); 
    writeln!(temp_xdatcar, "0.500000000  0.500000000  0.500000000").unwrap();
    writeln!(temp_xdatcar, "0.500000000  0.500000000  0.000000000").unwrap();
    writeln!(temp_xdatcar, "0.500000000  0.000000000  0.500000000").unwrap();
    writeln!(temp_xdatcar, "0.000000000  0.500000000  0.500000000").unwrap();

    writeln!(temp_xdatcar, "Direct").unwrap(); 
    writeln!(temp_xdatcar, "0.000000000  0.000000000  0.000000000").unwrap(); 
    writeln!(temp_xdatcar, "0.500000000  0.500000000  0.500000000").unwrap();
    writeln!(temp_xdatcar, "0.500000000  0.500000000  0.000000000").unwrap();
    writeln!(temp_xdatcar, "0.500000000  0.000000000  0.500000000").unwrap();
    writeln!(temp_xdatcar, "0.000000000  0.500000000  0.500000000").unwrap();

    temp_xdatcar.seek(std::io::SeekFrom::Start(0)).unwrap();
    temp_xdatcar
}
