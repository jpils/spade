use crate::prelude::*;
use crate::types::{Atom, Coordinates, Element, Matrix, ReadResult, ReadResultVariant, Vector, Vectors, Elements};
use core::f64;
use std::fs::File;
use std::io::{Lines, BufRead, BufReader};
use std::path::{Path};

pub(crate) fn read_poscar<P: AsRef<Path>>(filepath: P) -> Result<ReadResultVariant> {
    let filepath = filepath.as_ref();

    let file = File::open(filepath)?;
    let mut lines = BufReader::new(file).lines();
    
    let _comment_line = lines
        .next()
        .transpose()?
        .ok_or_else(|| Error::Format("File is empty".into()))?;

    let header = read_config_header(&mut lines)?;
    let positions = read_positions(header.atom_count, &mut lines)?;
    let res = ReadResult { 
        positions, 
        element_count: header.n_atoms_per_element,
        cell_matrix: header.cell_matrix, 
        coordinate_type: header.coordinate_type 
    };

    Ok(ReadResultVariant::ReadResult { res })
}

pub(crate) fn read_xdatcar<P: AsRef<Path>>(filepath: P) -> Result<ReadResultVariant> {
    let filepath = filepath.as_ref();

    let file = File::open(filepath)?;
    let mut lines = BufReader::new(file).lines();

    let _comment_line = lines
        .next()
        .transpose()?
        .ok_or_else(|| Error::Format("File is empty".into()))?;

    let mut res: Vec<ReadResult> = Vec::new();
    let header = read_config_header(&mut lines)?;
    loop {
        let positions = read_positions(header.atom_count, &mut lines)?;
        res.push(
            ReadResult { 
                positions, 
                element_count: header.n_atoms_per_element.clone(),
                cell_matrix: header.cell_matrix, 
                coordinate_type: header.coordinate_type 
        });

        if lines.next().transpose()?.is_none() { // skip config number and check if EOF
            break;
        }
    }

    Ok(ReadResultVariant::ReadResults { res })
}

#[derive(Debug, Clone)]
struct ConfigHeader {
    n_atoms_per_element: Vec<(Element, usize)>,
    atom_count: usize,
    cell_matrix: Matrix,
    coordinate_type: Coordinates,
    _scaling_factor: f64,
}

fn read_config_header(lines: &mut Lines<BufReader<File>>) -> Result<ConfigHeader> {
    let scaling_factor = lines
        .next()
        .transpose()?
        .ok_or_else(|| Error::Format("Failed to read scaling factor".into()))?
        .parse::<f64>()?;

    let mut cell_vectors: Vec<Vector> = Vec::new();
    for _ in 0..3 {
        let lattice_vec: Vec<f64> = lines
            .next()
            .transpose()?
            .ok_or_else(|| Error::Format("Failed to read lattice vector".into()))?
            .split_whitespace()
            .map(|s| s.parse::<f64>().map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;

        cell_vectors.push(Vector::from_iterator(lattice_vec));
    }

    let elements: Elements = lines
        .next()
        .transpose()?
        .ok_or_else(|| Error::Format("Failed to read elements".into()))?
        .split_whitespace()
        .map(|s| s.parse::<Element>().map_err(Into::into))
        .collect::<Result<Vec<_>>>()?;

    let element_count: Vec<usize> = lines
        .next()
        .transpose()?
        .ok_or_else(|| Error::Format("Failed to read atom count per element".into()))?
        .split_whitespace()
        .map(|s| s.parse::<usize>().map_err(Into::into))
        .collect::<Result<Vec<_>>>()?;

    let coordinate_type = lines
        .next()
        .transpose()?
        .ok_or_else(|| Error::Format("Failed to read coordinate_type".into()))?
        .trim()
        .to_string();

    if coordinate_type.to_lowercase().starts_with('s') {
        todo!("Parsing POSCAR/XDATCAR with selective dynamics, not yet supported")
    }

    let cell_vectors_t = cell_vectors.iter().map(nalgebra::Matrix::transpose).collect::<Vec<_>>();
    let cell_matrix = Matrix::from_rows(&cell_vectors_t);
    let coordinate_type: Coordinates = coordinate_type.parse()?;

    let n_atoms_per_element: Vec<(Element, usize)> = elements
       .into_iter()
       .zip(element_count)
       .collect();

    let atom_count = n_atoms_per_element
        .iter()
        .map(|(_, count)| count)
        .sum();
    
    Ok(ConfigHeader { 
        n_atoms_per_element, 
        atom_count,
        cell_matrix, 
        coordinate_type, 
        _scaling_factor: scaling_factor 
    })
}

fn read_positions(n_atoms: usize, lines: &mut Lines<BufReader<File>>) -> Result<Vectors> {
    let mut positions: Vectors = Vectors::new();

    for _ in 0..n_atoms {
        let vec: Vec<f64> = lines
            .next()
            .transpose()?
            .ok_or_else(|| Error::Format("Failed to read ion position".into()))?
            .split_whitespace()
            .map(|s| s.parse::<f64>().map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;

        positions.push(Vector::from_vec(vec));
    }

    Ok(positions)
}

#[cfg(test)]
mod tests;
