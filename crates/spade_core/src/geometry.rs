use std::f64::consts::PI;

use crate::prelude::*;
use crate::types::{Atom, Atoms, Matrix, UVector, Vector, Vectors};

use nalgebra::{Rotation3};

pub fn get_displacements_between(lhs: &Atoms, rhs: &Atoms) -> Result<Vectors> {
    if lhs.len() != rhs.len() {
        return Err(Error::SizeMismatch { expected: lhs.len(), got: rhs.len() });
    }

    let mut displacements: Vectors = Vec::with_capacity(lhs.len());
    for (atom_lhs, atom_rhs) in lhs.iter().zip(rhs) {
        displacements.push(atom_lhs.position - atom_rhs.position);
    }

    Ok(displacements)
}

pub fn rotate_x_site_atoms(
    x_site_atoms: &mut Atoms, 
    tilt_axis_angle: (UVector, f64 /* rad */)
) {

    let tilt_axis = tilt_axis_angle.0;
    let tilt_angle = tilt_axis_angle.1;

    for atom in x_site_atoms.iter_mut() {
        rotate_atom(atom, &tilt_axis, tilt_angle);
    }
}

pub fn rotate_atom(atom: &mut Atom, axis: &UVector, tilt_angle: f64) {
    let rot = Rotation3::from_axis_angle(axis, tilt_angle);
    rot * atom.position;
}

#[must_use]
pub fn compute_metric(cell_matrix: &Matrix) -> Matrix {
    cell_matrix.transpose() * cell_matrix
}

#[must_use]
pub fn compute_x_angle(vec: &Vector) -> f64 {
    let mut angle = vec[2].atan2(vec[0]);
    if angle < 0.0 {
        angle += 2.0*PI;
    }
    angle
}

/// minimum image convention, vec in direct coordinates
#[must_use]
pub fn wrap_vector(vec: &Vector) -> Vector {
    vec - (vec.add_scalar(f64::floor(0.5 - f64::EPSILON)))
}

#[must_use]
pub fn direct_to_cartesian(vec: Vector, cell_matrix: &Matrix) -> Vector {
    cell_matrix * vec
}

#[cfg(test)]
mod tests {
    use super::*;
}
