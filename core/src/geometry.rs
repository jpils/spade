use std::f64::consts::PI;

use crate::prelude::*;
use crate::types::{Atom, Atoms, Matrix, UVector, Vector, Vectors};

use nalgebra::{Rotation3, SVector};

pub type NN = Vec<(Atom, Vec<(Atom, f64)>)>;

#[derive(Debug, Clone)]
pub struct NearestNeighbor {
    pub(crate) reference: Atom,
    pub(crate) neighbors: Vec<(Atom, Atom, f64)>
}

/// find n nearest neighbors for atom a with positions relative to a
pub(crate) fn n_nearest_neighbors(a: &Atoms, b: Atoms, n: usize, cell_matrix: &Matrix) -> Vec<NearestNeighbor> {
    let mut distances: Vec<NearestNeighbor> = Vec::new();
    let is_self = *a == b;
    for atom_a in a {
        let atom_distances = b
            .iter()
            .map(|&current| { 
                let d_unwrapped_dir = atom_a.position - current.position;
                let min_img = wrap_vector(&d_unwrapped_dir);
                let d_wrapped_cart = direct_to_cartesian(min_img, cell_matrix);
                let dist = d_wrapped_cart.norm_squared();
                let current_abs_cart = Atom {
                    atom_type: current.atom_type,
                    position: direct_to_cartesian(current.position, cell_matrix)
                };
                let current_rel_cart = Atom {
                    atom_type: current.atom_type,
                    position: d_wrapped_cart
                };
                (current_abs_cart, current_rel_cart, dist)
            })
            .collect::<Vec<_>>();

        let mut atom_distances = atom_distances;
        atom_distances.sort_by(|(_, _, a), (_, _, b)| a.total_cmp(&b));
        
        let n_nearest = if is_self {
            let len = atom_distances.len().saturating_sub(1);
            let n = n.min(len);
            &atom_distances[1..=n]
        } else {
            let len = atom_distances.len();
            let n = n.min(len);
            &atom_distances[..n]
        };
         
        let atom_a_cart_pos = direct_to_cartesian(atom_a.position, cell_matrix);
        let atom_a = Atom {
            atom_type: atom_a.atom_type,
            position: atom_a_cart_pos
        };
        let nn = NearestNeighbor {
            reference: atom_a,
            neighbors: n_nearest.into()
        };
        distances.push(nn);
    }
    distances
}

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
    atom.position = rot * atom.position;
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
    vec - vec.map(|x| (x + 0.5).floor())
}

#[must_use]
pub fn direct_to_cartesian(vec: Vector, cell_matrix: &Matrix) -> Vector {
    cell_matrix * vec
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Element;
    use nalgebra::{Matrix3, Vector3};

    #[test]
    fn test_n_nearest_neighbors() {
        let (atoms_a, atoms_b) = test_atoms();
        let cell = test_cell();
        let requested_neighbors = 3;
        
        let nn = n_nearest_neighbors(&atoms_a, atoms_b, requested_neighbors, &cell);
        
        assert_eq!(nn.len(), atoms_a.len());
        for neighbor_data in nn {
            assert_eq!(neighbor_data.neighbors.len(), requested_neighbors);
            
            // check that neighbors are correctly sorted by distance: ascending
            let d1 = neighbor_data.neighbors[0].2;
            let d2 = neighbor_data.neighbors[1].2;
            let d3 = neighbor_data.neighbors[2].2;
            
            assert!(d1 <= d2, "Neighbors are not sorted properly: d1 > d2");
            assert!(d2 <= d3, "Neighbors are not sorted properly: d2 > d3");
        }
    }

    #[test]
    fn test_wrap_vector_minimum_image() {
        // inside the unit cell: should remain unchanged
        let v1 = Vector::new(0.2, -0.1, 0.4);
        assert!((wrap_vector(&v1) - v1).norm() < 1e-12);

        // outside boundary (> 0.5): should wrap to negative image
        let v2 = Vector::new(0.6, 0.0, 0.0);
        let expected_v2 = Vector::new(-0.4, 0.0, 0.0);
        assert!((wrap_vector(&v2) - expected_v2).norm() < 1e-12);

        // outside boundary (< -0.5): should wrap to positive image
        let v3 = Vector::new(0.0, -0.7, 0.0);
        let expected_v3 = Vector::new(0.0, 0.3, 0.0);
        assert!((wrap_vector(&v3) - expected_v3).norm() < 1e-12);
    }

    #[test]
    fn test_direct_to_cartesian() {
        let cell = test_cell();
        let direct_pos = Vector::new(0.5, 0.2, -0.1);
        let expected_cart = Vector::new(5.0, 2.0, -1.0);
        
        let cartesian = direct_to_cartesian(direct_pos, &cell);
        assert!((cartesian - expected_cart).norm() < 1e-12);
    }

    #[test]
    fn test_get_displacements_between() {
        let lhs = vec![
            Atom { atom_type: None, position: Vector::new(1.0, 2.0, 3.0) },
            Atom { atom_type: None, position: Vector::new(4.0, 5.0, 6.0) },
        ];
        let rhs = vec![
            Atom { atom_type: None, position: Vector::new(0.5, 1.0, 1.5) },
            Atom { atom_type: None, position: Vector::new(2.0, 2.5, 3.0) },
        ];

        let displacements = get_displacements_between(&lhs, &rhs).unwrap();
        
        assert_eq!(displacements.len(), 2);
        assert!((displacements[0] - Vector::new(0.5, 1.0, 1.5)).norm() < 1e-12);
        assert!((displacements[1] - Vector::new(2.0, 2.5, 3.0)).norm() < 1e-12);
    }

    #[test]
    fn test_get_displacements_size_mismatch() {
        let lhs = vec![Atom { atom_type: None, position: Vector::zeros() }];
        let rhs = vec![];

        let result = get_displacements_between(&lhs, &rhs);
        assert!(result.is_err(), "Expected an error due to size mismatch");
    }

    #[test]
    fn test_rotate_atom() {
        let mut atom = Atom {
            atom_type: None,
            position: Vector::new(1.0, 0.0, 0.0),
        };
        let axis = UVector::new_normalize(Vector::new(0.0, 0.0, 1.0));
        let angle = PI / 2.0;

        rotate_atom(&mut atom, &axis, angle);

        let expected_position = Vector::new(0.0, 1.0, 0.0);
        assert!((atom.position - expected_position).norm() < 1e-12);
    }

    #[test]
    fn test_rotate_x_site_atoms() {
        let mut atoms = vec![
            Atom { atom_type: None, position: Vector::new(1.0, 0.0, 0.0) },
            Atom { atom_type: None, position: Vector::new(0.0, 1.0, 0.0) },
        ];
        let axis = UVector::new_normalize(Vector::new(0.0, 0.0, 1.0));
        let angle = PI / 2.0;

        rotate_x_site_atoms(&mut atoms, (axis, angle));

        assert!((atoms[0].position - Vector::new(0.0, 1.0, 0.0)).norm() < 1e-12);
        assert!((atoms[1].position - Vector::new(-1.0, 0.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_compute_metric() {
        let cell_matrix = Matrix::new(
            2.0, 1.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, 4.0
        );
        
        let metric = compute_metric(&cell_matrix);
        
        let expected_metric = Matrix::new(
            4.0, 2.0, 0.0,
            2.0, 10.0, 0.0,
            0.0, 0.0, 16.0
        );

        assert!((metric - expected_metric).norm() < 1e-12);
    }

    #[test]
    fn test_compute_x_angle() {
        let v1 = Vector::new(1.0, 0.0, 0.0);
        assert!((compute_x_angle(&v1) - 0.0).abs() < 1e-12);

        let v2 = Vector::new(0.0, 0.0, 1.0);
        assert!((compute_x_angle(&v2) - (PI / 2.0)).abs() < 1e-12);

        let v3 = Vector::new(1.0, 0.0, -1.0);
        assert!((compute_x_angle(&v3) - (7.0 * PI / 4.0)).abs() < 1e-12);
    }

    fn test_cell() -> Matrix {
        Matrix::from_diagonal(&Vector::new(10.0, 10.0, 10.0))
    }

    fn test_atoms() -> (Atoms, Atoms) {
        let a = vec![
            Atom { atom_type: None, position: Vector::new(0.0, 0.0, 0.0) },
            Atom { atom_type: None, position: Vector::new(0.5, 0.5, 0.5) },
            Atom { atom_type: None, position: Vector::new(0.2, 0.2, 0.2) },
            Atom { atom_type: None, position: Vector::new(0.8, 0.8, 0.8) },
        ];
        let b = a.clone();
        (a, b)
    }
}
