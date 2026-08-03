use std::f64::consts::PI;

use crate::prelude::*;
use crate::types::{Atom, Atoms, Matrix, UVector, Vector, Vectors};

use nalgebra::{Rotation3, SVector};

#[derive(Debug, Clone)]
pub(crate) struct NearestNeighbors {
    pub(crate) reference: Atom,
    pub(crate) neighbors: Vec<Neighbor>
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Neighbor {
    pub(crate) atom_abs_cart: Atom,
    pub(crate) atom_rel_cart: Atom,
    pub(crate) dist: f64
}

/// find n nearest neighbors for atom a with positions relative to a
pub(crate) fn n_nearest_neighbors(a: &Atoms, b: &Atoms, n: usize, cell_matrix: &Matrix) -> Vec<NearestNeighbors> {
    let mut distances: Vec<NearestNeighbors> = Vec::new();
    let is_self = *a == *b;
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
        atom_distances.sort_by(|(_, _, a), (_, _, b)| a.total_cmp(b));
        
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

        let n_nearest: Vec<Neighbor> = n_nearest.iter()
            .map(|(atom_abs, atom_rel, dist)| 
                Neighbor { 
                    atom_abs_cart: *atom_abs, 
                    atom_rel_cart: *atom_rel, 
                    dist: *dist
                }
            )
            .collect();
        let nn = NearestNeighbors {
            reference: atom_a,
            neighbors: n_nearest
        };
        distances.push(nn);
    }
    distances
}

pub(crate) fn get_displacements_between(lhs: &Atoms, rhs: &Atoms) -> Result<Vectors> {
    if lhs.len() != rhs.len() {
        return Err(Error::SizeMismatch { expected: lhs.len(), got: rhs.len() });
    }

    let mut displacements: Vectors = Vec::with_capacity(lhs.len());
    for (atom_lhs, atom_rhs) in lhs.iter().zip(rhs) {
        displacements.push(atom_lhs.position - atom_rhs.position);
    }

    Ok(displacements)
}

pub(crate) fn rotate_x_site_atoms(
    x_site_atoms: &mut Atoms, 
    tilt_axis_angle: (UVector, f64 /* rad */)
) {

    let tilt_axis = tilt_axis_angle.0;
    let tilt_angle = tilt_axis_angle.1;

    for atom in x_site_atoms.iter_mut() {
        rotate_atom(atom, &tilt_axis, tilt_angle);
    }
}

pub(crate) fn rotate_atom(atom: &mut Atom, axis: &UVector, tilt_angle: f64) {
    let rot = Rotation3::from_axis_angle(axis, tilt_angle);
    atom.position = rot * atom.position;
}

#[must_use]
pub(crate) fn compute_metric(cell_matrix: &Matrix) -> Matrix {
    cell_matrix.transpose() * cell_matrix
}

#[must_use]
pub(crate) fn compute_x_angle(vec: &Vector) -> f64 {
    let mut angle = vec[2].atan2(vec[0]);
    if angle < 0.0 {
        angle += 2.0*PI;
    }
    angle
}

/// minimum image convention, vec in direct coordinates
#[must_use]
pub(crate) fn wrap_vector(vec: &Vector) -> Vector {
    vec - vec.map(|x| (x + 0.5).floor())
}

#[must_use]
pub(crate) fn direct_to_cartesian(vec: Vector, cell_matrix: &Matrix) -> Vector {
    cell_matrix * vec
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Element;

    const EPS: f64 = 1e-12;

    fn assert_close(actual: f64, expected: f64) {
        let tolerance = EPS * expected.abs().max(1.0);

        assert!(
            (actual - expected).abs() <= tolerance,
            "Expected {expected}, got {actual}"
        );
    }

    fn assert_vec_close(actual: &Vector, expected: &Vector) {
        let diff = actual - expected;
        let tolerance = EPS * expected.norm().max(1.0);

        assert!(
            diff.norm() <= tolerance,
            "Expected {expected:?}, got {actual:?}"
        );
    }

    #[test]
    fn test_n_nearest_neighbors() {
        let (atoms_a, atoms_b) = test_atoms();
        let cell = test_cell();

        let nn = n_nearest_neighbors(&atoms_a, &atoms_b, 3, &cell);

        let expected_distances = [
            [12.0, 12.0, 75.0],
            [27.0, 27.0, 75.0],
            [12.0, 27.0, 48.0],
            [12.0, 27.0, 48.0],
        ];

        assert_eq!(nn.len(), atoms_a.len());

        for (i, neighbor_data) in nn.iter().enumerate() {
            assert_eq!(neighbor_data.neighbors.len(), 3);

            let expected_position =
                direct_to_cartesian(atoms_a[i].position, &cell);

            assert_vec_close(
                &neighbor_data.reference.position,
                &expected_position,
            );

            for (j, neighbor) in neighbor_data.neighbors.iter().enumerate() {
                assert_close(neighbor.dist, expected_distances[i][j]);

                assert_close(
                    neighbor.dist,
                    neighbor.atom_rel_cart.position.norm_squared(),
                );
            }

            for pair in neighbor_data.neighbors.windows(2) {
                assert!(
                    pair[0].dist <= pair[1].dist,
                    "Neighbors are not sorted by distance"
                );
            }
        }
    }

    #[test]
    fn test_n_nearest_neighbors_without_self_exclusion() {
        let a = vec![
            Atom {
                atom_type: None,
                position: Vector::new(0.0, 0.0, 0.0),
            },
        ];

        let b = vec![
            Atom {
                atom_type: None,
                position: Vector::new(0.1, 0.0, 0.0),
            },
            Atom {
                atom_type: None,
                position: Vector::new(0.3, 0.0, 0.0),
            },
        ];

        let nn = n_nearest_neighbors(&a, &b, 5, &test_cell());

        assert_eq!(nn.len(), 1);
        assert_eq!(nn[0].neighbors.len(), 2);

        assert_close(nn[0].neighbors[0].dist, 1.0);
        assert_close(nn[0].neighbors[1].dist, 9.0);

        assert_vec_close(
            &nn[0].neighbors[0].atom_abs_cart.position,
            &Vector::new(1.0, 0.0, 0.0),
        );
    }

    #[test]
    fn test_n_nearest_neighbors_zero_requested() {
        let (a, b) = test_atoms();

        let nn = n_nearest_neighbors(&a, &b, 0, &test_cell());

        assert_eq!(nn.len(), a.len());

        for entry in nn {
            assert!(entry.neighbors.is_empty());
        }
    }

    #[test]
    fn test_n_nearest_neighbors_single_atom() {
        let atoms = vec![
            Atom {
                atom_type: None,
                position: Vector::new(0.0, 0.0, 0.0),
            },
        ];

        let nn = n_nearest_neighbors(&atoms, &atoms, 3, &test_cell());

        assert_eq!(nn.len(), 1);
        assert!(nn[0].neighbors.is_empty());
    }

    #[test]
    fn test_n_nearest_neighbors_empty_input() {
        let atoms: Atoms = Vec::new();

        let nn = n_nearest_neighbors(&atoms, &atoms, 3, &test_cell());

        assert!(nn.is_empty());
    }

    #[test]
    fn test_n_nearest_neighbors_periodic_boundary() {
        let a = vec![
            Atom {
                atom_type: None,
                position: Vector::new(0.95, 0.0, 0.0),
            },
        ];

        let b = vec![
            Atom {
                atom_type: None,
                position: Vector::new(0.05, 0.0, 0.0),
            },
        ];

        let nn = n_nearest_neighbors(&a, &b, 1, &test_cell());

        assert_eq!(nn.len(), 1);
        assert_eq!(nn[0].neighbors.len(), 1);

        let neighbor = &nn[0].neighbors[0];

        assert_close(neighbor.dist, 1.0);

        assert_vec_close(
            &neighbor.atom_rel_cart.position,
            &Vector::new(-1.0, 0.0, 0.0),
        );

        assert_vec_close(
            &neighbor.atom_abs_cart.position,
            &Vector::new(0.5, 0.0, 0.0),
        );
    }

    #[test]
    #[ignore = "Self exclusion assumes the first sorted neighbor is the reference atom"]
    fn test_n_nearest_neighbors_duplicate_positions() {
        let atoms = vec![
            Atom {
                atom_type: Some(Element::Ti),
                position: Vector::zeros(),
            },
            Atom {
                atom_type: None,
                position: Vector::zeros(),
            },
        ];

        let nn = n_nearest_neighbors(&atoms, &atoms, 1, &test_cell());

        assert_eq!(nn.len(), 2);

        for entry in nn {
            assert_eq!(entry.neighbors.len(), 1);

            assert_ne!(
                entry.reference.atom_type,
                entry.neighbors[0].atom_abs_cart.atom_type,
            );
        }
    }

    #[test]
    fn test_wrap_vector_minimum_image() {
        let v1 = Vector::new(0.2, -0.1, 0.4);
        assert_vec_close(&wrap_vector(&v1), &v1);

        let v2 = Vector::new(0.6, 0.0, 0.0);
        let expected_v2 = Vector::new(-0.4, 0.0, 0.0);
        assert_vec_close(&wrap_vector(&v2), &expected_v2);

        let v3 = Vector::new(0.0, -0.7, 0.0);
        let expected_v3 = Vector::new(0.0, 0.3, 0.0);
        assert_vec_close(&wrap_vector(&v3), &expected_v3);
    }

    #[test]
    fn test_wrap_vector_multiple_periods() {
        let v = Vector::new(2.6, -1.7, 0.5);
        let expected = Vector::new(-0.4, 0.3, -0.5);

        assert_vec_close(&wrap_vector(&v), &expected);
    }

    #[test]
    fn test_direct_to_cartesian() {
        let cell = test_cell();
        let direct_pos = Vector::new(0.5, 0.2, -0.1);
        let expected = Vector::new(5.0, 2.0, -1.0);

        let cartesian = direct_to_cartesian(direct_pos, &cell);

        assert_vec_close(&cartesian, &expected);
    }

    #[test]
    fn test_get_displacements_between() {
        let lhs = vec![
            Atom {
                atom_type: None,
                position: Vector::new(1.0, 2.0, 3.0),
            },
            Atom {
                atom_type: None,
                position: Vector::new(4.0, 5.0, 6.0),
            },
        ];

        let rhs = vec![
            Atom {
                atom_type: None,
                position: Vector::new(0.5, 1.0, 1.5),
            },
            Atom {
                atom_type: None,
                position: Vector::new(2.0, 2.5, 3.0),
            },
        ];

        let displacements =
            get_displacements_between(&lhs, &rhs).unwrap();

        assert_eq!(displacements.len(), 2);

        assert_vec_close(
            &displacements[0],
            &Vector::new(0.5, 1.0, 1.5),
        );

        assert_vec_close(
            &displacements[1],
            &Vector::new(2.0, 2.5, 3.0),
        );
    }

    #[test]
    fn test_get_displacements_size_mismatch() {
        let lhs = vec![
            Atom {
                atom_type: None,
                position: Vector::zeros(),
            },
        ];

        let rhs = vec![];

        assert!(matches!(
            get_displacements_between(&lhs, &rhs),
            Err(Error::SizeMismatch {
                expected: 1,
                got: 0
            })
        ));
    }

    #[test]
    fn test_get_displacements_empty() {
        let lhs: Atoms = Vec::new();
        let rhs: Atoms = Vec::new();

        let result = get_displacements_between(&lhs, &rhs).unwrap();

        assert!(result.is_empty());
    }

    #[test]
    fn test_rotate_atom() {
        let mut atom = Atom {
            atom_type: None,
            position: Vector::new(1.0, 0.0, 0.0),
        };

        let axis =
            UVector::new_normalize(Vector::new(0.0, 0.0, 1.0));

        rotate_atom(&mut atom, &axis, PI / 2.0);

        assert_vec_close(
            &atom.position,
            &Vector::new(0.0, 1.0, 0.0),
        );
    }

    #[test]
    fn test_rotate_atom_zero_angle() {
        let original = Vector::new(1.0, 2.0, 3.0);

        let mut atom = Atom {
            atom_type: None,
            position: original,
        };

        let axis =
            UVector::new_normalize(Vector::new(1.0, 0.0, 0.0));

        rotate_atom(&mut atom, &axis, 0.0);

        assert_vec_close(&atom.position, &original);
    }

    #[test]
    fn test_rotate_x_site_atoms() {
        let mut atoms = vec![
            Atom {
                atom_type: None,
                position: Vector::new(1.0, 0.0, 0.0),
            },
            Atom {
                atom_type: None,
                position: Vector::new(0.0, 1.0, 0.0),
            },
        ];

        let axis =
            UVector::new_normalize(Vector::new(0.0, 0.0, 1.0));

        rotate_x_site_atoms(&mut atoms, (axis, PI / 2.0));

        assert_vec_close(
            &atoms[0].position,
            &Vector::new(0.0, 1.0, 0.0),
        );

        assert_vec_close(
            &atoms[1].position,
            &Vector::new(-1.0, 0.0, 0.0),
        );
    }

    #[test]
    fn test_compute_metric() {
        let cell_matrix = Matrix::new(
            2.0, 1.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, 4.0,
        );

        let metric = compute_metric(&cell_matrix);

        let expected = Matrix::new(
            4.0, 2.0, 0.0,
            2.0, 10.0, 0.0,
            0.0, 0.0, 16.0,
        );

        assert!(
            (metric - expected).norm() <= EPS
        );
    }

    #[test]
    fn test_compute_x_angle() {
        let v1 = Vector::new(1.0, 0.0, 0.0);
        assert_close(compute_x_angle(&v1), 0.0);

        let v2 = Vector::new(0.0, 0.0, 1.0);
        assert_close(compute_x_angle(&v2), PI / 2.0);

        let v3 = Vector::new(1.0, 0.0, -1.0);
        assert_close(compute_x_angle(&v3), 7.0 * PI / 4.0);

        let v4 = Vector::new(-1.0, 0.0, 0.0);
        assert_close(compute_x_angle(&v4), PI);
    }

    fn test_cell() -> Matrix {
        Matrix::from_diagonal(&Vector::new(10.0, 10.0, 10.0))
    }

    fn test_atoms() -> (Atoms, Atoms) {
        let atoms = vec![
            Atom {
                atom_type: None,
                position: Vector::new(0.0, 0.0, 0.0),
            },
            Atom {
                atom_type: None,
                position: Vector::new(0.5, 0.5, 0.5),
            },
            Atom {
                atom_type: None,
                position: Vector::new(0.2, 0.2, 0.2),
            },
            Atom {
                atom_type: None,
                position: Vector::new(0.8, 0.8, 0.8),
            },
        ];

        (atoms.clone(), atoms)
    }
}
