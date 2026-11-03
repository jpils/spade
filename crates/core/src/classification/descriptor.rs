//! providing a method to calculate descriptors to classify domains and domain boundaries
use crate::prelude::*;
use crate::{error::Error, geometry::{self, NearestNeighbors}, parser::Parser, types::{AsVector, Atom, Atoms, CartAtom, Cartesian, Matrix, SupercellAtoms, Vector, Vectors}};
use nalgebra::{SVector};

pub type Descriptor = SVector<f64, 18>;
pub type Descriptors = Vec<Descriptor>;

#[derive(Clone, Copy, Debug)]
pub struct BbDescriptor {
    pub descriptor: Descriptor,
    pub reference_b_atom: CartAtom
}

pub type BbDescriptors = Vec<BbDescriptor>;

pub fn calculate_bb_descriptors(mut b_b: Vec<NearestNeighbors>, b_x: &[NearestNeighbors]) -> Result<BbDescriptors> {
    let mut descriptors: BbDescriptors = Vec::with_capacity(b_b.len());
    while let Some(b) = b_b.pop() {
        let current = b.reference;
        let neighbor = b.neighbors[0].atom_abs;

        let current_x = b_x.iter().find(|nn| nn.reference == current).unwrap();
        let neighbor_x = b_x.iter().find(|nn| nn.reference == neighbor).unwrap();

        let current_desc = get_b_site_descriptor(current_x)?;
        let neighbor_desc = get_b_site_descriptor(neighbor_x)?;
        let avg_desc = (current_desc + neighbor_desc)/2.0;

        let bb_descriptor = BbDescriptor { descriptor: avg_desc, reference_b_atom: current };
        descriptors.push(bb_descriptor);
    }

    Ok(descriptors)
}

fn get_b_site_descriptor(b_nns: &NearestNeighbors) -> Result<Descriptor> {
    let vectors = b_nns.neighbors.iter()
        .map(|neighbor| neighbor.atom_rel.position)
        .collect::<Vec<_>>();
    let sorted_vectors = sort_octahedral_vectors(&vectors)?;
    
    let mut arr = [0.0; 18];
    for (vec_id, vec) in sorted_vectors.into_iter().enumerate() {
        let arr_id = vec_id * 3;
        arr[arr_id] = vec.as_vector().x;
        arr[arr_id + 1] = vec.as_vector().y;
        arr[arr_id + 2] = vec.as_vector().z;
    }

    Ok(Descriptor::from(arr))
}

fn sort_octahedral_vectors(positions: &Vec<Cartesian>) -> Result<Vec<Cartesian>> {
    let (y_top_id, &y_top) = positions.iter()
        .enumerate()
        .max_by(|a, b| a.1.as_vector().y.total_cmp(&b.1.as_vector().y))
        .ok_or_else(|| Error::Generic("Comparison of octahedral positions failed!".into()))?;
    let mut positions = positions.clone();
    positions.swap_remove(y_top_id);

    let (y_bottom_id, &y_bottom) = positions.iter()
        .enumerate()
        .min_by(|a, b| a.1.as_vector().y.total_cmp(&b.1.as_vector().y))
        .ok_or_else(|| Error::Generic("Comparison of octahedral positions failed!".into()))?;
    positions.swap_remove(y_bottom_id);

    let mut pos_angle = positions.into_iter()
        .map(|pos| {
           let angle = geometry::compute_x_angle(&pos);
           (pos, angle)
        })
        .collect::<Vec<_>>();

    pos_angle.sort_by(|a, b| a.1.total_cmp(&b.1));
    
    let mut pos = pos_angle.into_iter()
        .map(|(pos, _)| pos)
        .collect::<Vec<_>>();

    pos.insert(0, y_top);
    pos.insert(1, y_bottom);
    Ok(pos)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Direct, DirectAtom, DirectAtoms, Element};

    const EPS: f64 = 1e-12;

    fn assert_descriptor_close(actual: &Descriptor, expected: &Descriptor) {
        assert!(
            (actual - expected).norm() <= EPS,
            "Expected {expected:?}, got {actual:?}"
        );
    }

    fn assert_vector_close(actual: &Vector, expected: &Vector) {
        assert!(
            (actual - expected).norm() <= EPS,
            "Expected {expected:?}, got {actual:?}"
        );
    }

    #[test]
    fn test_sort_octahedral_vectors_ordering() {
        let positions = vec![
            Vector::new(0.0, 0.0, 0.5),
            Vector::new(0.0, -0.5, 0.0),
            Vector::new(0.5, 0.0, 0.0),
            Vector::new(0.0, 0.5, 0.0),
            Vector::new(0.0, 0.0, -0.5),
            Vector::new(-0.5, 0.0, 0.0),
        ];

        let positions: Vec<Cartesian> = positions.into_iter().map(Cartesian::new).collect();
        let original = positions.clone();
        let sorted = sort_octahedral_vectors(&positions).unwrap();

        let expected = [
            Vector::new(0.0, 0.5, 0.0),
            Vector::new(0.0, -0.5, 0.0),
            Vector::new(0.5, 0.0, 0.0),
            Vector::new(0.0, 0.0, 0.5),
            Vector::new(-0.5, 0.0, 0.0),
            Vector::new(0.0, 0.0, -0.5),
        ];

        assert_eq!(sorted.len(), 6);
        assert_eq!(positions, original);

        for (actual, expected) in sorted.iter().zip(expected.iter()) {
            assert_vector_close(actual.as_vector(), expected);
        }
    }

    #[test]
    fn test_sort_octahedral_vectors_independent_of_input_order() {
        let positions = vec![
            Vector::new(0.0, 0.5, 0.0),
            Vector::new(0.0, -0.5, 0.0),
            Vector::new(0.5, 0.0, 0.0),
            Vector::new(-0.5, 0.0, 0.0),
            Vector::new(0.0, 0.0, 0.5),
            Vector::new(0.0, 0.0, -0.5),
        ];

        let positions: Vec<Cartesian> = positions.into_iter().map(Cartesian::new).collect();
        let expected = sort_octahedral_vectors(&positions).unwrap();

        for shift in 0..positions.len() {
            let mut rotated = positions.clone();
            rotated.rotate_left(shift);

            let actual = sort_octahedral_vectors(&rotated).unwrap();

            for (lhs, rhs) in actual.iter().zip(&expected) {
                assert_vector_close(lhs.as_vector(), rhs.as_vector());
            }
        }
    }

    #[test]
    fn test_get_b_site_descriptor() {
        let nn = create_mock_octahedron(
            Vector::new(0.25, 0.5, 0.5),
        );

        let descriptor = get_b_site_descriptor(&nn).unwrap();

        let expected = Descriptor::from([
            0.0, 0.5, 0.0,
            0.0, -0.5, 0.0,
            0.5, 0.0, 0.0,
            0.0, 0.0, 0.5,
            -0.5, 0.0, 0.0,
            0.0, 0.0, -0.5,
        ]);

        assert_eq!(descriptor.len(), 18);
        assert_descriptor_close(&descriptor, &expected);
    }

    #[test]
    fn test_get_b_site_descriptor_independent_of_neighbor_order() {
        let center = Vector::new(0.25, 0.5, 0.5);
        let b_atoms = vec![create_mock_atom(Element::Ti, center)];

        let mut x_atoms = create_mock_oxygen_shell(
            center,
            Vector::zeros(),
        );

        let original = geometry::n_nearest_neighbors(
            &b_atoms,
            &x_atoms,
            6,
            &test_cell(),
        ).unwrap();

        x_atoms.reverse();

        let reversed = geometry::n_nearest_neighbors(
            &b_atoms,
            &x_atoms,
            6,
            &test_cell(),
        ).unwrap();

        let expected = get_b_site_descriptor(&original[0]).unwrap();
        let actual = get_b_site_descriptor(&reversed[0]).unwrap();

        assert_descriptor_close(&actual, &expected);
    }

    #[test]
    fn test_calculate_bb_link_descriptors_averaging() {
        let b1_position = Vector::new(0.2, 0.5, 0.5);
        let b2_position = Vector::new(0.7, 0.5, 0.5);

        let b_atoms = vec![
            create_mock_atom(Element::Ti, b1_position),
            create_mock_atom(Element::Ti, b2_position),
        ];

        let mut x_atoms = create_mock_oxygen_shell(
            b1_position,
            Vector::zeros(),
        );

        x_atoms.extend(create_mock_oxygen_shell(
            b2_position,
            Vector::new(0.0, 0.01, 0.0),
        ));

        let cell = test_cell();

        let b_b = geometry::n_nearest_neighbors(
            &b_atoms,
            &b_atoms,
            1,
            &cell,
        ).unwrap();

        let b_x = geometry::n_nearest_neighbors(
            &b_atoms,
            &x_atoms,
            6,
            &cell,
        ).unwrap();

        let descriptors = calculate_bb_descriptors(b_b, &b_x).unwrap();

        assert_eq!(descriptors.len(), 2);

        let expected = Descriptor::from([
            0.0, 0.45, 0.0,
            0.0, -0.55, 0.0,
            0.5, -0.05, 0.0,
            0.0, -0.05, 0.5,
            -0.5, -0.05, 0.0,
            0.0, -0.05, -0.5,
        ]);

        for result in &descriptors {
            assert_descriptor_close(&result.descriptor, &expected);
        }

        let b1_cart = geometry::direct_to_cartesian(
            Direct::new(b1_position).unwrap(),
            &cell,
        );

        let b2_cart = geometry::direct_to_cartesian(
            Direct::new(b2_position).unwrap(),
            &cell,
        );

        assert!(descriptors.iter().any(|desc| {
            (desc.reference_b_atom.position - b1_cart).as_vector().norm() <= EPS
        }));

        assert!(descriptors.iter().any(|desc| {
            (desc.reference_b_atom.position - b2_cart).as_vector().norm() <= EPS
        }));
    }

    #[test]
    fn test_calculate_bb_descriptors_empty_input() {
        let b_b: Vec<NearestNeighbors> = Vec::new();
        let b_x: Vec<NearestNeighbors> = Vec::new();

        let result = calculate_bb_descriptors(b_b, &b_x).unwrap();

        assert!(result.is_empty());
    }

    #[test]
    #[should_panic]
    fn test_get_b_site_descriptor_rejects_empty_neighbors() {
        let b_atoms = vec![
            create_mock_atom(
                Element::Ti,
                Vector::new(0.25, 0.5, 0.5),
            ),
        ];

        let x_atoms: DirectAtoms = Vec::new();

        let nn = geometry::n_nearest_neighbors(
            &b_atoms,
            &x_atoms,
            0,
            &test_cell(),
        ).unwrap();

        get_b_site_descriptor(&nn[0]).unwrap();
    }

    fn create_mock_atom(element: Element, position: Vector) -> DirectAtom {
        Atom {
            atom_type: Some(element),
            position: Direct::new(position).unwrap(),
        }
    }

    fn create_mock_oxygen_shell(
        center: Vector,
        distortion: Vector,
    ) -> DirectAtoms {
        let offsets = [
            Vector::new(0.0, 0.05, 0.0),
            Vector::new(0.0, -0.05, 0.0),
            Vector::new(0.05, 0.0, 0.0),
            Vector::new(0.0, 0.0, 0.05),
            Vector::new(-0.05, 0.0, 0.0),
            Vector::new(0.0, 0.0, -0.05),
        ];

        offsets
            .into_iter()
            .map(|offset| {
                create_mock_atom(
                    Element::O,
                    center + offset + distortion,
                )
            })
            .collect()
    }

    fn create_mock_octahedron(
        reference_position: Vector,
    ) -> NearestNeighbors {
        let b_atoms = vec![
            create_mock_atom(Element::Ti, reference_position),
        ];

        let x_atoms = create_mock_oxygen_shell(
            reference_position,
            Vector::zeros(),
        );

        let mut nn = geometry::n_nearest_neighbors(
            &b_atoms,
            &x_atoms,
            6,
            &test_cell(),
        ).unwrap();

        nn.remove(0)
    }

    fn test_cell() -> Matrix {
        Matrix::from_diagonal(
            &Vector::new(10.0, 10.0, 10.0),
        )
    }
}
