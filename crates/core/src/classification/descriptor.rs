//! providing a method to calculate descriptors to classify domains and domain boundaries
use crate::cell::simulation::SimCell;
use crate::observables::op::phase_factors;
use crate::prelude::*;
use crate::types::PhaseFactor;
use crate::{error::Error, geometry::{self, NearestNeighbors}, parser::Parser, types::{AsVector, Atom, Atoms, CartAtom, Cartesian, Matrix, SupercellAtoms, Vector, Vectors}};
use nalgebra::{SVector};

pub type Descriptor = SVector<f64, 2>;
pub type Descriptors = Vec<Descriptor>;

#[derive(Clone, Copy, Debug)]
pub struct BDescriptor {
    pub descriptor: Descriptor,
    pub reference_b_atom: CartAtom
}

#[derive(Clone, Debug)]
pub struct DescribedCell {
    pub b_descriptor: BDescriptor,
    pub cell: SimCell
}

pub type BDescriptors = Vec<BDescriptor>;

pub fn calculate_descriptors(
    b_b: &[NearestNeighbors],
    b_x: &[NearestNeighbors]
) -> Result<BDescriptors> {

    if b_b.is_empty() {
        return Ok(Vec::new());
    }

    let phase_factors = phase_factors(b_b)?;
    let mut descriptors: BDescriptors = Vec::with_capacity(b_b.len());

    for (b, phase_factor) in b_b.iter().zip(phase_factors) {
        let current = b.reference;

        let current_x = b_x.iter().find(|nn| nn.reference == current).unwrap();

        let current_desc = get_b_site_descriptor(current_x, phase_factor)?;

        let bb_descriptor = BDescriptor { descriptor: current_desc, reference_b_atom: current };
        descriptors.push(bb_descriptor);
    }

    Ok(descriptors)
}

fn get_b_site_descriptor(
    b_x_nns: &NearestNeighbors,
    phase_factor: PhaseFactor
) -> Result<Descriptor> {
    let vectors = b_x_nns.neighbors.iter()
        .map(|neighbor| neighbor.atom_rel.position)
        .collect::<Vec<_>>();
    let oy_pair = get_oy_pair(&vectors)?;

    let ry = (oy_pair.0 - oy_pair.1);
    let ry = ry.as_vector();

    let phase_factor = phase_factor.try_to_int()?;

    let dx = (ry.x + ry.z) * phase_factor as f64;
    let dy = (ry.x - ry.z) * phase_factor as f64;

    let desc = [dx, dy];

    Ok(Descriptor::from(desc))
}

fn get_oy_pair(positions: &Vec<Cartesian>) -> Result<(Cartesian, Cartesian)> {
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

    Ok((y_top, y_bottom))
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
    fn test_get_oy_pair_ordering() {
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
        let (top, bottom) = get_oy_pair(&positions).unwrap();

        assert_eq!(positions, original);
        assert_vector_close(top.as_vector(), &Vector::new(0.0, 0.5, 0.0));
        assert_vector_close(bottom.as_vector(), &Vector::new(0.0, -0.5, 0.0));
    }

    #[test]
    fn test_get_oy_pair_independent_of_input_order() {
        let positions = vec![
            Vector::new(0.0, 0.5, 0.0),
            Vector::new(0.0, -0.5, 0.0),
            Vector::new(0.5, 0.0, 0.0),
            Vector::new(-0.5, 0.0, 0.0),
            Vector::new(0.0, 0.0, 0.5),
            Vector::new(0.0, 0.0, -0.5),
        ];

        let positions: Vec<Cartesian> = positions.into_iter().map(Cartesian::new).collect();
        let expected = get_oy_pair(&positions).unwrap();

        for shift in 0..positions.len() {
            let mut rotated = positions.clone();
            rotated.rotate_left(shift);

            let actual = get_oy_pair(&rotated).unwrap();

            assert_vector_close(actual.0.as_vector(), expected.0.as_vector());
            assert_vector_close(actual.1.as_vector(), expected.1.as_vector());
        }
    }

    #[test]
    fn test_get_b_site_descriptor() {
        let nn = create_mock_octahedron(
            Vector::new(0.25, 0.5, 0.5),
        );

        let descriptor = get_b_site_descriptor(&nn, PhaseFactor::Positive).unwrap();

        let expected = Descriptor::from([0.0, 0.0]);

        assert_eq!(descriptor.len(), 2);
        assert_descriptor_close(&descriptor, &expected);
    }

    #[test]
    fn test_get_b_site_descriptor_tilted_oy_pair() {
        for (x, z, expected) in [
            (0.2, 0.1, [0.6, 0.2]),
            (0.1, 0.2, [0.6, -0.2]),
            (0.2, -0.1, [0.2, 0.6]),
        ] {
            let mut nn = create_mock_octahedron(Vector::new(0.25, 0.5, 0.5));
            for neighbor in &mut nn.neighbors {
                let position = *neighbor.atom_rel.position.as_vector();
                if position.y.abs() > 0.3 {
                    let sign = position.y.signum();
                    neighbor.atom_rel.position = Cartesian::new(
                        position + Vector::new(x * sign, 0.0, z * sign),
                    );
                }
            }

            let expected = Descriptor::from(expected);
            assert_descriptor_close(
                &get_b_site_descriptor(&nn, PhaseFactor::Positive).unwrap(), &expected,
            );
            assert_descriptor_close(
                &get_b_site_descriptor(&nn, PhaseFactor::Negative).unwrap(), &-expected,
            );
            assert!(get_b_site_descriptor(&nn, PhaseFactor::Undefined).is_err());
            nn.neighbors.reverse();
            assert_descriptor_close(
                &get_b_site_descriptor(&nn, PhaseFactor::Positive).unwrap(), &expected,
            );
        }
    }

    #[test]
    fn test_get_oy_pair_rejects_fewer_than_two_positions() {
        assert!(get_oy_pair(&vec![]).is_err());
        assert!(get_oy_pair(&vec![Cartesian::new(Vector::zeros())]).is_err());
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

        let expected = get_b_site_descriptor(&original[0], PhaseFactor::Positive).unwrap();
        let actual = get_b_site_descriptor(&reversed[0], PhaseFactor::Positive).unwrap();

        assert_descriptor_close(&actual, &expected);
    }

    #[test]
    fn test_calculate_bb_descriptors_applies_site_phase_factors() {
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

        let mut b_x = geometry::n_nearest_neighbors(
            &b_atoms,
            &x_atoms,
            6,
            &cell,
        ).unwrap();

        // Tilt the second site's O_y pair: r_y = (0.4, 1.0, 0.2).
        for neighbor in &mut b_x[1].neighbors {
            let position = *neighbor.atom_rel.position.as_vector();
            if position.y.abs() > 0.3 {
                let sign = position.y.signum();
                neighbor.atom_rel.position = Cartesian::new(
                    position + Vector::new(0.2 * sign, 0.0, 0.1 * sign),
                );
            }
        }

        let descriptors = calculate_descriptors(&b_b, &b_x).unwrap();

        assert_eq!(descriptors.len(), 2);

        // First site is untilted (+ phase); second has r_y = (0.4, 1.0, 0.2)
        // and negative phase, giving [-(0.4 + 0.2), -(0.4 - 0.2)].
        assert_descriptor_close(&descriptors[0].descriptor, &Descriptor::zeros());
        assert_descriptor_close(&descriptors[1].descriptor, &Descriptor::from([-0.6, -0.2]));

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

        let result = calculate_descriptors(&b_b, &b_x).unwrap();

        assert!(result.is_empty());
    }

    #[test]
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

        assert!(get_b_site_descriptor(&nn[0], PhaseFactor::Positive).is_err());
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
