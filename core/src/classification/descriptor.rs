use crate::{error::Error, geometry, parser::Parser, types::{Atom, Atoms, Matrix, ReadResultVariant, SupercellAtoms, Vector, Vectors}, geometry::NearestNeighbor};
use nalgebra::{SVector};

pub(crate) type Descriptor = SVector<f64, 18>;
pub(crate) type Descriptors = Vec<SVector<f64, 18>>;

pub(crate) fn calculate_descriptors(mut b_b: Vec<NearestNeighbor>, b_x: Vec<NearestNeighbor>) -> Vec<(Atom, Descriptor)> {
    let mut descriptors: Vec<(Atom, Descriptor)> = Vec::with_capacity(b_b.len());
    while let Some(b) = b_b.pop() {
        let current = b.reference;
        let neighbor = b.neighbors[0].0;

        let current_x = b_x.iter().find(|nn| nn.reference == current).unwrap();
        let neighbor_x = b_x.iter().find(|nn| nn.reference == neighbor).unwrap();

        let current_desc = get_domain_descriptor(current_x.to_owned());
        let neighbor_desc = get_domain_descriptor(neighbor_x.to_owned());
        let avg_desc = (current_desc + neighbor_desc)/2.0;

        descriptors.push((current, avg_desc));
    }
    descriptors
}

fn get_domain_descriptor(b_nns: NearestNeighbor) -> Descriptor {
    let vectors = b_nns.neighbors.iter()
        .map(|(_, neighbor, _)| neighbor.position)
        .collect::<Vec<_>>();
    let sorted_vectors = sort_x(vectors);
    
    let mut arr = [0.0; 18];
    for (vec_id, vec) in sorted_vectors.into_iter().enumerate() {
        let arr_id = vec_id * 3;
        arr[arr_id] = vec.x;
        arr[arr_id + 1] = vec.y;
        arr[arr_id + 2] = vec.z;
    }

    Descriptor::from(arr)
}

fn sort_x(positions: Vectors) -> Vectors { // do something with the y pair
    let (y_top_id, &y_top) = positions.iter().enumerate().max_by(|a, b| a.1.y.total_cmp(&b.1.y)).unwrap();

    let mut positions = positions.clone();
    positions.swap_remove(y_top_id);
    let (y_bottom_id, &y_bottom) = positions.iter().enumerate().min_by(|a, b| a.1.y.total_cmp(&b.1.y)).unwrap();
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

    pos.insert(0, y_top.to_owned());
    pos.insert(1, y_bottom.to_owned());
    pos
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Element;
    use std::f64::consts::PI;

    #[test]
    fn test_sort_x_ordering() {
        // shuffle the octahedral relative vectors arbitrarily for testing
        let input_positions = vec![
            Vector::new(0.0, 0.0, 0.5),   // Eq: PI/2 rad
            Vector::new(0.0, -0.5, 0.0),  // Apical: Bottom
            // Checking arbitrary XZ plane signs for angle computations
            Vector::new(0.5, 0.0, 0.0),   // Eq: 0 rad
            Vector::new(0.0, 0.5, 0.0),   // Apical: Top
            Vector::new(0.0, 0.0, -0.5),  // Eq: 3*PI/2 rad
            Vector::new(-0.5, 0.0, 0.0),  // Eq: PI rad
        ];

        let sorted = sort_x(input_positions);

        assert_eq!(sorted.len(), 6);
        
        // 1. Verify index 0 is top apical (highest Y value)
        assert_eq!(sorted[0], Vector::new(0.0, 0.5, 0.0));
        
        // 2. Verify index 1 is bottom apical (lowest Y value)
        assert_eq!(sorted[1], Vector::new(0.0, -0.5, 0.0));

        // 3. Verify indexes 2..5 are sorted by their XZ-plane rotation angle ascending
        // 0 rad -> PI/2 rad -> PI rad -> 3*PI/2 rad
        assert_eq!(sorted[2], Vector::new(0.5, 0.0, 0.0));   // 0 rad
        assert_eq!(sorted[3], Vector::new(0.0, 0.0, 0.5));   // PI/2 rad (Z positive, X zero)
        assert_eq!(sorted[4], Vector::new(-0.5, 0.0, 0.0));  // PI rad
        assert_eq!(sorted[5], Vector::new(0.0, 0.0, -0.5));  // 3*PI/2 rad
    }

    #[test]
    fn test_get_domain_descriptor() {
        let nn_octahedron = create_mock_octahedron(Vector::zeros());
        let descriptor = get_domain_descriptor(nn_octahedron);

        // Vector length validation (18 elements total for 6 coordinates * 3 dimensions)
        assert_eq!(descriptor.len(), 18);

        // Spot-check the flattened contents after sorting:
        // First 3 elements must belong to top apical: [0.0, 0.5, 0.0]
        assert_eq!(descriptor[0], 0.0);
        assert_eq!(descriptor[1], 0.5);
        assert_eq!(descriptor[2], 0.0);

        // Next 3 elements must belong to bottom apical: [0.0, -0.5, 0.0]
        assert_eq!(descriptor[3], 0.0);
        assert_eq!(descriptor[4], -0.5);
        assert_eq!(descriptor[5], 0.0);
    }

#[test]
    fn test_calculate_descriptors_averaging() {
        // Establish two coordinated B-sites adjacent to each other
        let b1_center = Vector::new(0.0, 0.0, 0.0);
        let b2_center = Vector::new(1.0, 0.0, 0.0);

        let atom_b1 = create_mock_atom(Element::Ti, b1_center);
        let atom_b2 = create_mock_atom(Element::Ti, b2_center);

        // 1. Build B-B linkages (Who is whose neighbor)
        let b_b_pair = vec![
            NearestNeighbor {
                reference: atom_b1.clone(),
                neighbors: vec![(atom_b2.clone(), atom_b2.clone(), 1.0)],
            },
            NearestNeighbor {
                reference: atom_b2.clone(),
                neighbors: vec![(atom_b1.clone(), atom_b1.clone(), 1.0)],
            }
        ];

        // 2. Build local Anion contexts (X networks)
        // To ensure the descriptors evaluate to different values, we distort the 
        // neighbor relative coordinates *without* altering the root reference center.
        let distortion = Vector::new(0.0, 0.1, 0.0);
        let mut distorted_octahedron = create_mock_octahedron(b2_center);
        for (_, rel_atom, _) in distorted_octahedron.neighbors.iter_mut() {
            rel_atom.position += distortion;
        }

        let b_x_pair = vec![
            create_mock_octahedron(b1_center),
            distorted_octahedron.clone(),
        ];

        let results = calculate_descriptors(b_b_pair, b_x_pair);

        assert_eq!(results.len(), 2);

        // Extract individual independent descriptors to manually calculate the ground truth
        let desc_b1 = get_domain_descriptor(create_mock_octahedron(b1_center));
        let desc_b2 = get_domain_descriptor(distorted_octahedron);
        let expected_average_desc = (desc_b1 + desc_b2) / 2.0;

        for (mapped_atom, computed_desc) in results {
            if mapped_atom.position == b1_center {
                assert!((computed_desc - expected_average_desc).norm() < 1e-12);
            } else if mapped_atom.position == b2_center {
                assert!((computed_desc - expected_average_desc).norm() < 1e-12);
            } else {
                panic!("Encountered unknown structural mapping reference point!");
            }
        }
    }

    fn create_mock_atom(element: Element, pos: Vector) -> Atom {
        Atom {
            atom_type: element,
            position: pos,
        }
    }

    fn create_mock_octahedron(reference_pos: Vector) -> NearestNeighbor {
        let reference = create_mock_atom(Element::Ti, reference_pos);
        
        let rel_positions = vec![
            Vector::new(0.0, 0.5, 0.0),   // Top apical (y-high)
            Vector::new(0.0, -0.5, 0.0),  // Bottom apical (y-low)
            Vector::new(0.5, 0.0, 0.0),   // Equatorial (0 rad)
            Vector::new(0.0, 0.0, 0.5),   // Equatorial (PI/2 rad)
            Vector::new(-0.5, 0.0, 0.0),  // Equatorial (PI rad)
            Vector::new(0.0, 0.0, -0.5),  // Equatorial (3*PI/2 rad)
        ];

        let neighbors = rel_positions
            .into_iter()
            .map(|rel_pos| {
                let abs_atom = create_mock_atom(Element::O, reference_pos + rel_pos);
                let rel_atom = create_mock_atom(Element::O, rel_pos);
                (abs_atom, rel_atom, 0.25)
            })
            .collect::<Vec<_>>();

        NearestNeighbor { reference, neighbors }
    }
}
