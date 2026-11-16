use crate::{cell::simulation::SimCell, classification::descriptor::{self, DescribedCell}, types::{Atom, CartAtom, Cartesian}};
use super::descriptor::{Descriptor, Descriptors, BDescriptor, BDescriptors};
use crate::prelude::*;
use rand::seq::{IteratorRandom};

#[derive(Clone, Default, Debug)]
pub struct Cluster {
    pub described_cells: Vec<DescribedCell>,
    pub centroid: Descriptor,
}

pub type Clusters = Vec<Cluster>;

pub fn k_means(data: Vec<DescribedCell>, n_clusters: usize) -> Clusters {
    assert!(n_clusters > 0 && n_clusters < data.len());

    let mut current_centroids: Descriptors = data
        .iter()
        .sample(&mut rand::rng(), n_clusters)
        .into_iter()
        .map(|cell| cell.b_descriptor.descriptor)
        .collect();

    loop {
        let assignments: Vec<usize> = data
            .iter()
            .map(|cell| {
                current_centroids
                    .iter()
                    .enumerate()
                    .map(|(id, centroid)| {
                        (
                            id,
                            (cell.b_descriptor.descriptor - centroid).norm_squared(),
                        )
                    })
                    .min_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap()
                    .0
            })
            .collect();

        let new_centroids =
            update_centroids(&data, &assignments, &current_centroids);

        if converged(&current_centroids, &new_centroids) {
            let mut clusters: Clusters = new_centroids
                .into_iter()
                .map(|centroid| Cluster {
                    described_cells: Vec::new(),
                    centroid,
                })
                .collect();

            for (cell, cluster_id) in data.into_iter().zip(assignments) {
                clusters[cluster_id].described_cells.push(cell);
            }

            return clusters;
        }

        current_centroids = new_centroids;
    }
}

#[allow(clippy::cast_precision_loss)]
pub fn descriptor_closest_to_domain_center(cluster: &BDescriptors) -> Result<&BDescriptor> {
    let centroid = cluster
        .iter()
        .fold(Descriptor::zeros(), |acc, &point| acc + point.descriptor);

    let cluster_center = centroid / cluster.len() as f64;

    let descriptor_dists = cluster.iter()
        .map(|desc| (desc.descriptor - cluster_center).norm_squared())
        .collect::<Vec<_>>();

    let center_descriptor = cluster.iter()
        .zip(descriptor_dists)
        .min_by(|(_, dist_a), (_, dist_b)| dist_a.total_cmp(dist_b))
        .ok_or_else(|| Error::Generic("cluster is empty".into()))?;

    Ok(center_descriptor.0)
}

fn converged(current: &[Descriptor], new: &[Descriptor]) -> bool {
    current
        .iter()
        .zip(new)
        .all(|(o, n)| (o - n)
            .norm_squared() < 1e-12)
}

#[allow(clippy::cast_precision_loss)]
fn update_centroids(
    data: &[DescribedCell],
    assignments: &[usize],
    current_centroids: &[Descriptor],
) -> Descriptors {
    let mut sums = vec![Descriptor::zeros(); current_centroids.len()];
    let mut counts = vec![0usize; current_centroids.len()];

    for (cell, &cluster_id) in data.iter().zip(assignments) {
        sums[cluster_id] += cell.b_descriptor.descriptor;
        counts[cluster_id] += 1;
    }

    sums.into_iter()
        .zip(counts)
        .zip(current_centroids)
        .map(|((sum, count), &previous)| {
            if count == 0 {
                previous
            } else {
                sum / count as f64
            }
        })
        .collect()
}

fn calculate_dist<'a>(point: &'a Descriptor, centroids: &Vec<Descriptor>) -> (&'a Descriptor, Vec<f64>) {
    let mut dists: Vec<f64> = Vec::with_capacity(centroids.len()); //change to static arr
    for &centroid in centroids {
        let dist = (point - centroid).norm_squared();
        dists.push(dist);
    }
    (point, dists)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cell::{simulation::SimCell, UnitCell, UnitCellAtoms};
    use crate::types::{Atom, Cartesian, Element, Vector};

    #[test]
    fn test_calculate_dist() {
        let point = uniform_descriptor(1.0);
        let centroids = vec![
            uniform_descriptor(1.0),
            uniform_descriptor(2.0),
        ];

        let (returned_point, distances) = calculate_dist(&point, &centroids);

        assert_eq!(returned_point, &point);
        assert_eq!(distances.len(), 2);
        assert!((distances[0] - 0.0).abs() < 1e-12);
        assert!((distances[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_descriptor_closest_to_center() {
        let descriptors = vec![
            b_descriptor(1.0),
            b_descriptor(2.0),
            b_descriptor(6.0),
        ];

        let center = descriptor_closest_to_domain_center(&descriptors).unwrap();
        let expected = uniform_descriptor(2.0);

        assert!((center.descriptor - expected).norm_squared() < 1e-12);
        assert!(std::ptr::eq(center, &descriptors[1]));
    }

    #[test]
    fn test_descriptor_closest_to_center_empty_cluster() {
        let error = descriptor_closest_to_domain_center(&Vec::new())
            .unwrap_err();

        assert!(matches!(
            error,
            Error::Generic(message) if message == "cluster is empty"
        ));
    }

    #[test]
    fn test_update_centroids() {
        let data = vec![
            described_cell(0.0),
            described_cell(2.0),
            described_cell(10.0),
        ];

        let assignments = vec![0, 0, 1];
        let current_centroids = vec![
            Descriptor::zeros(),
            Descriptor::zeros(),
        ];

        let updated = update_centroids(
            &data,
            &assignments,
            &current_centroids,
        );

        assert_eq!(updated.len(), 2);
        assert!((updated[0] - uniform_descriptor(1.0)).norm_squared() < 1e-12);
        assert!((updated[1] - uniform_descriptor(10.0)).norm_squared() < 1e-12);
    }

    #[test]
    fn test_update_centroids_empty_cluster() {
        let data = vec![
            described_cell(0.0),
            described_cell(2.0),
        ];

        let assignments = vec![0, 0];
        let current_centroids = vec![
            uniform_descriptor(0.0),
            uniform_descriptor(10.0),
        ];

        let updated = update_centroids(
            &data,
            &assignments,
            &current_centroids,
        );

        assert_eq!(updated.len(), 2);
        assert!((updated[0] - uniform_descriptor(1.0)).norm_squared() < 1e-12);
        assert!((updated[1] - uniform_descriptor(10.0)).norm_squared() < 1e-12);
    }

    #[test]
    fn test_converged_conditions() {
        let c1 = vec![
            uniform_descriptor(1.0),
            uniform_descriptor(5.0),
        ];

        // Exact match
        let c2 = c1.clone();
        assert!(converged(&c1, &c2));

        // Within tolerance
        let c3 = vec![
            uniform_descriptor(1.0) + Descriptor::from_element(1e-8),
            uniform_descriptor(5.0),
        ];
        assert!(converged(&c1, &c3));

        // Outside tolerance
        let c4 = vec![
            uniform_descriptor(1.01),
            uniform_descriptor(5.0),
        ];
        assert!(!converged(&c1, &c4));
    }

    #[test]
    fn test_k_means_clustering_execution() {
        let data = vec![
            described_structured_cell(0.0, 0.1),
            described_structured_cell(0.1, 0.0),
            described_structured_cell(0.0, 0.0),
            described_structured_cell(100.0, 100.0),
            described_structured_cell(100.1, 100.0),
            described_structured_cell(100.0, 100.1),
        ];

        let final_clusters = k_means(data, 2);

        assert_eq!(final_clusters.len(), 2);

        let mut lengths: Vec<usize> = final_clusters
            .iter()
            .map(|cluster| cluster.described_cells.len())
            .collect();

        lengths.sort();

        assert_eq!(
            lengths,
            vec![3, 3],
            "Data points failed to separate into two clusters."
        );

        // Verify that each cluster contains points from only one group.
        for cluster in &final_clusters {
            let first = cluster.described_cells[0]
                .b_descriptor.descriptor[0];

            for cell in &cluster.described_cells {
                let x = cell.b_descriptor.descriptor[0];

                assert!(
                    (x - first).abs() < 1.0,
                    "Cluster contains points from different groups."
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_k_means_panic_on_zero_clusters() {
        let data = vec![
            described_cell(1.0),
            described_cell(2.0),
        ];

        let _ = k_means(data, 0);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_k_means_panic_on_excessive_clusters() {
        let data = vec![
            described_cell(1.0),
            described_cell(2.0),
        ];

        let _ = k_means(data, 2);
    }

    fn uniform_descriptor(val: f64) -> Descriptor {
        Descriptor::from_element(val)
    }

    fn structured_descriptor(x: f64, y: f64) -> Descriptor {
        Descriptor::from([x, y])
    }

    fn b_descriptor(val: f64) -> BDescriptor {
        BDescriptor {
            descriptor: uniform_descriptor(val),
            reference_b_atom: test_b_atom(),
        }
    }

    fn described_cell(val: f64) -> DescribedCell {
        described_cell_from_descriptor(uniform_descriptor(val))
    }

    fn described_structured_cell(x: f64, y: f64) -> DescribedCell {
        described_cell_from_descriptor(structured_descriptor(x, y))
    }

    fn described_cell_from_descriptor(descriptor: Descriptor) -> DescribedCell {
        let b_atom = test_b_atom();

        DescribedCell {
            b_descriptor: BDescriptor {
                descriptor,
                reference_b_atom: b_atom,
            },
            cell: SimCell(UnitCell {
                cell_atoms: UnitCellAtoms {
                    a: Vec::new(),
                    b: vec![b_atom],
                    x: Vec::new(),
                },
                center_of_mass: Cartesian::new(Vector::zeros()),
            }),
        }
    }

    fn test_b_atom() -> Atom<Cartesian> {
        Atom {
            atom_type: Some(Element::Ti),
            position: Cartesian::new(Vector::zeros()),
        }
    }
}
