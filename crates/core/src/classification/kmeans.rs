use crate::{classification::descriptor, types::{Atom, CartAtom, Cartesian}};
use super::descriptor::{Descriptor, Descriptors, BDescriptor, BbDescriptors};
use crate::prelude::*;
use rand::seq::{IteratorRandom};

#[derive(Clone, Default, Debug)]
pub struct Cluster {
    pub bb_descriptors: BbDescriptors,
    pub centroid: Descriptor
}

pub type Clusters = Vec<Cluster>;

pub fn k_means(data: &BbDescriptors, n_clusters: usize) -> Clusters {
    assert!(n_clusters > 0 && n_clusters < data.len());

    let mut current_centroids: Descriptors = data
        .iter()
        .sample(&mut rand::rng(), n_clusters)
        .into_iter()
        .copied()
        .map(|bb_desc| bb_desc.descriptor)
        .collect();

    loop {
        let mut clusters: Clusters = vec![Cluster::default(); n_clusters];
        for (cluster, centroid) in clusters.iter_mut().zip(current_centroids.clone()) {
            cluster.centroid = centroid;
        }

        for bb_desc in data {
            let cluster_id = current_centroids
                .iter()
                .enumerate()
                .map(|(id, centroid)| (id, (bb_desc.descriptor - centroid).norm_squared()))
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(id, _)| id)
                .unwrap();

            clusters[cluster_id].bb_descriptors.push(*bb_desc);
        }

        let new_centroids = update_centroids(&clusters);

        if converged(&current_centroids, &new_centroids) {
            for (cluster, centroid) in clusters.iter_mut().zip(new_centroids.clone()) {
                cluster.centroid = centroid;
            }
            return clusters;
        }

        current_centroids = new_centroids;
    }
}

#[allow(clippy::cast_precision_loss)]
pub fn descriptor_closest_to_center(cluster: &BbDescriptors) -> Result<&BDescriptor> {
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
fn update_centroids(clusters: &Clusters) -> Descriptors {
    let mut centroids: Vec<Descriptor> = Vec::with_capacity(clusters.len());
    for cluster in clusters {
        let centroid = cluster.bb_descriptors
            .iter()
            .fold(Descriptor::zeros(), |acc, &point| acc + point.descriptor);

        centroids.push(centroid / cluster.bb_descriptors.len() as f64);
    }
    centroids
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
    use crate::types::{Atom, Cartesian, Element, Vector};

    use super::*;

    #[test]
    fn test_calculate_dist() {
        let point = uniform_descriptor(1.0);
        let centroids = vec![
            uniform_descriptor(1.0), // Distance should be 0
            uniform_descriptor(2.0), // Distance: 2 dimensions * (1.0 - 2.0)^2 = 2.0
        ];

        let (returned_point, distances) = calculate_dist(&point, &centroids);

        assert_eq!(returned_point, &point);
        assert_eq!(distances.len(), 2);
        assert!((distances[0] - 0.0).abs() < 1e-12);
        assert!((distances[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_descriptor_closest_to_center() {
        let cluster = vec![
            bb_descriptor(1.0),
            bb_descriptor(2.0),
            bb_descriptor(6.0),
        ];

        let center = descriptor_closest_to_center(&cluster).unwrap();
        // Mean is 3.0; the closest input descriptor is 2.0.
        let expected_center = uniform_descriptor(2.0);

        assert!((center.descriptor - expected_center).norm_squared() < 1e-12);
        assert!(std::ptr::eq(center, &cluster[1]));
    }

    #[test]
    fn test_descriptor_closest_to_center_empty_cluster() {
        let error = descriptor_closest_to_center(&Vec::new()).unwrap_err();

        assert!(matches!(error, Error::Generic(message) if message == "cluster is empty"));
    }

    #[test]
    fn test_update_centroids() {
        let clusters = vec![
            Cluster {
                bb_descriptors: vec![
                    bb_descriptor(0.0),
                    bb_descriptor(2.0),
                ],
                centroid: Descriptor::zeros(),
            },
            Cluster {
                bb_descriptors: vec![bb_descriptor(10.0)],
                centroid: Descriptor::zeros(),
            },
        ];

        let updated = update_centroids(&clusters);

        assert_eq!(updated.len(), 2);
        assert!((updated[0] - uniform_descriptor(1.0)).norm_squared() < 1e-12);
        assert!((updated[1] - uniform_descriptor(10.0)).norm_squared() < 1e-12);
    }

    #[test]
    fn test_converged_conditions() {
        let c1 = vec![uniform_descriptor(1.0), uniform_descriptor(5.0)];
        
        // Exact match
        let c2 = vec![uniform_descriptor(1.0), uniform_descriptor(5.0)];
        assert!(converged(&c1, &c2));

        // Within allowed float precision threshold (< 1e-12 per element squared)
        let c3 = vec![
            uniform_descriptor(1.0) + Descriptor::from_element(1e-8), 
            uniform_descriptor(5.0),
        ];
        // 2 elements * (1e-8)^2 = 2e-16, which is < 1e-12
        assert!(converged(&c1, &c3));

        // Outside threshold
        let c4 = vec![uniform_descriptor(1.01), uniform_descriptor(5.0)];
        assert!(!converged(&c1, &c4));
    }

    #[test]
    fn test_k_means_clustering_execution() {
        // Build a highly distinct, separable dataset:
        // 3 points close to 0.0, 3 points close to 100.0
        let data = vec![
            bb_structured_descriptor(0.0, 0.1, 0.0),
            bb_structured_descriptor(0.1, 0.0, 0.1),
            bb_structured_descriptor(0.0, 0.0, 0.0),
            bb_structured_descriptor(100.0, 100.0, 100.0),
            bb_structured_descriptor(100.1, 100.0, 100.0),
            bb_structured_descriptor(100.0, 100.1, 100.2),
        ];

        let n_clusters = 2;
        let final_clusters = k_means(&data, n_clusters);

        // Verify the exact cluster split size matching the target groups
        assert_eq!(final_clusters.len(), 2);
        
        let mut lengths = vec![
            final_clusters[0].bb_descriptors.len(),
            final_clusters[1].bb_descriptors.len(),
        ];
        lengths.sort();
        
        assert_eq!(lengths, vec![3, 3], "Data points failed to segment evenly into clear domains.");
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_k_means_panic_on_zero_clusters() {
        let data = vec![bb_descriptor(1.0), bb_descriptor(2.0)];
        let _ = k_means(&data, 0);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_k_means_panic_on_excessive_clusters() {
        let data = vec![bb_descriptor(1.0), bb_descriptor(2.0)];
        // Requesting 2 or more clusters when data length is 2 breaks your `n_clusters < data.len()` guardrail
        let _ = k_means(&data, 2);
    }

    fn uniform_descriptor(val: f64) -> Descriptor {
        Descriptor::from_element(val)
    }

    fn bb_descriptor(val: f64) -> BDescriptor {
        BDescriptor {
            descriptor: uniform_descriptor(val),
            reference_b_atom: Atom {
                atom_type: Some(Element::Ti),
                position: Cartesian::new(Vector::zeros())
            }
        }
    }

    fn bb_structured_descriptor(x: f64, y: f64, z: f64) -> BDescriptor {
        BDescriptor {
            descriptor: structured_descriptor(x, y, z),
            reference_b_atom: Atom {
                atom_type: Some(Element::Ti),
                position: Cartesian::new(Vector::zeros())
            }
        }
    }

    fn structured_descriptor(x: f64, y: f64, _z: f64) -> Descriptor {
        Descriptor::from([x, y])
    }
}
