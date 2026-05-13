use super::descriptor::{Descriptor, Descriptors};
use rand::seq::{IteratorRandom};

pub fn k_means(data: Descriptors, n_clusters: usize) -> Vec<Descriptors> {
    assert!(n_clusters > 0 && n_clusters < data.len());

    let mut current_centroids: Vec<Descriptor> = data
        .iter()
        .sample(&mut rand::rng(), n_clusters)
        .into_iter()
        .cloned()
        .collect();

    loop {
        let mut clusters: Vec<Vec<&Descriptor>> = vec![Vec::new(); current_centroids.len()];
        for point in &data {
            let cluster_id = current_centroids
                .iter()
                .enumerate()
                .map(|(id, centroid)| (id, (point - centroid).norm_squared()))
                .min_by(|(_, a), (_, b)| a.total_cmp(&b))
                .map(|(id, _)| id)
                .unwrap();

            clusters[cluster_id].push(point);
        }

        let new_centroids = update_centroids(&clusters);

        if converged(&current_centroids, &new_centroids) {
            let clusters: Vec<Descriptors> = clusters
                .into_iter()
                .map(|cluster| cluster.into_iter().cloned().collect())
                .collect();
            return clusters;
        }
        current_centroids = new_centroids;
    }
}

pub fn get_cluster_center(cluster: &Descriptors) -> Descriptor {
    let centroid = cluster
        .iter()
        .fold(Descriptor::zeros(), |acc, &point| acc + point);

    centroid / cluster.len() as f64
}

fn converged(current: &Vec<Descriptor>, new: &Vec<Descriptor>) -> bool {
    current.iter().zip(new).all(|(o, n)| (o - n).norm_squared() < 1e-12)
}

fn update_centroids(clusters: &Vec<Vec<&Descriptor>>) -> Vec<Descriptor> {
    let mut centroids: Vec<Descriptor> = Vec::with_capacity(clusters.len());
    for cluster in clusters {
        let centroid = cluster
            .iter()
            .fold(Descriptor::zeros(), |acc, &point| acc + point);

        centroids.push(centroid / cluster.len() as f64);
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
    use super::*;

    #[test]
    fn test_calculate_dist() {
        let point = uniform_descriptor(1.0);
        let centroids = vec![
            uniform_descriptor(1.0), // Distance should be 0
            uniform_descriptor(2.0), // Distance: 18 dimensions * (1.0 - 2.0)^2 = 18.0
        ];

        let (returned_point, distances) = calculate_dist(&point, &centroids);

        assert_eq!(returned_point, &point);
        assert_eq!(distances.len(), 2);
        assert!((distances[0] - 0.0).abs() < 1e-12);
        assert!((distances[1] - 18.0).abs() < 1e-12);
    }

    #[test]
    fn test_get_cluster_center() {
        let cluster = vec![
            uniform_descriptor(1.0),
            uniform_descriptor(2.0),
            uniform_descriptor(3.0),
        ];

        let center = get_cluster_center(&cluster);
        let expected_center = uniform_descriptor(2.0); // Mean of 1, 2, 3 is 2

        assert!((center - expected_center).norm_squared() < 1e-12);
    }

    #[test]
    fn test_update_centroids() {
        let p1 = uniform_descriptor(0.0);
        let p2 = uniform_descriptor(2.0);
        let p3 = uniform_descriptor(10.0);

        // Map cluster references to match Vec<Vec<&Descriptor>> signature
        let clusters: Vec<Vec<&Descriptor>> = vec![
            vec![&p1, &p2], // Center should be 1.0
            vec![&p3],      // Center should be 10.0
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
        // 18 elements * (1e-8)^2 = 1.8e-15, which is < 1e-12
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
            structured_descriptor(0.0, 0.1, 0.0),
            structured_descriptor(0.1, 0.0, 0.1),
            structured_descriptor(0.0, 0.0, 0.0),
            structured_descriptor(100.0, 100.0, 100.0),
            structured_descriptor(100.1, 100.0, 100.0),
            structured_descriptor(100.0, 100.1, 100.2),
        ];

        let n_clusters = 2;
        let final_clusters = k_means(data, n_clusters);

        // Verify the exact cluster split size matching the target groups
        assert_eq!(final_clusters.len(), 2);
        
        let mut lengths = vec![final_clusters[0].len(), final_clusters[1].len()];
        lengths.sort();
        
        assert_eq!(lengths, vec![3, 3], "Data points failed to segment evenly into clear domains.");
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_k_means_panic_on_zero_clusters() {
        let data = vec![uniform_descriptor(1.0), uniform_descriptor(2.0)];
        let _ = k_means(data, 0);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_k_means_panic_on_excessive_clusters() {
        let data = vec![uniform_descriptor(1.0), uniform_descriptor(2.0)];
        // Requesting 2 or more clusters when data length is 2 breaks your `n_clusters < data.len()` guardrail
        let _ = k_means(data, 2);
    }

    fn uniform_descriptor(val: f64) -> Descriptor {
        Descriptor::from_element(val)
    }

    fn structured_descriptor(x: f64, y: f64, z: f64) -> Descriptor {
        let mut arr = [0.0; 18];
        for i in 0..6 {
            arr[i * 3] = x;
            arr[i * 3 + 1] = y;
            arr[i * 3 + 2] = z;
        }
        Descriptor::from(arr)
    }
}
