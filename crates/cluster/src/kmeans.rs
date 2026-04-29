use crate::{Descriptor, descriptor::{self, Descriptors}};
use rand::seq::{IteratorRandom};

pub fn k_means(data: Descriptors, n_clusters: usize) -> Vec<Descriptors> {
    assert!(n_clusters > 0 && n_clusters < data.len());

    let mut current_centroids: Vec<Descriptor> = data.iter()
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

fn converged(current: &Vec<Descriptor>, new: &Vec<Descriptor>) -> bool {
    current.iter().zip(new).all(|(o, n)| (o - n).norm_squared() < 1e-12)
}

fn update_centroids(clusters: &Vec<Vec<&Descriptor>>) -> Vec<Descriptor> {
    let mut centroids: Vec<Descriptor> = Vec::with_capacity(clusters.len());
    for cluster in clusters {
        let centroid = cluster.iter()
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

