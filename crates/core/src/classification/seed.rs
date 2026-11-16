use crate::classification::kmeans::Clusters;
use crate::prelude::*;
use crate::types::CartAtoms;

pub fn find_closest_to_centroid(clusters: &Clusters) -> Result<CartAtoms> {
    let mut centroid_atoms = CartAtoms::with_capacity(clusters.len());

    for cluster in clusters {
        let closest_to_centroid = cluster
            .described_cells
            .iter()
            .min_by(|a, b| {
                let dist_a =
                    (a.b_descriptor.descriptor - cluster.centroid).norm_squared();
                let dist_b =
                    (b.b_descriptor.descriptor - cluster.centroid).norm_squared();

                dist_a.total_cmp(&dist_b)
            })
            .ok_or_else(|| Error::Generic(
                "failed to find descriptor closest to calculated centroid".into()
            ))?;

        centroid_atoms.push(closest_to_centroid.b_descriptor.reference_b_atom);
    }

    Ok(centroid_atoms)
}
