use crate::{cell::UnitCell, classification::kmeans::Clusters, types::Atoms};
use crate::prelude::*;

pub(crate) fn find_closest_to_centroid(clusters: Clusters) -> Result<Atoms> {
    let mut centroid_atoms = Atoms::with_capacity(clusters.len());

    for cluster in clusters {
        let closest_to_centroid = cluster.bb_descriptors
            .iter()
            .map(|&desc| (desc, (desc.descriptor - cluster.centroid).norm_squared()))
            .min_by(|(_,a), (_, b)| a.total_cmp(b))
            .map(|(desc, _)| desc)
            .ok_or_else(|| Error::Generic("failed to find descriptor closest to calculated centroid".into()))?;

        centroid_atoms.push(closest_to_centroid.reference_b_atom);
    }

    Ok(centroid_atoms)
}
