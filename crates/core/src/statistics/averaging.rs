use crate::observables::{BinnedObservable, Observable};
use crate::{cell::simulation::FittedSimCell, types::AsVector};
use crate::prelude::{Result, Error};
use crate::types::Vector;

mod preprocessing;

/// averages observable over z-y crosssection and returns observable profile along x axis
pub fn crosssection_average<Obs>(
    cell_observables: &Vec<(&FittedSimCell, Obs)>,
    threshold: f64
) -> BinnedObservable<Obs>
where 
    Obs: Observable + AsVector + From<Vector>
{
    assert!(
        !cell_observables.is_empty(), 
        "Bug: Crosssection avereraging is called but observables are empty!"
    );

    let cell_id_bins = bins_along_x(cell_observables, threshold);

    let (bins, avgs) = cross_avg(cell_observables, cell_id_bins);

    BinnedObservable::<Obs> {
        bins,
        observable: avgs
    }
}

#[allow(clippy::cast_precision_loss)]
pub fn block_average<Obs>( // TODO check function again & refactor
    profiles: &[BinnedObservable<Obs>],
    n_blocks: usize
) -> Result<BinnedObservable<Obs>>
where
    Obs: Observable + AsVector + From<Vector>
{
    assert!(!profiles.is_empty(), "avg_observables is empty");

    if n_blocks == 0 {
        return Err(Error::Generic("Number of blocks is set to 0".into()))
    }

    if !profiles.len().is_multiple_of(n_blocks) {
        return Err(Error::Generic("Length of avg_observables not divisible by n_blocks".into()))
    }     

    let block_size = profiles.len() / n_blocks;
    let obs_len = profiles[0].observable.len();

    let mut block_avg_obs: Vec<Vec<Obs>> = Vec::with_capacity(n_blocks);
    let mut block_avg_bins: Vec<Vec<f64>> = Vec::with_capacity(n_blocks);

    for profile_block in profiles.chunks_exact(block_size) {
        let mut subblock_avg_bins: Vec<f64> = Vec::with_capacity(obs_len);
        let mut subblock_avg_obs: Vec<Obs> = Vec::with_capacity(obs_len);

        for k in 0..obs_len {
            let cell_obs_bins = profile_block
                .iter()
                .fold(0.0, |acc, profile|
                    acc + profile.bins[k]
                );

            let bins = cell_obs_bins / block_size as f64;
            subblock_avg_bins.push(bins);

            let cell_obs_avg = profile_block
                .iter()
                .fold(Vector::zeros(), |acc, profile|
                    acc + profile.observable[k].as_vector()
                );

            let obs = Obs::from(cell_obs_avg / block_size as f64);
            subblock_avg_obs.push(obs);
        }

        block_avg_bins.push(subblock_avg_bins);
        block_avg_obs.push(subblock_avg_obs);
    }

    let mut avg_bins: Vec<f64> = Vec::with_capacity(obs_len);
    for k in 0..obs_len {
        let bins = block_avg_bins
            .iter()
            .fold(0.0, |acc, avg| 
                acc + avg[k]
            );

        let bins = bins / block_avg_bins.len() as f64;
        avg_bins.push(bins);
    }

    let mut avg_obs: Vec<Obs> = Vec::with_capacity(obs_len);
    for k in 0..obs_len {
        let obs = block_avg_obs
            .iter()
            .fold(Vector::zeros(), |acc, avg| 
                acc + avg[k].as_vector()
            );

        let obs = Obs::from(obs / block_avg_bins.len() as f64);
        avg_obs.push(obs);
    }

    let avg_obs = BinnedObservable {
        bins: avg_bins,
        observable: avg_obs
    };

    Ok(avg_obs)
}


fn bins_along_x<Obs: AsVector>(
    cell_observables: &Vec<(&FittedSimCell, Obs)>,
    threshold: f64
) -> Vec<Vec<usize>> {

    let ids = argsort(cell_observables);
    find_and_fill_bins(cell_observables, ids, threshold)
}

fn argsort<Obs: AsVector>(cell_observables: &[(&FittedSimCell, Obs)]) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..cell_observables.len()).collect();

    ids.sort_by(|&a, &b| { 
        cell_observables[a].0.unit_cell.cell_atoms.b[0].position.as_vector().x.total_cmp(
            &cell_observables[b].0.unit_cell.cell_atoms.b[0].position.as_vector().x
        )
    });

    ids
}

fn find_and_fill_bins<Obs: AsVector>(
    cell_observables: &[(&FittedSimCell, Obs)],
    ids: Vec<usize>,
    threshold: f64
) -> Vec<Vec<usize>> {

    let cell_x_at = |id: usize| { 
        cell_observables[id].0.unit_cell.cell_atoms.b[0].position.as_vector().x
    };

    let mut cell_id_bins: Vec<Vec<usize>> = Vec::new();
    let mut current_bin_x = cell_x_at(ids[0]);
    let mut current_bin: Vec<usize> = Vec::new();

    for id in ids {
        let diff = (current_bin_x - cell_x_at(id)).abs();
        if diff <= threshold {
            current_bin.push(id);
        } else {
            cell_id_bins.push(current_bin);
            current_bin_x = cell_x_at(id);
            current_bin = vec![id];
        }
    }

    cell_id_bins.push(current_bin);

    cell_id_bins
}

#[allow(clippy::cast_precision_loss)]
fn cross_avg<Obs>(
    cell_observables: &[(&FittedSimCell, Obs)],
    cell_id_bins: Vec<Vec<usize>>
) -> (Vec<f64>, Vec<Obs>)
where
    Obs: Observable + AsVector + From<Vector>
{
    let mut avg_obs: Vec<Obs> = Vec::with_capacity(cell_id_bins.len());
    let mut avg_bins: Vec<f64> = Vec::with_capacity(cell_id_bins.len());

    for bin in cell_id_bins {
        let obs_avg: Vector = bin.iter()
            .fold(Vector::zeros(), |acc, &id| acc + cell_observables[id].1.as_vector())
            .map(|x| x / bin.len() as f64);

        let bin_sum = bin.iter()
            .fold(0.0, |acc, &id| acc + cell_observables[id].0.unit_cell.cell_atoms.b[0].position.as_vector().x);

        let bin_avg = bin_sum / bin.len() as f64;
        avg_bins.push(bin_avg);

        let obs_avg = Obs::from(obs_avg);
        avg_obs.push(obs_avg);
    }

    (avg_bins, avg_obs)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cell::{UnitCell, UnitCellAtoms},
        observables::op::Op,
        types::{Atom, Cartesian, CellAlignment, Element, RotationVariant, UQuaternion},
    };

    const EPS: f64 = 1.0e-12;

    #[test]
    fn argsort_orders_observables_by_b_site_x_position() {
        let items = vec![
            sample(2.0, Vector::zeros()),
            sample(0.0, Vector::zeros()),
            sample(1.0, Vector::zeros()),
        ];
        let cell_observables = refs(&items);

        assert_eq!(argsort(&cell_observables), vec![1, 2, 0]);
    }

    #[test]
    fn find_and_fill_bins_groups_sorted_ids_within_threshold() {
        let items = vec![
            sample(0.00, Vector::zeros()),
            sample(0.05, Vector::zeros()),
            sample(1.00, Vector::zeros()),
            sample(1.04, Vector::zeros()),
        ];
        let cell_observables = refs(&items);

        let bins = find_and_fill_bins(
            &cell_observables,
            vec![0, 1, 2, 3],
            0.1,
        );

        assert_eq!(bins, vec![vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn find_and_fill_bins_uses_first_position_as_reference() {
        let items = vec![
            sample(0.00, Vector::zeros()),
            sample(0.10, Vector::zeros()),
            sample(0.19, Vector::zeros()),
        ];
        let cell_observables = refs(&items);

        let bins = find_and_fill_bins(
            &cell_observables,
            vec![0, 1, 2],
            0.1,
        );

        assert_eq!(bins, vec![vec![0, 1], vec![2]]);
    }

    #[test]
    fn bins_along_x_sorts_then_groups_cells() {
        let items = vec![
            sample(1.04, Vector::zeros()),
            sample(0.05, Vector::zeros()),
            sample(1.00, Vector::zeros()),
            sample(0.00, Vector::zeros()),
        ];
        let cell_observables = refs(&items);

        let bins = bins_along_x(&cell_observables, 0.1);

        assert_eq!(bins, vec![vec![3, 1], vec![2, 0]]);
    }

    #[test]
    fn cross_avg_averages_positions_and_observables() {
        let items = vec![
            sample(0.0, Vector::new(1.0, 2.0, 3.0)),
            sample(0.1, Vector::new(3.0, 4.0, 5.0)),
            sample(1.0, Vector::new(10.0, 20.0, 30.0)),
        ];
        let cell_observables = refs(&items);

        let (bins, observables) =
            cross_avg(&cell_observables, vec![vec![0, 1], vec![2]]);

        assert_eq!(bins.len(), 2);
        assert_eq!(observables.len(), 2);

        assert_close(bins[0], 0.05);
        assert_close(bins[1], 1.0);

        assert_vec_approx_eq(
            observables[0].as_vector(),
            &Vector::new(2.0, 3.0, 4.0),
        );
        assert_vec_approx_eq(
            observables[1].as_vector(),
            &Vector::new(10.0, 20.0, 30.0),
        );
    }

    #[test]
    fn crosssection_average_groups_by_x_and_returns_bin_averages() {
        let items = vec![
            sample(1.00, Vector::new(10.0, 0.0, 0.0)),
            sample(0.05, Vector::new(3.0, 0.0, 0.0)),
            sample(0.00, Vector::new(1.0, 0.0, 0.0)),
        ];
        let cell_observables = refs(&items);

        let averaged = crosssection_average(&cell_observables, 0.1);

        assert_eq!(averaged.bins.len(), 2);
        assert_eq!(averaged.observable.len(), 2);

        assert_close(averaged.bins[0], 0.025);
        assert_close(averaged.bins[1], 1.0);

        assert_vec_approx_eq(
            averaged.observable[0].as_vector(),
            &Vector::new(2.0, 0.0, 0.0),
        );
        assert_vec_approx_eq(
            averaged.observable[1].as_vector(),
            &Vector::new(10.0, 0.0, 0.0),
        );
    }

    #[test]
    #[should_panic(expected = "observables are empty")]
    fn crosssection_average_rejects_empty_input() {
        let cell_observables: Vec<(&FittedSimCell, Op)> = Vec::new();

        crosssection_average(&cell_observables, 0.1);
    }

    #[test]
    fn block_average_averages_bins_and_observables() {
        let profiles = test_profiles();

        let averaged = block_average(&profiles, 2).unwrap();

        assert_eq!(averaged.bins.len(), 2);
        assert_eq!(averaged.observable.len(), 2);

        assert_close(averaged.bins[0], 0.3);
        assert_close(averaged.bins[1], 10.3);

        assert_vec_approx_eq(
            averaged.observable[0].as_vector(),
            &Vector::new(4.0, 5.0, 6.0),
        );
        assert_vec_approx_eq(
            averaged.observable[1].as_vector(),
            &Vector::new(40.0, 50.0, 60.0),
        );
    }

    #[test]
    fn block_average_is_independent_of_block_count() {
        let profiles = test_profiles();

        for n_blocks in [1, 2, 4] {
            let averaged = block_average(&profiles, n_blocks).unwrap();

            assert_eq!(averaged.bins.len(), 2);
            assert_eq!(averaged.observable.len(), 2);

            assert_close(averaged.bins[0], 0.3);
            assert_close(averaged.bins[1], 10.3);

            assert_vec_approx_eq(
                averaged.observable[0].as_vector(),
                &Vector::new(4.0, 5.0, 6.0),
            );
            assert_vec_approx_eq(
                averaged.observable[1].as_vector(),
                &Vector::new(40.0, 50.0, 60.0),
            );
        }
    }

    #[test]
    fn block_average_single_profile() {
        let profiles = vec![profile(
            [1.0, 2.0],
            Vector::new(3.0, 4.0, 5.0),
            Vector::new(6.0, 7.0, 8.0),
        )];

        let averaged = block_average(&profiles, 1).unwrap();

        assert_close(averaged.bins[0], 1.0);
        assert_close(averaged.bins[1], 2.0);

        assert_vec_approx_eq(
            averaged.observable[0].as_vector(),
            &Vector::new(3.0, 4.0, 5.0),
        );
        assert_vec_approx_eq(
            averaged.observable[1].as_vector(),
            &Vector::new(6.0, 7.0, 8.0),
        );
    }

    #[test]
    fn block_average_rejects_zero_blocks() {
        let profiles = test_profiles();

        assert!(block_average(&profiles, 0).is_err());
    }

    #[test]
    fn block_average_rejects_nondivisible_blocks() {
        let profiles = test_profiles();

        assert!(block_average(&profiles, 3).is_err());
    }

    #[test]
    fn block_average_rejects_more_blocks_than_profiles() {
        let profiles = test_profiles();

        assert!(block_average(&profiles, 5).is_err());
    }

    #[test]
    #[should_panic(expected = "avg_observables is empty")]
    fn block_average_rejects_empty_profiles() {
        let profiles: Vec<BinnedObservable<Op>> = Vec::new();

        let _ = block_average(&profiles, 1);
    }

    #[test]
    #[ignore = "Profile lengths are not yet validated"]
    fn block_average_rejects_inconsistent_profile_lengths() {
        let profiles = vec![
            profile(
                [0.0, 1.0],
                Vector::new(1.0, 0.0, 0.0),
                Vector::new(2.0, 0.0, 0.0),
            ),
            BinnedObservable {
                bins: vec![0.0],
                observable: vec![op(Vector::new(3.0, 0.0, 0.0))],
            },
        ];

        assert!(block_average(&profiles, 1).is_err());
    }

    #[test]
    #[ignore = "Empty bins are not yet rejected"]
    fn cross_avg_rejects_empty_bins() {
        let items = vec![
            sample(0.0, Vector::new(1.0, 0.0, 0.0)),
        ];
        let cell_observables = refs(&items);

        let (bins, observables) =
            cross_avg(&cell_observables, vec![vec![]]);

        assert!(bins.iter().all(|x| x.is_finite()));
        assert!(observables.iter().all(|op| {
            op.as_vector().iter().all(|x| x.is_finite())
        }));
    }

    fn test_profiles() -> Vec<BinnedObservable<Op>> {
        vec![
            profile(
                [0.0, 10.0],
                Vector::new(1.0, 2.0, 3.0),
                Vector::new(10.0, 20.0, 30.0),
            ),
            profile(
                [0.2, 10.2],
                Vector::new(3.0, 4.0, 5.0),
                Vector::new(30.0, 40.0, 50.0),
            ),
            profile(
                [0.4, 10.4],
                Vector::new(5.0, 6.0, 7.0),
                Vector::new(50.0, 60.0, 70.0),
            ),
            profile(
                [0.6, 10.6],
                Vector::new(7.0, 8.0, 9.0),
                Vector::new(70.0, 80.0, 90.0),
            ),
        ]
    }

    fn profile(
        bins: [f64; 2],
        first: Vector,
        second: Vector,
    ) -> BinnedObservable<Op> {
        BinnedObservable {
            bins: bins.to_vec(),
            observable: vec![op(first), op(second)],
        }
    }

    fn sample(x: f64, value: Vector) -> (FittedSimCell, Vector) {
        (cell_at_x(x), value)
    }

    fn refs(items: &[(FittedSimCell, Vector)]) -> Vec<(&FittedSimCell, Op)> {
        items
            .iter()
            .map(|(cell, value)| (cell, op(value.clone())))
            .collect()
    }

    fn op(value: Vector) -> Op {
        Op::new(value)
    }

    fn cell_at_x(x: f64) -> FittedSimCell {
        FittedSimCell {
            unit_cell: UnitCell {
                cell_atoms: UnitCellAtoms {
                    a: Vec::new(),
                    b: vec![Atom {
                        atom_type: Some(Element::Ti),
                        position: Cartesian::new(Vector::new(x, 0.0, 0.0)),
                    }],
                    x: Vec::new(),
                },
                center_of_mass: Cartesian::new(Vector::zeros()),
            },
            alignment: CellAlignment {
                unit_quaternion: UQuaternion::identity(),
                ref_variant: RotationVariant::R0,
                error: 0.0,
            },
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        let tolerance = EPS * expected.abs().max(1.0);

        assert!(
            (actual - expected).abs() <= tolerance,
            "Expected {expected}, got {actual}"
        );
    }

    fn assert_vec_approx_eq(lhs: &Vector, rhs: &Vector) {
        let diff = lhs - rhs;

        assert!(
            diff.norm() <= EPS * rhs.norm().max(1.0),
            "lhs={lhs:?}, rhs={rhs:?}"
        );
    }
}
