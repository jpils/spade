use crate::{cell::simulation::FittedSimCell, types::AsVector};
use crate::prelude::{Result, Error};
use crate::types::Vector;

/// averages observable over z-y crosssection and returns observable profile along x axis
pub(crate) fn crosssection_average<Obs>(
    cell_observables: &Vec<(&FittedSimCell, Obs)>,
    threshold: f64
) -> Vec<Obs> 
where 
    Obs: AsVector + From<Vector> 
{
    if cell_observables.is_empty() {
        panic!("Bug: Crosssection avereraging is called but observables are empty!");
    }

    let cell_id_bins = bins_along_x(cell_observables, threshold);
    let avg_observables = cross_avg(cell_observables, cell_id_bins);

    avg_observables
}

pub(crate) fn block_average<Obs>( // TODO check function again & refactor
    avg_observables: &Vec<Vec<Obs>>,
    n_blocks: usize
) -> Result<Vec<Obs>>
where
    Obs: AsVector + From<Vector>
{
    if avg_observables.is_empty() {
        panic!("avg_observables is empty");
    }

    if n_blocks == 0 {
        return Err(Error::Generic("Number of blocks is set to 0".into()))
    }

    if avg_observables.len() % n_blocks != 0 {
        return Err(Error::Generic("Length of avg_observables not divisible by n_blocks".into()))
    }     

    let block_size = avg_observables.len() / n_blocks;
    let obs_len = avg_observables[0].len();

    let mut blocks: Vec<Vec<Obs>> = Vec::with_capacity(n_blocks);
    for i in (0..avg_observables.len()).step_by(block_size) {
        let mut block: Vec<Obs> = Vec::with_capacity(obs_len);
        for k in 0..obs_len {
            let mut obs = Vector::zeros();
            for j in 0..block_size {
                obs += avg_observables[i+j][k].as_vector();
            }

            let obs = Obs::from(obs / block_size as f64);
            block.push(obs);
        }

        blocks.push(block);
    }

    let mut avg_obs: Vec<Obs> = Vec::with_capacity(obs_len);
    for k in 0..obs_len {
        let mut obs = Vector::zeros();
        for i in 0..blocks.len() {
            obs += blocks[i][k].as_vector();
        }

        let obs = Obs::from(obs / blocks.len() as f64);
        avg_obs.push(obs);
    }

    Ok(avg_obs)
}


fn bins_along_x<Obs: AsVector>(
    cell_observables: &Vec<(&FittedSimCell, Obs)>,
    threshold: f64
) -> Vec<Vec<usize>> {

    let ids = argsort(cell_observables);

    find_and_fill_bins(cell_observables, ids, threshold)
}

fn argsort<Obs: AsVector>(cell_observables: &Vec<(&FittedSimCell, Obs)>) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..cell_observables.len()).collect();

    ids.sort_by(|&a, &b| { 
        cell_observables[a].0.unit_cell.cell_atoms.b[0].position.x.total_cmp(
            &cell_observables[b].0.unit_cell.cell_atoms.b[0].position.x
        )
    });

    ids
}

fn find_and_fill_bins<Obs: AsVector>(
    cell_observables: &Vec<(&FittedSimCell, Obs)>,
    ids: Vec<usize>,
    threshold: f64
) -> Vec<Vec<usize>> {

    let cell_x_at = |id: usize| { 
        cell_observables[id].0.unit_cell.cell_atoms.b[0].position.x
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

fn cross_avg<Obs>(
    cell_observables: &Vec<(&FittedSimCell, Obs)>,
    cell_id_bins: Vec<Vec<usize>>
) -> Vec<Obs> 
where
    Obs: AsVector + From<Vector>
{
    let mut avg_obs: Vec<Obs> = Vec::with_capacity(cell_id_bins.len());
    for bin in cell_id_bins {
        let obs: Vector = bin.iter()
            .fold(Vector::zeros(), |acc, &id| acc + cell_observables[id].1.as_vector())
            .map(|x| x / bin.len() as f64);

        let obs = Obs::from(obs);
        avg_obs.push(obs);
    }

    avg_obs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cell::{UnitCell, UnitCellAtoms},
        types::{Atom, CellAlignment, Element, RotationVariant, UQuaternion},
    };

    const EPS: f64 = 1.0e-12;

    #[derive(Clone, Debug)]
    struct TestObs(Vector);

    impl From<Vector> for TestObs {
        fn from(value: Vector) -> Self {
            Self(value)
        }
    }

    impl AsVector for TestObs {
        fn as_vector(&self) -> &Vector {
            &self.0
        }

        fn as_vector_mut(&mut self) -> &mut Vector {
            &mut self.0
        }
    }

    #[test]
    fn argsort_orders_observables_by_b_site_x_position() {
        let cell_observables = vec![
            (cell_at_x(2.0), obs(20.0)),
            (cell_at_x(0.0), obs(0.0)),
            (cell_at_x(1.0), obs(10.0)),
        ];
        let refs = refs(&cell_observables);

        let ids = argsort(&refs);

        assert_eq!(ids, vec![1, 2, 0]);
    }

    #[test]
    fn find_and_fill_bins_groups_sorted_ids_within_threshold() {
        let cell_observables = vec![
            (cell_at_x(0.00), obs(0.0)),
            (cell_at_x(0.05), obs(1.0)),
            (cell_at_x(1.00), obs(2.0)),
            (cell_at_x(1.04), obs(3.0)),
        ];
        let refs = refs(&cell_observables);

        let bins = find_and_fill_bins(&refs, vec![0, 1, 2, 3], 0.1);

        assert_eq!(bins, vec![vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn bins_along_x_sorts_then_groups_cells() {
        let cell_observables = vec![
            (cell_at_x(1.04), obs(3.0)),
            (cell_at_x(0.05), obs(1.0)),
            (cell_at_x(1.00), obs(2.0)),
            (cell_at_x(0.00), obs(0.0)),
        ];
        let refs = refs(&cell_observables);

        let bins = bins_along_x(&refs, 0.1);

        assert_eq!(bins, vec![vec![3, 1], vec![2, 0]]);
    }

    #[test]
    fn cross_avg_averages_observables_inside_each_bin() {
        let cell_observables = vec![
            (cell_at_x(0.0), obs_vec(Vector::new(1.0, 2.0, 3.0))),
            (cell_at_x(0.0), obs_vec(Vector::new(3.0, 4.0, 5.0))),
            (cell_at_x(1.0), obs_vec(Vector::new(10.0, 20.0, 30.0))),
        ];
        let refs = refs(&cell_observables);

        let averaged = cross_avg(&refs, vec![vec![0, 1], vec![2]]);

        assert_eq!(averaged.len(), 2);
        assert_vec_approx_eq(averaged[0].as_vector(), &Vector::new(2.0, 3.0, 4.0));
        assert_vec_approx_eq(averaged[1].as_vector(), &Vector::new(10.0, 20.0, 30.0));
    }

    #[test]
    fn crosssection_average_groups_by_x_and_returns_bin_averages() {
        let cell_observables = vec![
            (cell_at_x(0.00), obs_vec(Vector::new(1.0, 0.0, 0.0))),
            (cell_at_x(0.05), obs_vec(Vector::new(3.0, 0.0, 0.0))),
            (cell_at_x(1.00), obs_vec(Vector::new(10.0, 0.0, 0.0))),
        ];
        let refs = refs(&cell_observables);

        let averaged = crosssection_average(&refs, 0.1);

        assert_eq!(averaged.len(), 2);
        assert_vec_approx_eq(averaged[0].as_vector(), &Vector::new(2.0, 0.0, 0.0));
        assert_vec_approx_eq(averaged[1].as_vector(), &Vector::new(10.0, 0.0, 0.0));
    }

    #[test]
    fn block_average_averages_equal_sized_blocks_then_averages_blocks() {
        let profiles = vec![
            vec![obs_vec(Vector::new(1.0, 0.0, 0.0)), obs_vec(Vector::new(10.0, 0.0, 0.0))],
            vec![obs_vec(Vector::new(3.0, 0.0, 0.0)), obs_vec(Vector::new(30.0, 0.0, 0.0))],
            vec![obs_vec(Vector::new(5.0, 0.0, 0.0)), obs_vec(Vector::new(50.0, 0.0, 0.0))],
            vec![obs_vec(Vector::new(7.0, 0.0, 0.0)), obs_vec(Vector::new(70.0, 0.0, 0.0))],
        ];

        let averaged = block_average(&profiles, 2).unwrap();

        assert_eq!(averaged.len(), 2);
        assert_vec_approx_eq(averaged[0].as_vector(), &Vector::new(4.0, 0.0, 0.0));
        assert_vec_approx_eq(averaged[1].as_vector(), &Vector::new(40.0, 0.0, 0.0));
    }

    #[test]
    fn block_average_errors_when_profiles_do_not_divide_into_blocks() {
        let profiles = vec![
            vec![obs(1.0)],
            vec![obs(2.0)],
            vec![obs(3.0)],
        ];

        assert!(block_average(&profiles, 2).is_err());
    }

    fn refs(items: &[(FittedSimCell, TestObs)]) -> Vec<(&FittedSimCell, TestObs)> {
        items.iter().map(|(cell, obs)| (cell, obs.clone())).collect()
    }

    fn cell_at_x(x: f64) -> FittedSimCell {
        FittedSimCell {
            unit_cell: UnitCell {
                cell_atoms: UnitCellAtoms {
                    a: Vec::new(),
                    b: vec![Atom { atom_type: Some(Element::Ti), position: Vector::new(x, 0.0, 0.0) }],
                    x: Vec::new(),
                },
                center_of_mass: Vector::zeros(),
            },
            alignment: CellAlignment {
                unit_quaternion: UQuaternion::identity(),
                ref_variant: RotationVariant::R0,
                error: 0.0,
            },
        }
    }

    fn obs(value: f64) -> TestObs {
        obs_vec(Vector::new(value, 0.0, 0.0))
    }

    fn obs_vec(value: Vector) -> TestObs {
        TestObs(value)
    }

    fn assert_vec_approx_eq(lhs: &Vector, rhs: &Vector) {
        let diff = lhs - rhs;
        assert!(diff.norm() <= EPS, "lhs={lhs:?}, rhs={rhs:?}");
    }
}
