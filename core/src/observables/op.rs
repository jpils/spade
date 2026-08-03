// TODO implement different functions based on the geometry

use std::collections::VecDeque;
use std::ops::Add;

use crate::cell::simulation::FittedSimCell;
use crate::cell::{AsUnitCell, CellGeometry, UnitCellOps, reference};
use crate::cell::reference::{RefCell, RefCellBuilder};
use crate::geometry::NearestNeighbors;
use crate::prelude::*;
use crate::types::{AsVector, Atom, Atoms, Elements};
use crate::{cell::UnitCell, types::{PhaseFactor, AtomDisplacements, Vector}};
use super::Observable;
use nalgebra::{ComplexField, SVector, Unit};

type Vector15 = SVector<f64, 15>;

pub(crate) struct OpSpec {
    lattice_const: f64,
    elements: Elements,
    geometry: CellGeometry,
}

pub(crate) struct OpCtx<'a, 'b, 'c> {
    fitted_ucs: Vec<&'a FittedSimCell>,
    ref_uc: &'b RefCell,
    b_site_nearest_neighbors: &'c Vec<NearestNeighbors>
}

#[derive(Debug)]
pub(crate) struct Op(Vector);

impl Add for Op {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let res = self.0 + rhs.0; 
        Self(res)
    }
}

impl Observable for Op {}

impl Op {
    pub fn new(vec: Vector) -> Self {
        Self(vec)
    }
}

impl From<Vector> for Op {
    fn from(value: Vector) -> Self {
        Op(value)
    }
}

impl AsVector for Op {
    fn as_vector(&self) -> &Vector {
        &self.0
    }

    fn as_vector_mut(&mut self) -> &mut Vector {
        &mut self.0
    }
}

pub(crate) fn calculate_op<'a>(spec: &OpSpec, ctx: &'a OpCtx) -> Result<Vec<(&'a FittedSimCell, Op)>> {
    let mut res: Vec<(&FittedSimCell, Op)> = Vec::with_capacity(ctx.fitted_ucs.len());
    let phase_factors = phase_factors(ctx.b_site_nearest_neighbors)?;

    for (&cell, &phase_factor) in ctx.fitted_ucs.iter().zip(phase_factors.iter()) {
        let op_per_cell = calculate_op_for_cell(cell, ctx.ref_uc, phase_factor, spec)?;
        res.push(op_per_cell);
    }
    Ok(res)
}

fn phase_factors(b_nearest_neighbors: &[NearestNeighbors]) -> Result<Vec<PhaseFactor>> {
    if b_nearest_neighbors.is_empty() {
        return Err(Error::Generic("b_nearest_neighbors is is_empty".into()));
    }

    let mut phase_factors: Vec<PhaseFactor> = vec![PhaseFactor::Undefined; b_nearest_neighbors.len()];
    let mut b_site_id_queue = VecDeque::from([0]);

    phase_factors[0] = PhaseFactor::Positive;
    while let Some(current_parent_id) = b_site_id_queue.pop_front() {
        for neighbor in &b_nearest_neighbors[current_parent_id].neighbors {
            let current_child_id = find_id(&neighbor.atom_abs_cart, b_nearest_neighbors)?;

            match phase_factors[current_child_id] {
                PhaseFactor::Undefined => {
                    b_site_id_queue.push_back(current_child_id);
                    phase_factors[current_child_id] = 
                        phase_factors[current_parent_id].get_flipped()?;
                }
                _ => if phase_factors[current_child_id] == phase_factors[current_parent_id] {
                    return Err(Error::Generic("Neighboring UCs have the same phase factor".into()));
                }
            }
        }
    }

    Ok(phase_factors)
}

fn find_id(atom: &Atom, nearest_neighbors: &[NearestNeighbors]) -> Result<usize> {
    nearest_neighbors.iter()
        .position(|neighbor| neighbor.reference == *atom)
        .ok_or_else(|| Error::Generic("Could not find atom in provided list!".into()))
}

//TODO log this to file with async
fn calculate_op_for_cell<'a>(
    fitted_uc: &'a FittedSimCell, 
    ref_uc: &RefCell,
    phase_factor: PhaseFactor, 
    spec: &OpSpec
) -> Result<(&'a FittedSimCell, Op)> {

    let unit_quaternion = fitted_uc.alignment.unit_quaternion;
    
    let phonon_pol_x = Vector15::from_row_slice(&[
        0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0,
        0.0, 0.0, -1.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0,
    ]).normalize();
    let phonon_pol_y = Vector15::from_row_slice(&[
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        -1.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
    ]).normalize();
    let phonon_pol_z = Vector15::from_row_slice(&[
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
    ]).normalize();

    let ref_uc = ref_uc.rotated_cell(&unit_quaternion);
    let centered_fitted_uc = fitted_uc
        .centered_cell()
        .rotated_cell(&unit_quaternion.conjugate());

    let mass_weighted_disp = match spec.geometry {
        CellGeometry::Cubic => {
            let total_mass: f64 = spec.elements.iter().map(|e| e.get_mass()).sum();

            let mut displacements = ref_uc.displacements_from(&centered_fitted_uc)?;
            displacements.a
                .iter_mut()
                .for_each(|d| *d *= (spec.elements[0].get_mass() / total_mass).sqrt());
            displacements.b
                .iter_mut()
                .for_each(|d| *d *= (spec.elements[1].get_mass() / total_mass).sqrt());
            displacements.x
                .iter_mut()
                .for_each(|d| *d *= (spec.elements[2].get_mass() / total_mass).sqrt());

            Vector15::from_row_iterator(
                displacements.a[0]
                .as_slice()
                .iter()
                .chain(displacements.b[0].as_slice())
                .chain(displacements.x[0].as_slice())
                .chain(displacements.x[1].as_slice())
                .chain(displacements.x[2].as_slice())
                .copied()
            )
        },
        CellGeometry::Tetragonal => todo!()
    };

    let phi1 = mass_weighted_disp.dot(&phonon_pol_x);
    let phi2 = mass_weighted_disp.dot(&phonon_pol_y);
    let phi3 = mass_weighted_disp.dot(&phonon_pol_z);

    let phase_factor = phase_factor.try_to_int()?;

    let phi = Vector::new(phi1, phi2, phi3);
    let phi = Op(unit_quaternion*(f64::from(phase_factor)* phi));

    Ok((fitted_uc, phi))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Atoms, Matrix};

    const EPS: f64 = 1e-12;

    fn test_cell() -> Matrix {
        Matrix::from_diagonal(&Vector::new(10.0, 10.0, 10.0))
    }

    fn atom(position: Vector) -> Atom {
        Atom {
            atom_type: None,
            position,
        }
    }

    fn graph(
        positions: &[Vector],
        neighbors_per_site: usize,
    ) -> Vec<NearestNeighbors> {
        let atoms: Atoms = positions
            .iter()
            .copied()
            .map(atom)
            .collect();

        crate::geometry::n_nearest_neighbors(
            &atoms,
            &atoms,
            neighbors_per_site,
            &test_cell(),
        )
    }

    fn square_graph() -> Vec<NearestNeighbors> {
        graph(
            &[
                Vector::new(0.25, 0.25, 0.5),
                Vector::new(0.75, 0.25, 0.5),
                Vector::new(0.75, 0.75, 0.5),
                Vector::new(0.25, 0.75, 0.5),
            ],
            2,
        )
    }

    #[test]
    fn test_op_new() {
        let expected = Vector::new(1.0, -2.0, 3.0);
        let op = Op::new(expected);

        assert!((op.as_vector() - expected).norm() < EPS);
    }

    #[test]
    fn test_op_from_vector() {
        let expected = Vector::new(1.0, 2.0, 3.0);
        let op = Op::from(expected);

        assert!((op.as_vector() - expected).norm() < EPS);
    }

    #[test]
    fn test_op_add() {
        let lhs = Op::new(Vector::new(1.0, 2.0, 3.0));
        let rhs = Op::new(Vector::new(4.0, -1.0, 2.0));

        let result = lhs + rhs;
        let expected = Vector::new(5.0, 1.0, 5.0);

        assert!((result.as_vector() - expected).norm() < EPS);
    }

    #[test]
    fn test_op_as_vector_mut() {
        let mut op = Op::new(Vector::zeros());

        *op.as_vector_mut() = Vector::new(1.0, 2.0, 3.0);

        let expected = Vector::new(1.0, 2.0, 3.0);

        assert!((op.as_vector() - expected).norm() < EPS);
    }

    #[test]
    fn test_find_id_existing_atoms() {
        let neighbors = square_graph();

        assert_eq!(neighbors.len(), 4);

        for (expected_id, entry) in neighbors.iter().enumerate() {
            let id = find_id(&entry.reference, &neighbors).unwrap();

            assert_eq!(id, expected_id);
        }
    }

    #[test]
    fn test_find_id_missing_atom() {
        let neighbors = square_graph();

        let unknown = atom(Vector::new(99.0, 99.0, 99.0));

        assert!(find_id(&unknown, &neighbors).is_err());
    }

    #[test]
    fn test_find_id_empty_list() {
        let neighbors: Vec<NearestNeighbors> = Vec::new();

        let target = atom(Vector::zeros());

        assert!(find_id(&target, &neighbors).is_err());
    }

    #[test]
    fn test_phase_factors_two_sites() {
        let neighbors = graph(
            &[
                Vector::new(0.1, 0.5, 0.5),
                Vector::new(0.4, 0.5, 0.5),
            ],
            1,
        );

        let factors = phase_factors(&neighbors).unwrap();

        assert_eq!(factors.len(), 2);
        assert!(factors[0] == PhaseFactor::Positive);
        assert!(factors[1] == PhaseFactor::Negative);
    }

    #[test]
    fn test_phase_factors_bipartite_cycle() {
        let neighbors = square_graph();

        let factors = phase_factors(&neighbors).unwrap();

        assert_eq!(factors.len(), 4);

        assert!(factors[0] == PhaseFactor::Positive);
        assert!(factors[1] == PhaseFactor::Negative);
        assert!(factors[2] == PhaseFactor::Positive);
        assert!(factors[3] == PhaseFactor::Negative);
    }

    #[test]
    fn test_phase_factors_neighbor_consistency() {
        let neighbors = square_graph();

        let factors = phase_factors(&neighbors).unwrap();

        for (parent_id, entry) in neighbors.iter().enumerate() {
            for neighbor in &entry.neighbors {
                let child_id = find_id(
                    &neighbor.atom_abs_cart,
                    &neighbors,
                )
                .unwrap();

                assert_ne!(parent_id, child_id);

                assert!(
                    factors[parent_id] != factors[child_id],
                    "Neighboring sites {parent_id} and {child_id} have equal phase factors"
                );
            }
        }
    }

    #[test]
    fn test_phase_factors_odd_cycle() {
        let neighbors = graph(
            &[
                Vector::new(0.1, 0.1, 0.5),
                Vector::new(0.3, 0.1, 0.5),
                Vector::new(0.2, 0.3, 0.5),
            ],
            2,
        );

        assert!(phase_factors(&neighbors).is_err());
    }

    #[test]
    fn test_phase_factors_disconnected_graph() {
        let neighbors = graph(
            &[
                Vector::new(0.10, 0.1, 0.1),
                Vector::new(0.12, 0.1, 0.1),
                Vector::new(0.60, 0.6, 0.6),
                Vector::new(0.62, 0.6, 0.6),
            ],
            1,
        );

        let factors = phase_factors(&neighbors).unwrap();

        assert_eq!(factors.len(), 4);

        assert!(factors[0] == PhaseFactor::Positive);
        assert!(factors[1] == PhaseFactor::Negative);
        assert!(factors[2] == PhaseFactor::Undefined);
        assert!(factors[3] == PhaseFactor::Undefined);
    }

    #[test]
    fn test_phase_factors_empty_graph() {
        let neighbors: Vec<NearestNeighbors> = Vec::new();

        assert!(phase_factors(&neighbors).is_err());
    }

    #[test]
    fn test_phase_factors_missing_neighbor_reference() {
        let references = vec![
            atom(Vector::new(0.1, 0.5, 0.5)),
        ];

        let candidates = vec![
            atom(Vector::new(0.3, 0.5, 0.5)),
        ];

        let neighbors = crate::geometry::n_nearest_neighbors(
            &references,
            &candidates,
            1,
            &test_cell(),
        );

        assert!(phase_factors(&neighbors).is_err());
    }

    #[test]
    fn test_phase_factors_single_isolated_site() {
        let neighbors = graph(
            &[Vector::new(0.25, 0.5, 0.5)],
            0,
        );

        let factors = phase_factors(&neighbors).unwrap();

        assert_eq!(factors.len(), 1);
        assert!(factors[0] == PhaseFactor::Positive);
    }

    #[test]
    fn test_phase_factors_deterministic() {
        let neighbors = square_graph();

        let first = phase_factors(&neighbors).unwrap();
        let second = phase_factors(&neighbors).unwrap();

        assert_eq!(first.len(), second.len());

        for (a, b) in first.iter().zip(second.iter()) {
            assert!(a == b);
        }
    }
}
