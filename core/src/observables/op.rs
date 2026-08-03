use crate::cell::simulation::FittedSimCell;
use crate::cell::{AsUnitCell, CellGeometry, UnitCellOps, reference};
use crate::cell::reference::{RefCell, RefCellBuilder};
use crate::prelude::*;
use crate::types::{AsVector, Elements};
use crate::{cell::UnitCell, types::{PhaseFactor, AtomDisplacements, Vector}};
use super::Observable;
use nalgebra::{ComplexField, SVector, Unit};

type Vector15 = SVector<f64, 15>;

pub(crate) struct OpSpec {
    lattice_const: f64,
    elements: Elements,
    geometry: CellGeometry,
}

pub(crate) struct OpCtx<'a, 'b> {
    fitted_ucs: Vec<&'a FittedSimCell>,
    ref_uc: &'b RefCell,
    phase_factors: Vec<PhaseFactor>,
}

#[derive(Debug)]
pub(crate) struct Op(Vector);

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
    for (&cell, &phase_factor) in ctx.fitted_ucs.iter().zip(ctx.phase_factors.iter()) {
        let op_per_cell = calculate_op_for_cell(cell, ctx.ref_uc, phase_factor, spec)?;
        res.push(op_per_cell);
    }
    Ok(res)
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
            displacements.a_site_displacements
                .iter_mut()
                .for_each(|d| *d *= (spec.elements[0].get_mass() / total_mass).sqrt());
            displacements.b_site_displacements
                .iter_mut()
                .for_each(|d| *d *= (spec.elements[1].get_mass() / total_mass).sqrt());
            displacements.x_site_displacements
                .iter_mut()
                .for_each(|d| *d *= (spec.elements[2].get_mass() / total_mass).sqrt());

            Vector15::from_row_iterator(
                displacements.a_site_displacements[0]
                .as_slice()
                .iter()
                .chain(displacements.b_site_displacements[0].as_slice())
                .chain(displacements.x_site_displacements[0].as_slice())
                .chain(displacements.x_site_displacements[1].as_slice())
                .chain(displacements.x_site_displacements[2].as_slice())
                .copied()
            )
        },
        CellGeometry::Tetragonal => todo!()
    };

    let phi1 = mass_weighted_disp.dot(&phonon_pol_x);
    let phi2 = mass_weighted_disp.dot(&phonon_pol_y);
    let phi3 = mass_weighted_disp.dot(&phonon_pol_z);

    let phase_factor = phase_factor.to_int();

    let phi = Vector::new(phi1, phi2, phi3);
    let phi = Op(unit_quaternion*(phase_factor as f64 * phi));

    Ok((fitted_uc, phi))
}

// TODO implement different functions based on the geometry
