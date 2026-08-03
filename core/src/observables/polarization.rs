use crate::{cell::{AsUnitCell, UnitCellOps, reference::RefCell, simulation::{FittedSimCell, SimCell}}, observables::polarization, types::{AsVector, Matrix, Vector}};
use crate::prelude::{Result, Error};

pub(crate) struct PolarizationSpec<'a> {
    bec: &'a Vec<Matrix>,
    lattice_const: f64
}

pub(crate) struct PolarizationCtx<'a, 'b> {
    fitted_ucs: &'a Vec<FittedSimCell>,
    ref_uc: &'b RefCell
}

pub(crate) struct Polarization(Vector);

impl From<Vector> for Polarization {
    fn from(value: Vector) -> Self {
        Polarization(value)
    }
}

impl AsVector for Polarization {
    fn as_vector(&self) -> &Vector {
        &self.0
    }
    
    fn as_vector_mut(&mut self) -> &mut Vector {
        &mut self.0
    }
}

pub(crate) fn calculate_pol<'a>(
    spec: &PolarizationSpec, 
    ctx: &'a PolarizationCtx
) -> Result<Vec<(&'a FittedSimCell, Polarization)>> {

    let mut res: Vec<(&FittedSimCell, Polarization)> = Vec::with_capacity(ctx.fitted_ucs.len());
    for cell in ctx.fitted_ucs {
        let polarization_per_cell = calculate_pol_for_cell(cell, ctx.ref_uc, spec.bec, spec.lattice_const)?;
        res.push(polarization_per_cell);
    }

    Ok(res)
}

fn calculate_pol_for_cell<'a>(
    fitted_uc: &'a FittedSimCell, 
    ref_uc: &RefCell,
    bec: &Vec<Matrix>,
    lattice_const: f64
) -> Result<(&'a FittedSimCell, Polarization)> {

    const ELEMENTRY_CHARGE: f64 = 1.602176634e-19;
    let unit_quaternion = fitted_uc.alignment.unit_quaternion;
    let ref_uc = ref_uc.rotated_cell(&unit_quaternion);
    let centered_fitted_uc = fitted_uc
        .centered_cell()
        .rotated_cell(&unit_quaternion.conjugate());

    let displacements =  centered_fitted_uc.displacements_from(&ref_uc)?;
    let polarization_a_site = displacements.a_site_displacements
        .iter()
        .fold(Vector::zeros(), |acc, displacement| acc + bec[0]*displacement);
    let polarization_b_site = bec[1]*displacements.b_site_displacements[0];
    let polarization_x_site = displacements.x_site_displacements
        .iter()
        .fold(Vector::zeros(), |acc, displacement| acc + bec[2]*displacement);
    
    let mut polarization = polarization_a_site + polarization_b_site + polarization_x_site;
    polarization *= ELEMENTRY_CHARGE/lattice_const.powi(3) * 1e20; // TODO converting into correct units?

    let polarization = Polarization(unit_quaternion*polarization);

    Ok((fitted_uc, polarization))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cell::{reference::RefCellBuilder, CellGeometry},
        types::{CellAlignment, Element, RotationVariant, UQuaternion},
    };

    const EPS: f64 = 1.0e-12;

    #[test]
    fn calculate_pol_for_matching_cells_returns_zero() {
        let ref_uc = reference_cell();
        let fitted_uc = fitted_from_ref(&ref_uc);
        let bec = isotropic_bec(1.0, 1.0, 1.0);

        let (_, polarization) = calculate_pol_for_cell(&fitted_uc, &ref_uc, &bec, 1.0).unwrap();

        assert_vector_approx_eq(polarization.0, Vector::zeros(), EPS);
    }

    #[test]
    fn calculate_pol_for_single_b_site_displacement_uses_bec_and_volume_conversion() {
        let ref_uc = reference_cell();
        let mut fitted_uc = fitted_from_ref(&ref_uc);
        let displacement = Vector::new(0.01, 0.0, 0.0);
        fitted_uc.unit_cell.cell_atoms.b[0].position += displacement;

        let bec = isotropic_bec(1.0, 2.0, 1.0);
        let (_, polarization) = calculate_pol_for_cell(&fitted_uc, &ref_uc, &bec, 1.0).unwrap();

        let expected = 2.0 * displacement * 1.602176634e-19 * 1e20;
        assert_vector_approx_eq(polarization.0, expected, EPS);
    }

    #[test]
    fn calculate_pol_returns_one_result_per_fitted_cell() {
        let ref_uc = reference_cell();
        let fitted_ucs = vec![fitted_from_ref(&ref_uc), fitted_from_ref(&ref_uc)];
        let bec = isotropic_bec(1.0, 1.0, 1.0);
        let spec = PolarizationSpec {
            bec: &bec,
            lattice_const: 1.0,
        };
        let ctx = PolarizationCtx {
            fitted_ucs: &fitted_ucs,
            ref_uc: &ref_uc,
        };

        let results = calculate_pol(&spec, &ctx).unwrap();

        assert_eq!(results.len(), fitted_ucs.len());
    }

    fn reference_cell() -> RefCell {
        RefCellBuilder::new()
            .geometry(CellGeometry::Cubic)
            .lattice_constant(1.0)
            .elements(vec![Element::Sr, Element::Ti, Element::O])
            .rotation_variant(RotationVariant::R0)
            .build()
            .unwrap()
    }

    fn fitted_from_ref(ref_uc: &RefCell) -> FittedSimCell {
        FittedSimCell {
            unit_cell: ref_uc.as_unit_cell().clone(),
            alignment: CellAlignment {
                unit_quaternion: UQuaternion::identity(),
                ref_variant: RotationVariant::R0,
                error: 0.0,
            },
        }
    }

    fn isotropic_bec(a: f64, b: f64, x: f64) -> Vec<Matrix> {
        vec![
            Matrix::identity() * a,
            Matrix::identity() * b,
            Matrix::identity() * x,
        ]
    }

    fn assert_vector_approx_eq(lhs: Vector, rhs: Vector, eps: f64) {
        let diff = lhs - rhs;
        assert!(diff.norm() <= eps, "lhs={lhs:?}, rhs={rhs:?}");
    }
}
