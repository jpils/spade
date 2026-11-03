use std::ops::Add;

use crate::{cell::{AsUnitCell, UnitCellOps, reference::RefCell, simulation::{FittedSimCell, SimCell}}, observables::polarization, types::{AsVector, Matrix, Vector}};
use crate::prelude::{Result, Error};
use super::Observable;

pub struct PolarizationSpec<'a> {
    bec: &'a Vec<Matrix>,
    lattice_const: f64
}

pub struct PolarizationCtx<'a, 'b> {
    fitted_ucs: &'a Vec<FittedSimCell>,
    ref_uc: &'b RefCell
}

#[derive(Clone, Debug)]
pub struct Polarization(Vector);

impl Add for Polarization {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let res = self.0 + rhs.0;

        Polarization(res)
    }
}

impl Observable for Polarization {}

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

pub fn calculate_pol<'a>(
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
    bec: &[Matrix],
    lattice_const: f64
) -> Result<(&'a FittedSimCell, Polarization)> {

    const ELEMENTRY_CHARGE: f64 = 1.602_176_634e-19;
    let unit_quaternion = fitted_uc.alignment.unit_quaternion;
    let ref_uc = ref_uc.rotated_cell(&unit_quaternion);
    let centered_fitted_uc = fitted_uc
        .centered_cell()
        .rotated_cell(&unit_quaternion.conjugate());

    let displacements =  centered_fitted_uc.displacements_from(&ref_uc)?;
    let pol_a = displacements.a
        .iter()
        .fold(Vector::zeros(), |acc, displacement| acc + bec[0]*displacement.as_vector());
    let pol_b = bec[1]*displacements.b[0].as_vector();
    let pol_x = displacements.x
        .iter()
        .fold(Vector::zeros(), |acc, displacement| acc + bec[2]*displacement.as_vector());
    
    let mut polarization = pol_a + pol_b + pol_x;
    polarization *= ELEMENTRY_CHARGE/lattice_const.powi(3) * 1e20; // TODO converting into correct units?

    let polarization = Polarization(unit_quaternion*polarization);

    Ok((fitted_uc, polarization))
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cell::{reference::RefCellBuilder, CellGeometry},
        types::{Cartesian, CellAlignment, Element, RotationVariant, UQuaternion},
    };

    const EPS: f64 = 1e-12;
    const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;
    const CONVERSION: f64 = ELEMENTARY_CHARGE * 1e20;

    fn assert_vector_approx_eq(actual: &Vector, expected: &Vector) {
        let tolerance = EPS * expected.norm().max(1.0);

        assert!(
            (actual - expected).norm() <= tolerance,
            "Expected {expected:?}, got {actual:?}"
        );
    }

    #[test]
    fn polarization_from_vector() {
        let expected = Vector::new(1.0, -2.0, 3.0);
        let polarization = Polarization::from(expected);

        assert_vector_approx_eq(polarization.as_vector(), &expected);
    }

    #[test]
    fn polarization_add() {
        let lhs = Polarization::from(Vector::new(1.0, 2.0, 3.0));
        let rhs = Polarization::from(Vector::new(4.0, -1.0, 2.0));

        let result = lhs + rhs;
        let expected = Vector::new(5.0, 1.0, 5.0);

        assert_vector_approx_eq(result.as_vector(), &expected);
    }

    #[test]
    fn polarization_as_vector_mut() {
        let mut polarization = Polarization::from(Vector::zeros());

        *polarization.as_vector_mut() = Vector::new(1.0, 2.0, 3.0);

        let expected = Vector::new(1.0, 2.0, 3.0);

        assert_vector_approx_eq(polarization.as_vector(), &expected);
    }

    #[test]
    fn calculate_pol_for_matching_cells_returns_zero() {
        let ref_uc = reference_cell();
        let fitted_uc = fitted_from_ref(&ref_uc);
        let bec = isotropic_bec(1.0, 2.0, -2.0);

        let (_, polarization) =
            calculate_pol_for_cell(&fitted_uc, &ref_uc, &bec, 1.0)
                .unwrap();

        assert_vector_approx_eq(
            polarization.as_vector(),
            &Vector::zeros(),
        );
    }

    #[test]
    fn calculate_pol_for_b_site_displacement() {
        let ref_uc = reference_cell();
        let mut fitted_uc = fitted_from_ref(&ref_uc);

        let displacement = Vector::new(0.01, 0.0, 0.0);
        fitted_uc.unit_cell.cell_atoms.b[0].position += Cartesian::new(displacement);

        let bec = isotropic_bec(0.0, 2.0, 0.0);

        let (_, polarization) =
            calculate_pol_for_cell(&fitted_uc, &ref_uc, &bec, 1.0)
                .unwrap();

        let expected = displacement * 2.0 * CONVERSION;

        assert_vector_approx_eq(polarization.as_vector(), &expected);
    }

    #[test]
    fn calculate_pol_for_x_site_displacement() {
        let ref_uc = reference_cell();
        let mut fitted_uc = fitted_from_ref(&ref_uc);

        let displacement = Vector::new(0.0, 0.02, 0.0);
        fitted_uc.unit_cell.cell_atoms.x[0].position += Cartesian::new(displacement);

        let bec = isotropic_bec(0.0, 0.0, -2.0);

        let (_, polarization) =
            calculate_pol_for_cell(&fitted_uc, &ref_uc, &bec, 1.0)
                .unwrap();

        let expected = displacement * -2.0 * CONVERSION;

        assert_vector_approx_eq(polarization.as_vector(), &expected);
    }

    #[test]
    fn calculate_pol_uses_full_bec_tensor() {
        let ref_uc = reference_cell();
        let mut fitted_uc = fitted_from_ref(&ref_uc);

        let displacement = Vector::new(0.01, 0.0, 0.0);
        fitted_uc.unit_cell.cell_atoms.b[0].position += Cartesian::new(displacement);

        let tensor = Matrix::new(
            0.0, 2.0, 0.0,
            3.0, 0.0, 0.0,
            0.0, 0.0, 4.0,
        );

        let bec = vec![
            Matrix::zeros(),
            tensor,
            Matrix::zeros(),
        ];

        let (_, polarization) =
            calculate_pol_for_cell(&fitted_uc, &ref_uc, &bec, 1.0)
                .unwrap();

        let expected = tensor * displacement * CONVERSION;

        assert_vector_approx_eq(polarization.as_vector(), &expected);
    }

    #[test]
    fn calculate_pol_zero_bec_returns_zero() {
        let ref_uc = reference_cell();
        let mut fitted_uc = fitted_from_ref(&ref_uc);

        fitted_uc.unit_cell.cell_atoms.b[0].position +=
            Cartesian::new(Vector::new(0.01, -0.02, 0.03));

        let bec = isotropic_bec(0.0, 0.0, 0.0);

        let (_, polarization) =
            calculate_pol_for_cell(&fitted_uc, &ref_uc, &bec, 1.0)
                .unwrap();

        assert_vector_approx_eq(
            polarization.as_vector(),
            &Vector::zeros(),
        );
    }

    #[test]
    fn calculate_pol_scales_linearly_with_bec() {
        let ref_uc = reference_cell();
        let mut fitted_uc = fitted_from_ref(&ref_uc);

        fitted_uc.unit_cell.cell_atoms.b[0].position +=
            Cartesian::new(Vector::new(0.01, 0.02, -0.01));

        let bec1 = isotropic_bec(1.0, 2.0, -2.0);
        let bec2 = isotropic_bec(3.0, 6.0, -6.0);

        let (_, pol1) =
            calculate_pol_for_cell(&fitted_uc, &ref_uc, &bec1, 1.0)
                .unwrap();

        let (_, pol2) =
            calculate_pol_for_cell(&fitted_uc, &ref_uc, &bec2, 1.0)
                .unwrap();

        let expected = pol1.as_vector() * 3.0;

        assert_vector_approx_eq(pol2.as_vector(), &expected);
    }

    #[test]
    fn calculate_pol_scales_with_inverse_volume() {
        let ref_uc = reference_cell();
        let mut fitted_uc = fitted_from_ref(&ref_uc);

        fitted_uc.unit_cell.cell_atoms.b[0].position +=
            Cartesian::new(Vector::new(0.01, 0.02, -0.01));

        let bec = isotropic_bec(1.0, 2.0, -2.0);

        let (_, pol1) =
            calculate_pol_for_cell(&fitted_uc, &ref_uc, &bec, 1.0)
                .unwrap();

        let (_, pol2) =
            calculate_pol_for_cell(&fitted_uc, &ref_uc, &bec, 2.0)
                .unwrap();

        let expected = pol1.as_vector() / 8.0;

        assert_vector_approx_eq(pol2.as_vector(), &expected);
    }

    #[test]
    fn calculate_pol_preserves_fitted_cell_reference() {
        let ref_uc = reference_cell();
        let fitted_uc = fitted_from_ref(&ref_uc);
        let bec = isotropic_bec(1.0, 2.0, -2.0);

        let (returned_cell, _) =
            calculate_pol_for_cell(&fitted_uc, &ref_uc, &bec, 1.0)
                .unwrap();

        assert!(std::ptr::eq(returned_cell, &fitted_uc));
    }

    #[test]
    fn calculate_pol_returns_one_result_per_cell() {
        let ref_uc = reference_cell();

        let fitted_ucs = vec![
            fitted_from_ref(&ref_uc),
            fitted_from_ref(&ref_uc),
            fitted_from_ref(&ref_uc),
        ];

        let bec = isotropic_bec(1.0, 2.0, -2.0);

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

        for (i, (cell, polarization)) in results.iter().enumerate() {
            assert!(std::ptr::eq(*cell, &fitted_ucs[i]));

            assert_vector_approx_eq(
                polarization.as_vector(),
                &Vector::zeros(),
            );
        }
    }

    #[test]
    fn calculate_pol_preserves_cell_order() {
        let ref_uc = reference_cell();

        let cell0 = fitted_from_ref(&ref_uc);

        let mut cell1 = fitted_from_ref(&ref_uc);
        cell1.unit_cell.cell_atoms.b[0].position +=
            Cartesian::new(Vector::new(0.01, 0.0, 0.0));

        let mut cell2 = fitted_from_ref(&ref_uc);
        cell2.unit_cell.cell_atoms.b[0].position +=
            Cartesian::new(Vector::new(0.0, 0.02, 0.0));

        let fitted_ucs = vec![cell0, cell1, cell2];
        let bec = isotropic_bec(0.0, 2.0, 0.0);

        let spec = PolarizationSpec {
            bec: &bec,
            lattice_const: 1.0,
        };

        let ctx = PolarizationCtx {
            fitted_ucs: &fitted_ucs,
            ref_uc: &ref_uc,
        };

        let results = calculate_pol(&spec, &ctx).unwrap();

        assert_eq!(results.len(), 3);

        for (i, (cell, polarization)) in results.iter().enumerate() {
            assert!(std::ptr::eq(*cell, &fitted_ucs[i]));

            let (_, expected) =
                calculate_pol_for_cell(
                    &fitted_ucs[i],
                    &ref_uc,
                    &bec,
                    1.0,
                )
                .unwrap();

            assert_vector_approx_eq(
                polarization.as_vector(),
                expected.as_vector(),
            );
        }

        assert_vector_approx_eq(
            results[0].1.as_vector(),
            &Vector::zeros(),
        );

        assert_vector_approx_eq(
            results[1].1.as_vector(),
            &(Vector::new(0.02, 0.0, 0.0) * CONVERSION),
        );

        assert_vector_approx_eq(
            results[2].1.as_vector(),
            &(Vector::new(0.0, 0.04, 0.0) * CONVERSION),
        );
    }

    #[test]
    fn calculate_pol_empty_cells() {
        let ref_uc = reference_cell();
        let fitted_ucs: Vec<FittedSimCell> = Vec::new();
        let bec = isotropic_bec(1.0, 2.0, -2.0);

        let spec = PolarizationSpec {
            bec: &bec,
            lattice_const: 1.0,
        };

        let ctx = PolarizationCtx {
            fitted_ucs: &fitted_ucs,
            ref_uc: &ref_uc,
        };

        let results = calculate_pol(&spec, &ctx).unwrap();

        assert!(results.is_empty());
    }

    fn reference_cell() -> RefCell {
        RefCellBuilder::new()
            .geometry(CellGeometry::Cubic)
            .lattice_constant(1.0)
            .elements(vec![
                Element::Sr,
                Element::Ti,
                Element::O,
            ])
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
}
