use nalgebra::{Quaternion, Vector4};
use crate::cell::{UnitCellOps, AsUnitCell};
use crate::cell::reference::RefCell;
use crate::cell::simulation::SimCell;
use crate::prelude::*;
use crate::{types::{AtomDisplacements, Matrix, UQuaternion}};

pub(crate) struct GradientDescentCtx<'a> {
    reference_uc: &'a RefCell, 
    centered_simulation_uc: &'a SimCell,
    lattice_const: f64,
    init_quaternion: UQuaternion,
    step_size: f64,
    max_iter_gradient_descent: usize,
    max_iter_line_search: usize,
    max_step_size_adjustments: usize
}

pub(crate) fn gradient_descent(ctx: GradientDescentCtx) -> Result<UQuaternion> {
    let reference_uc_init = ctx.reference_uc.rotated_cell(&ctx.init_quaternion);
    let displacements = ctx.centered_simulation_uc
        .displacements_from(&reference_uc_init)?;
    let mut current_unit_quaternion = UQuaternion::identity();
    let mut current_sq_dist = sq_dist(&displacements);
    let mut current_step_size = ctx.step_size;
    let mut current_iter: usize = 0;
    let mut step_size_adjustments: usize = 0;

    while current_iter < ctx.max_iter_gradient_descent {
        let g_term = compute_g_term(
            &reference_uc_init, 
            ctx.centered_simulation_uc, 
            &current_unit_quaternion, 
            ctx.lattice_const
        );

        let line_search_ctx = LineSearchCtx {
            simulation_uc_centered: ctx.centered_simulation_uc,
            reference_uc: &reference_uc_init,
            current_sq_dist: current_sq_dist,
            quaternion: &current_unit_quaternion,
            g_term: &g_term,
            step_size: current_step_size,
            max_iter: ctx.max_iter_line_search
        };

        let (new_unit_quaternion, new_sq_dist) = line_search(line_search_ctx)?;

        if new_sq_dist >= current_sq_dist && step_size_adjustments < ctx.max_step_size_adjustments {
            current_step_size /= 5.0;
            step_size_adjustments += 1;
            current_iter += 1;
            continue;
        } else if new_sq_dist >= current_sq_dist && step_size_adjustments >= ctx.max_step_size_adjustments {
            break;
        }
        
        current_unit_quaternion = new_unit_quaternion;
        current_sq_dist = new_sq_dist;

        step_size_adjustments = 0;
        current_iter += 1;
    }

    Ok(current_unit_quaternion*ctx.init_quaternion)
}

struct LineSearchCtx<'a> {
    simulation_uc_centered: &'a SimCell,
    reference_uc: &'a RefCell,
    current_sq_dist: f64, 
    quaternion: &'a UQuaternion, 
    g_term: &'a Quaternion<f64>,
    step_size: f64,
    max_iter: usize
}

#[inline]
fn sq_dist(displacements: &AtomDisplacements) -> f64 {
    displacements.a_site_displacements
        .iter()
        .fold(0.0, |acc, displacement| acc + displacement.norm_squared())
}

#[inline]
fn compute_gradient_entry(i: usize, current_quaternion: &UQuaternion) -> Matrix {
    let q0 = current_quaternion.w;
    let q1 = current_quaternion.i;
    let q2 = current_quaternion.j;
    let q3 = current_quaternion.k;

    match i {
        0 => 2.0 * Matrix::new(
            2.0*q1, q2, q3,
            q2, 0.0, -q0,
            q3, q0, 0.0
        ),
        1 => 2.0 * Matrix::new(
            0.0, q1, q0,
            q1, 2.0*q2, q3,
            -q0, q3, 0.0
        ),
        2 => 2.0 * Matrix::new(
            0.0, -q0, q1,
            q0, 0.0, q2,
            q1, q2, 2.0*q3
        ),
        3 => 2.0 * Matrix::new(
            2.0*q0, -q3, q2,
            q3, 2.0*q0, -q1,
            -q2, q1, 2.0*q0
        ),
        _ => unreachable!()
    }
}

#[inline]
fn compute_g_term(
    reference_uc: &RefCell, 
    simulation_uc: &SimCell, 
    current_quaternion: &UQuaternion, 
    lattice_const: f64
) -> Quaternion<f64> {

    let mut g_term_coeffs = Vector4::zeros();
    for i in 0..4 {
        let gradient_entry_i = compute_gradient_entry(i, current_quaternion);
        let mut sum = 0.0;
        // TODO currently only cubic implementation
        for j in 0..4 {
            let ref_a_site_atoms_top = &reference_uc.as_unit_cell().cell_atoms.a[j];
            let ref_a_site_atoms_bottom = &reference_uc.as_unit_cell().cell_atoms.a[j+4];
            let sim_a_site_atoms_top = &simulation_uc.as_unit_cell().cell_atoms.a[j];
            let sim_a_site_atoms_bottom = &simulation_uc.as_unit_cell().cell_atoms.a[j+4];

            sum += sim_a_site_atoms_top.position.dot(&(gradient_entry_i*ref_a_site_atoms_top.position));
            sum += sim_a_site_atoms_bottom.position.dot(&(gradient_entry_i*ref_a_site_atoms_bottom.position));
        }

        g_term_coeffs[i] = -2.0*sum/lattice_const.powi(2);
    }
        
    Quaternion::from_vector(g_term_coeffs)
}

#[inline]
fn compute_lambda(unit_quaternion: &UQuaternion, g_term: &Quaternion<f64>, step_size: f64) -> f64 {
    let a_term = (unit_quaternion.conjugate().into_inner()*g_term).w;
    let b_term = g_term.norm_squared();

    let lambda = 1.0 - step_size*a_term - (step_size.powi(2)*(a_term.powi(2) - b_term) + 1.0).sqrt();
    lambda
}

fn line_search(ctx: LineSearchCtx) -> Result<(UQuaternion, f64)> {
    let mut best_quaternion = ctx.quaternion.clone();
    let mut best_dist = ctx.current_sq_dist;
    let current_step_size = ctx.step_size*0.01;

    for i in 0..ctx.max_iter {
        let exponent = i
            .try_into()
            .map_err(|_| Error::Generic("Could not convert usize to i32".into()))?;

        let new_step_size = (1.1 as f64) .powi(exponent) * current_step_size;
        let new_lambda = compute_lambda(ctx.quaternion, ctx.g_term, new_step_size);
        let delta = new_step_size*ctx.g_term.coords + new_lambda*ctx.quaternion.coords;

        //bidirectional search
        let new_quaternion_f = UQuaternion::new_normalize(Quaternion::from_vector(ctx.quaternion.coords - delta));
        let new_quaternion_b = UQuaternion::new_normalize(Quaternion::from_vector(ctx.quaternion.coords + delta));

        let rotated_reference_uc_f = ctx.reference_uc.rotated_cell(&new_quaternion_f);
        let rotated_reference_uc_b = ctx.reference_uc.rotated_cell(&new_quaternion_b);

        let displacements_f = ctx.simulation_uc_centered.displacements_from(&rotated_reference_uc_f)?;
        let displacements_b = ctx.simulation_uc_centered.displacements_from(&rotated_reference_uc_b)?;

        let new_sq_dist_f = sq_dist(&displacements_f);
        let new_sq_dist_b = sq_dist(&displacements_b);

        if new_sq_dist_f < best_dist || new_sq_dist_b < best_dist {
            if new_sq_dist_f <= new_sq_dist_b {
                best_quaternion = new_quaternion_f;
                best_dist = new_sq_dist_f;
            } else {
                best_quaternion = new_quaternion_b;
                best_dist = new_sq_dist_b;
            }
        }
    }

    Ok((best_quaternion.clone(), best_dist))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cell::{reference::{RefCell, RefCellBuilder}, simulation::SimCell, CellGeometry},
        types::{AtomDisplacements, Element, RotationVariant, Vector},
    };
    use nalgebra::Unit;
    use std::f64::consts::FRAC_PI_6;

    const EPS: f64 = 1.0e-12;

    #[test]
    fn sq_dist_sums_only_a_site_displacements() {
        let displacements = AtomDisplacements {
            a_site_displacements: vec![
                Vector::new(1.0, 0.0, 0.0),
                Vector::new(0.0, 2.0, 0.0),
                Vector::new(0.0, 0.0, 3.0),
            ],
            b_site_displacements: vec![Vector::new(10.0, 0.0, 0.0)],
            x_site_displacements: vec![Vector::new(10.0, 0.0, 0.0)],
        };

        assert_approx_eq(sq_dist(&displacements), 14.0);
    }

    #[test]
    fn gradient_entry_at_identity_matches_rotation_derivatives() {
        let q = UQuaternion::identity();

        assert_matrix_approx_eq(
            compute_gradient_entry(0, &q),
            2.0 * Matrix::new(
                0.0, 0.0, 0.0,
                0.0, 0.0, -1.0,
                0.0, 1.0, 0.0,
            ),
        );
        assert_matrix_approx_eq(
            compute_gradient_entry(1, &q),
            2.0 * Matrix::new(
                0.0, 0.0, 1.0,
                0.0, 0.0, 0.0,
                -1.0, 0.0, 0.0,
            ),
        );
        assert_matrix_approx_eq(
            compute_gradient_entry(2, &q),
            2.0 * Matrix::new(
                0.0, -1.0, 0.0,
                1.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ),
        );
        assert_matrix_approx_eq(
            compute_gradient_entry(3, &q),
            2.0 * Matrix::new(
                2.0, 0.0, 0.0,
                0.0, 2.0, 0.0,
                0.0, 0.0, 2.0,
            ),
        );
    }

    #[test]
    fn g_term_is_finite_for_md_like_displaced_cell() {
        let reference_uc = pristine_uc();
        let simulation_uc = md_like_displaced_uc(&reference_uc);

        let g_term = compute_g_term(
            &reference_uc,
            &simulation_uc,
            &UQuaternion::identity(),
            1.0,
        );

        assert!(g_term.coords.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn g_term_is_not_forced_to_unit_norm() {
        let reference_uc = pristine_uc();
        let simulation_uc = md_like_displaced_uc(&reference_uc);

        let g_term = compute_g_term(
            &reference_uc,
            &simulation_uc,
            &UQuaternion::identity(),
            1.0,
        );

        assert!((g_term.norm() - 1.0).abs() > 1.0e-6);
    }

    #[test]
    fn lambda_is_zero_for_zero_step_size() {
        let q = UQuaternion::identity();
        let g = Quaternion::new(0.0, 0.1, -0.2, 0.3);

        assert_approx_eq(compute_lambda(&q, &g, 0.0), 0.0);
    }

    #[test]
    fn lambda_keeps_update_on_unit_sphere() {
        let q = UQuaternion::from_axis_angle(
            &Unit::new_normalize(Vector::new(0.0, 1.0, 0.0)),
            FRAC_PI_6,
        );
        let g = Quaternion::new(0.0, 0.1, -0.2, 0.3);
        let step_size = 1.0e-3;
        let lambda = compute_lambda(&q, &g, step_size);
        let delta = step_size * g.coords + lambda * q.coords;
        let new_q = Quaternion::from_vector(q.coords - delta);

        assert_approx_eq(new_q.norm(), 1.0);
    }

    #[test]
    fn line_search_keeps_zero_distance_for_matching_cells() {
        let reference_uc = pristine_uc();
        let simulation_uc = sim_from_ref_cell(&reference_uc);
        let q = UQuaternion::identity();
        let g = compute_g_term(&reference_uc, &simulation_uc, &q, 1.0);
        let ctx = LineSearchCtx {
            simulation_uc_centered: &simulation_uc,
            reference_uc: &reference_uc,
            current_sq_dist: 0.0,
            quaternion: &q,
            g_term: &g,
            step_size: 5.0e-4,
            max_iter: 10,
        };

        let (best_q, best_dist) = line_search(ctx).unwrap();

        assert_approx_eq(best_dist, 0.0);
        assert_approx_eq(best_q.angle(), 0.0);
    }

    #[test]
    fn gradient_descent_returns_identity_for_matching_cells() {
        let reference_uc = pristine_uc();
        let simulation_uc = sim_from_ref_cell(&reference_uc);
        let ctx = GradientDescentCtx {
            reference_uc: &reference_uc,
            centered_simulation_uc: &simulation_uc,
            lattice_const: 1.0,
            init_quaternion: UQuaternion::identity(),
            step_size: 5.0e-4,
            max_iter_gradient_descent: 10,
            max_iter_line_search: 10,
            max_step_size_adjustments: 3,
        };

        let q = gradient_descent(ctx).unwrap();

        assert_approx_eq(q.angle(), 0.0);
    }

    #[test]
    fn gradient_descent_returns_initial_quaternion_when_it_already_matches() {
        let reference_uc = pristine_uc();
        let init_q = UQuaternion::from_axis_angle(
            &Unit::new_normalize(Vector::new(0.0, 1.0, 0.0)),
            FRAC_PI_6,
        );
        let simulation_uc = sim_from_ref_cell(&reference_uc.rotated_cell(&init_q));
        let ctx = GradientDescentCtx {
            reference_uc: &reference_uc,
            centered_simulation_uc: &simulation_uc,
            lattice_const: 1.0,
            init_quaternion: init_q,
            step_size: 5.0e-4,
            max_iter_gradient_descent: 10,
            max_iter_line_search: 10,
            max_step_size_adjustments: 3,
        };

        let q = gradient_descent(ctx).unwrap();

        assert_quaternion_approx_eq(q, init_q, 1.0e-12);
    }

    #[test]
    fn gradient_descent_reduces_distance_for_rotated_md_like_cell() {
        let reference_uc = pristine_uc();
        let target_q = UQuaternion::from_axis_angle(
            &Unit::new_normalize(Vector::new(0.0, 1.0, 0.0)),
            0.05,
        );
        let rotated_reference_uc = reference_uc.rotated_cell(&target_q);
        let simulation_uc = md_like_displaced_uc(&rotated_reference_uc);

        let initial_dist = sq_dist_between(&simulation_uc, &reference_uc);
        let ctx = GradientDescentCtx {
            reference_uc: &reference_uc,
            centered_simulation_uc: &simulation_uc,
            lattice_const: 1.0,
            init_quaternion: UQuaternion::identity(),
            step_size: 5.0e-4,
            max_iter_gradient_descent: 100,
            max_iter_line_search: 100,
            max_step_size_adjustments: 5,
        };

        let q = gradient_descent(ctx).unwrap();
        let rotated_reference_uc = reference_uc.rotated_cell(&q);
        let final_dist = sq_dist_between(&simulation_uc, &rotated_reference_uc);

        assert!(final_dist < initial_dist, "initial={initial_dist}, final={final_dist}");
    }

    fn pristine_uc() -> RefCell {
        RefCellBuilder::new()
            .geometry(CellGeometry::Cubic)
            .lattice_constant(1.0)
            .elements(vec![Element::Sr, Element::Ti, Element::O])
            .rotation_variant(RotationVariant::R0)
            .build()
            .unwrap()
    }

    fn sim_from_ref_cell(reference_uc: &RefCell) -> SimCell {
        SimCell(reference_uc.as_unit_cell().clone())
    }

    fn md_like_displaced_uc(reference_uc: &RefCell) -> SimCell {
        let mut simulation_uc = reference_uc.as_unit_cell().clone();

        displace_atoms(&mut simulation_uc.cell_atoms.a, 0.003);
        displace_atoms(&mut simulation_uc.cell_atoms.b, 0.001);
        displace_atoms(&mut simulation_uc.cell_atoms.x, 0.002);

        SimCell(simulation_uc)
    }

    fn displace_atoms(atoms: &mut [crate::types::Atom], scale: f64) {
        for (i, atom) in atoms.iter_mut().enumerate() {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            atom.position += scale
                * Vector::new(
                    sign * (i as f64 + 1.0),
                    -0.5 * sign,
                    0.25 * (i as f64 + 1.0),
                );
        }
    }

    fn sq_dist_between<Lhs: UnitCellOps, Rhs: UnitCellOps>(lhs: &Lhs, rhs: &Rhs) -> f64 {
        sq_dist(&lhs.displacements_from(rhs).unwrap())
    }

    fn assert_approx_eq(lhs: f64, rhs: f64) {
        assert!((lhs - rhs).abs() <= EPS, "lhs={lhs}, rhs={rhs}");
    }

    fn assert_matrix_approx_eq(lhs: Matrix, rhs: Matrix) {
        let diff = lhs - rhs;
        assert!(diff.norm() <= EPS, "lhs={lhs:?}, rhs={rhs:?}");
    }

    fn assert_quaternion_approx_eq(lhs: UQuaternion, rhs: UQuaternion, eps: f64) {
        let delta = lhs.inverse() * rhs;
        assert!(delta.angle() <= eps, "lhs={lhs:?}, rhs={rhs:?}");
    }
}
