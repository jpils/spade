use spade_core::cell::reference::RefCellBuilder;
use spade_core::geometry::NearestNeighbors;
use spade_core::observables::op::{Op, OpSpec, OpCtx};
use spade_core::observables::polarization::Polarization;
use spade_core::prelude::*;
use spade_core::types::{Matrix, PermutationVariant, Vector};
use spade_core::{cell::simulation::FittedSimCell, observables::{self, Observable, ObservableKind}};

use crate::config::{ConfigSpec};

#[derive(Debug, Clone)]
pub enum ObservableResult<'a> {
    OP(Vec<(&'a FittedSimCell, Op)>),
    Polarization(Vec<(&'a FittedSimCell, Polarization)>),
    Strain { epsilon: Matrix },
    GammaMode { gamma: Vector },
    DwDiffusionConst { diff_const: f64 }
}

pub(super) fn calculate_observables<'a>(
    fitted_cells: &Vec<FittedSimCell>, 
    b_b_nns: &Vec<NearestNeighbors>,
    spec: &ConfigSpec,
) -> Result<Vec<ObservableResult<'a>>> {

    let mut observables: Vec<ObservableResult> = 
        Vec::with_capacity(spec.observables.observables.len());

    for obs in &spec.observables.observables {
        let observables = match obs {
            ObservableKind::OP => todo!(),
            ObservableKind::Polarization => todo!(),
            ObservableKind::Strain => todo!(),
            ObservableKind::GammaMode => todo!(),
            ObservableKind::DwDiffusionConst => todo!()
        };

    }
    todo!()
}

fn calculate_op<'a>(
    fitted_cells: &'a Vec<FittedSimCell>, 
    b_b_nns: &'a Vec<NearestNeighbors>,
    spec: &ConfigSpec
) -> Result<ObservableResult<'a>> {

    let ref_cell = RefCellBuilder::new()
        .geometry(spec.cell.geometry)
        .permutation_variant(PermutationVariant::R0)
        .lattice_constant(spec.cell.lattice_constant)
        .elements(spec.cell.elements.clone().into())
        .build()?;

    let ctx = OpCtx {
        fitted_ucs: fitted_cells.iter().collect(),
        ref_uc: &ref_cell,
        b_site_nearest_neighbors: b_b_nns
    };

    let spec = OpSpec {
        lattice_const: spec.cell.lattice_constant,
        elements: spec.cell.elements.into(),
        geometry: spec.cell.geometry
    };

    let op = spade_core::observables::op::calculate_op(&spec, &ctx)?;

    Ok(ObservableResult::OP(op))
}
