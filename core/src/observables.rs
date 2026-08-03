use crate::observables::polarization::Polarization;
use crate::observables::op::Op;
use crate::types::{Matrix, Vector};
use crate::cell::simulation::FittedSimCell;

pub(crate) trait Observable {}

pub(crate) enum ObservableKind {
    OP,
    Polarization,
    Strain,
    GammaMode,
    DwDiffusionConst
}

pub(crate) enum ObservableResult {
    OP (Vec<Op>),
    Polarization (Vec<Polarization>),
    Strain { epsilon: Matrix },
    GammaMode { gamma: Vector },
    DwDiffusionConst { diff_const: f64 }
}

pub(crate) mod op;
pub(crate) mod polarization;
pub(crate) mod statistics;
