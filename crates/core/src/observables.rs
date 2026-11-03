use serde::Deserialize;

use crate::observables::polarization::Polarization;
use crate::observables::op::Op;
use crate::types::{Matrix, Vector};
use crate::cell::simulation::FittedSimCell;

pub trait Observable {}

pub struct BinnedObservable<T: Observable> {
    pub bins: Vec<f64>,
    pub observable: Vec<T>
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum ObservableKind {
    OP,
    Polarization,
    Strain,
    GammaMode,
    DwDiffusionConst
}

#[derive(Debug, Clone)]
pub enum ObservableResult {
    OP (Vec<Op>),
    Polarization (Vec<Polarization>),
    Strain { epsilon: Matrix },
    GammaMode { gamma: Vector },
    DwDiffusionConst { diff_const: f64 }
}

pub mod op;
pub mod polarization;
