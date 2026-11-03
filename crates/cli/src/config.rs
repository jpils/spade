use std::{fmt::format, fs, path::{Path, PathBuf}};
use anyhow::{Result, Context};

use serde::Deserialize;
use spade_core::{cell::CellGeometry, observables::ObservableKind, parser::Parser, types::{DWType, Element, Matrix}};

#[derive(Debug, Deserialize)]
pub(crate) struct ConfigSpec {
    pub(crate) io: IO,
    pub(crate) cell: Cell,
    pub(crate) observables: Observables,
    pub(crate) analysis: Analysis,
}

impl ConfigSpec {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let context = fs::read_to_string(path)
            .with_context(|| format!("Reading {}", path.display()))?;

        let config = toml::from_str(&context)
            .with_context(|| format!("Parsing {}", path.display()))?;

        Ok(config)
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct IO {
    pub(crate) file: PathBuf,
    pub(crate) format: Parser,
    pub(crate) out_dir: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Cell {
    pub(crate) geometry: CellGeometry,
    pub(crate) dw_type: DWType,
    pub(crate) lattice_constant: f64,
    pub(crate) elements: Option<[Element; 3]>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Analysis {
    pub(crate) fitting: Fitting,
    pub(crate) statistics: Statistics
}

#[derive(Deserialize, Debug)]
pub(crate) struct Fitting {
    pub(crate) step_size: Option<f64>,
    pub(crate) max_iter: Option<usize>,
}

#[derive(Deserialize, Debug)]
pub(crate) struct Statistics {
    pub(crate) blocks: usize,
    pub(crate) crosssection_bin_threshold: Option<f64>,
    pub(crate) sliding_window: Option<(usize, usize)>,
}

#[derive(Deserialize, Debug)]
pub(crate) struct Observables {
    pub(crate) observables: Vec<ObservableKind>,
    pub(crate) polarization: Option<Polarization>,
}

#[derive(Deserialize, Debug)]
pub(crate) struct Polarization {
    #[serde(deserialize_with = "deserialize_matrix")]
    pub(crate) bec_a: Matrix,

    #[serde(deserialize_with = "deserialize_matrix")]
    pub(crate) bec_b: Matrix,

    #[serde(deserialize_with = "deserialize_matrix")]
    pub(crate) bec_x: Matrix,
}

fn deserialize_matrix<'de, D>(
    deserializer: D
) -> Result<Matrix, D::Error> 
where D: serde::Deserializer<'de>,
{
    let rows = <[[f64; 3]; 3]>::deserialize(deserializer)?;

    Ok(Matrix::from_fn(|i, j| rows[i][j]))
}
