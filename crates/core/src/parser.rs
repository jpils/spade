use serde::Deserialize;

use crate::{prelude::*, types::ReadResult};
use std::path::Path;

#[derive(Clone, Copy, PartialEq, Eq, Deserialize, Debug)]
pub enum Parser {
    #[serde(rename = "xdatcar", alias = "XDTACAR", alias = "Xdatcar")]
    Poscar,

    #[serde(rename = "porcar", alias = "POSCAR", alias = "Poscar")]
    Xdatcar,
}

impl Parser {
    // AsRef<Path>: take everything that can be converted to &Path
    pub fn read<P: AsRef<Path>>(&self, filepath: P) -> Result<Vec<ReadResult>> { 
        match self {
            Parser::Poscar => vasp::read_poscar(filepath),
            Parser::Xdatcar => vasp::read_xdatcar(filepath)
        }
    }
}

mod vasp;
