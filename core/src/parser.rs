use crate::prelude::*;
use crate::types::ReadResultVariant;
use std::path::Path;

pub enum Parser {
    Poscar,
    Xdatcar,
}

impl Parser {
    // AsRef<Path>: take everything that can be converted to &Path
    pub fn read<P: AsRef<Path>>(&self, filepath: P) -> Result<ReadResultVariant> { 
        match self {
            Parser::Poscar => vasp::read_poscar(filepath),
            Parser::Xdatcar => vasp::read_xdatcar(filepath)
        }
    }
}

mod vasp;
