use super::errors::Error;
use nalgebra::{Vector3, UnitQuaternion, Matrix3};

pub type Vector = Vector3<f64>;
pub type Vectors = Vec<Vector>;
pub type NearestNeighborDist = Vec<(usize, f64)>;
pub type UQuaternion = UnitQuaternion<f64>;
pub type Matrix = Matrix3<f64>;

pub enum Element {
    Sr,
    Ti,
    O
}

impl Element {
    pub fn get_mass(&self) -> f64 {
        match *self {
            Element::Sr => 87.62,
            Element::Ti => 47.867,
            Element::O => 15.999,
        }
    }
}

pub enum Coordinates {
    Direct,
    Cartesian
}

pub enum DWType {
    HT,
    HH,
    APB
}

pub enum DWSide {
    Left,
    Right,
    Center
}

pub enum PhaseFactor {
    Negative,
    Positive
}

pub enum Rotation {
    Right,
    Left
}

pub enum Observable {
    OP { phi: Vector },
    Polarization { pol: Vector },
    Strain { epsilon: Matrix },
    GammaMode { gamma: Vector },
    DwDiffusionConst { diff_const: f64 }
}

pub type Observables = Vec<Observable>;

pub struct Atom {
    pub atom_type: Element,
    pub position: Vector,
}

impl Atom {
    pub fn new(atom_type: Element, position: Vector) -> Self {
        Atom { atom_type, position }
    }
}

pub type Atoms = Vec<Atom>;
pub type AtomPair = (Atom, Atom);
pub type AtomPairs = Vec<AtomPair>;

pub struct SupercellAtoms {
    pub a_site: Atoms,
    pub b_site: Atoms,
    pub o_site: Atoms
}

impl SupercellAtoms {
    pub fn new( positions: Vectors, n_sr: usize, n_ti: usize, n_o: usize) -> Result<Self, Error> {
        let n_elements = n_sr + n_ti + n_o;

        if positions.len() != n_elements {
            Err(Error::SizeMismatch { expected: n_elements, got: positions.len() })
        } else {
            let a_site = positions[..n_sr].iter().map(|pos| Atom{ atom_type: Element::Sr, position: *pos }).collect();
            let b_site = positions[n_sr..n_ti].iter().map(|pos| Atom{ atom_type: Element::Sr, position: *pos }).collect();
            let o_site = positions[n_sr+n_ti..].iter().map(|pos| Atom{ atom_type: Element::Sr, position: *pos }).collect();

            Ok(SupercellAtoms { a_site, b_site, o_site })
        }
    }
}
