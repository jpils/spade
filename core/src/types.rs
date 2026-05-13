use crate::prelude::*;
use nalgebra::{Matrix3, UnitQuaternion, UnitVector3, Vector3};
use strum_macros::EnumString;

pub type Vector = Vector3<f64>;
pub type Vectors = Vec<Vector>;
pub type NearestNeighborDist = Vec<(usize, f64)>;
pub type UVector = UnitVector3<f64>;
pub type UQuaternion = UnitQuaternion<f64>;
pub type Matrix = Matrix3<f64>;

#[derive(Debug, PartialEq, EnumString, Clone, Copy, Default)]
pub enum Element {
    Sr,
    Ti,
    O,
    #[default]
    Unknown
}

impl Element {
    #[must_use]
    pub fn get_mass(&self) -> Option<f64> {
        match *self {
            Element::Sr => Some(87.62),
            Element::Ti => Some(47.867),
            Element::O => Some(15.999),
            Element::Unknown => None
        }
    }
}

pub type Elements = Vec<Element>;

#[derive(Debug, EnumString, PartialEq, Clone, Copy)]
pub enum Coordinates {
    #[strum(serialize = "D", serialize = "Direct")]
    Direct,
    #[strum(serialize = "C", serialize = "Cartesian")]
    Cartesian
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DWType {
    HT,
    HH,
    APB
}

#[derive(Debug, Clone, Copy)]
pub enum DWSide {
    Left,
    Right,
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

#[derive(PartialEq, Debug, Copy, Clone, Default)]
pub struct Atom {
    pub atom_type: Element,
    pub position: Vector,
}

impl Atom {
    #[must_use]
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
    pub x_site: Atoms
}

impl SupercellAtoms {
    pub fn new( positions: &Vectors, n_sr: usize, n_ti: usize, n_o: usize) -> Result<Self> {
        let n_elements = n_sr + n_ti + n_o;

        if positions.len() == n_elements {
            let a_site = positions[..n_sr].iter().map(|pos| Atom{ atom_type: Element::Sr, position: *pos }).collect();
            let b_site = positions[n_sr..n_sr+n_ti].iter().map(|pos| Atom{ atom_type: Element::Ti, position: *pos }).collect();
            let c_site = positions[n_sr+n_ti..].iter().map(|pos| Atom{ atom_type: Element::O, position: *pos }).collect();

            Ok(SupercellAtoms { a_site, b_site, x_site: c_site })
        } else {
            Err(Error::SizeMismatch { expected: n_elements, got: positions.len() })
        }
    }
}

pub struct AtomDisplacements {
    pub a_site_displacements: Vectors,
    pub b_site_displacements: Vectors,
    pub x_site_displacements: Vectors
}

#[derive(PartialEq, Debug, Clone)]
pub struct ReadResult {
    pub positions: Vectors,
    pub element_count: Vec<(Element, usize)>,
    pub cell_matrix: Matrix,
    pub coordinate_type: Coordinates,
}

#[derive(PartialEq, Debug)]
pub enum ReadResultVariant {
    ReadResult { res: ReadResult },
    ReadResults { res: Vec<ReadResult> }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supercell_atoms_new_succeed() {
        let sr = Vector::new(1.0, 1.0, 1.0);
        let ti = Vector::new(2.0, 2.0, 2.0);
        let o1 = Vector::new(3.0, 3.0, 3.0);
        let o2 = Vector::new(4.0, 4.0, 4.0);
        let o3 = Vector::new(5.0, 5.0, 5.0);

        let positions = vec![sr, ti, o1, o2, o3];

        let supercell_atoms = SupercellAtoms::new(&positions, 1, 1, 3).unwrap();

        assert_eq!(supercell_atoms.a_site[0].atom_type, Element::Sr);
        assert_eq!(supercell_atoms.a_site[0].position, sr);

        assert_eq!(supercell_atoms.b_site[0].atom_type, Element::Ti);
        assert_eq!(supercell_atoms.b_site[0].position, ti);

        supercell_atoms.x_site.iter().zip(vec![o1, o2, o3]).for_each(|(atom, expected)| {
            assert_eq!(atom.atom_type, Element::O);
            assert_eq!(atom.position, expected);
        });

        assert_eq!(supercell_atoms.a_site.len(), 1);
        assert_eq!(supercell_atoms.b_site.len(), 1);
        assert_eq!(supercell_atoms.x_site.len(), 3);
    }

    #[test]
    fn test_supercell_atoms_new_fail() {
        let sr = Vector::new(1.0, 1.0, 1.0);
        let ti = Vector::new(2.0, 2.0, 2.0);
        let o1 = Vector::new(3.0, 3.0, 3.0);
        let o2 = Vector::new(4.0, 4.0, 4.0);
        let o3 = Vector::new(5.0, 5.0, 5.0);

        let positions = vec![sr, ti, o1, o2, o3];

        assert!(SupercellAtoms::new(&positions, 2, 1, 3).is_err())
    }
}
