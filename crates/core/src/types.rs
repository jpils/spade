use std::{fmt::Debug, ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign}, process::Output};

use crate::{geometry::direct_to_cartesian, prelude::*};
use nalgebra::{Matrix3, UnitQuaternion, UnitVector3, Vector3, geometry};
use serde::Deserialize;
use strum_macros::EnumString;

pub type Vector = Vector3<f64>;
pub type Vectors = Vec<Vector>;
pub type UVector = UnitVector3<f64>;
pub type UQuaternion = UnitQuaternion<f64>;
pub type Matrix = Matrix3<f64>;

pub trait AsVector {
    fn as_vector(&self) -> &Vector;
    fn as_vector_mut(&mut self) -> &mut Vector;
}

pub trait AsMatrix {
    fn as_matrix(&self) -> &Matrix;
    fn as_matrix_mut(&mut self) -> &mut Matrix;
}

pub trait Coordinates: 
    PartialEq + Clone + Copy + Debug + AsVector +
    Sub<Output = Self> + Add<Output = Self> + 
    Mul<f64, Output = Self> + Div<f64, Output = Self> +
    SubAssign + AddAssign +
    MulAssign<f64> + DivAssign<f64>
{}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Direct(Vector);

impl Coordinates for Direct {}

impl Sub for Direct {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Add for Direct {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Mul<f64> for Direct {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        Self(self.0 * rhs)
    } 
}

impl Div<f64> for Direct {
    type Output = Self; 

    fn div(self, rhs: f64) -> Self::Output {
        Self(self.0 / rhs)
    }
}

impl AddAssign for Direct {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0
    }
}

impl SubAssign for Direct {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0
    }
}

impl MulAssign<f64> for Direct {
    fn mul_assign(&mut self, rhs: f64) {
        self.0 *= rhs
    } 
}

impl DivAssign<f64> for Direct {
    fn div_assign(&mut self, rhs: f64) {
        self.0 /= rhs
    }
}

impl AsVector for Direct {
    fn as_vector(&self) -> &Vector {
        &self.0
    }

    fn as_vector_mut(&mut self) -> &mut Vector {
        &mut self.0
    }
}

impl Direct {
    pub fn new(vec: Vector) -> Result<Self> {
        if vec.iter().any(|elem| *elem > 1.0) {
            return Err(Error::Generic(
                "Vector is not in direct coordinates. At least one element greater than 1!"
                .into()
            ));
        }

        Ok(Self(vec))
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Cartesian(Vector);

impl Coordinates for Cartesian {}

impl Sub for Cartesian {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Add for Cartesian {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Mul<f64> for Cartesian {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        Self(self.0 * rhs)
    } 
}

impl Div<f64> for Cartesian {
    type Output = Self; 

    fn div(self, rhs: f64) -> Self::Output {
        Self(self.0 / rhs)
    }
}

impl AsVector for Cartesian {
    fn as_vector(&self) -> &Vector {
        &self.0
    }

    fn as_vector_mut(&mut self) -> &mut Vector {
        &mut self.0
    }
}

impl AddAssign for Cartesian {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0
    }
}

impl SubAssign for Cartesian {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0
    }
}

impl MulAssign<f64> for Cartesian {
    fn mul_assign(&mut self, rhs: f64) {
        self.0 *= rhs
    } 
}

impl DivAssign<f64> for Cartesian {
    fn div_assign(&mut self, rhs: f64) {
        self.0 /= rhs
    }
}

impl Cartesian {
    pub fn new(vec: Vector) -> Self {
        Self(vec)
    }
}

#[derive(Deserialize, Debug, PartialEq, EnumString, Clone, Copy)]
pub enum Element {
    Sr,
    Ti,
    O,
}

impl Element {
    #[must_use]
    pub fn get_mass(self) -> f64 {
        match self {
            Element::Sr => 87.62,
            Element::Ti => 47.867,
            Element::O => 15.999,
        }
    }
}

pub type Elements = Vec<Element>;

#[derive(Debug, EnumString, PartialEq, Clone, Copy)]
pub enum CoordinatesType {
    #[strum(serialize = "D", serialize = "Direct")]
    Direct,
    #[strum(serialize = "C", serialize = "Cartesian")]
    Cartesian
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
pub enum DWType {
    HT,
    HH,
    Apb
}

#[derive(Debug, Clone, Copy)]
pub enum DWSide {
    Left,
    Right,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PhaseFactor {
    Negative,
    Positive,
    Undefined
}

impl PhaseFactor {
    pub fn try_to_int(self) -> Result<i32> {
        match self {
            Self::Negative => Ok(-1),
            Self::Positive => Ok(1),
            Self::Undefined => Err(Error::Generic("PhaseFactor not set".into()))
        }
    }

    pub fn try_from_int(int: i8) -> Result<Self> {
        match int {
            -1 => Ok(Self::Negative),
            1 => Ok(Self::Positive),
            _ => Err(Error::Generic("Int must be either -1 or 1".into()))
        }
    }

    pub fn get_flipped(self) -> Result<Self> {
        match self {
            Self::Negative => Ok(Self::Positive),
            Self::Positive => Ok(Self::Negative),
            Self::Undefined => Err(Error::Generic("PhaseFactor not set".into()))
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CellAlignment {
    pub unit_quaternion: UQuaternion,
    pub ref_variant: RotationVariant, 
    pub error: f64
}

#[derive(PartialEq, Debug, Copy, Clone)]
pub struct Atom<C: Coordinates> {
    pub atom_type: Option<Element>,
    pub position: C,
}

impl<C: Coordinates> Atom<C> {
    #[must_use]
    pub fn new(atom_type: Element, position: C) -> Self {
        Atom { atom_type: Some(atom_type), position }
    }
}


pub type Atoms<C> = Vec<Atom<C>>;
pub type CartAtom = Atom<Cartesian>;
pub type CartAtoms = Vec<Atom<Cartesian>>;
pub type CartAtomPair = (Atom<Cartesian>, Atom<Cartesian>);
pub type CartAtomPairs = Vec<CartAtomPair>;
pub type DirectAtom = Atom<Direct>;
pub type DirectAtoms = Vec<Atom<Direct>>;
pub type DirectAtomPair = (Atom<Direct>, Atom<Direct>);
pub type DirectAtomPairs = Vec<DirectAtomPair>;

pub struct SupercellAtoms<C: Coordinates> {
    pub a: Atoms<C>,
    pub b: Atoms<C>,
    pub x: Atoms<C>
}

impl<C: Coordinates> SupercellAtoms<C> {
    pub fn new( 
        positions: &[C], 
        n_a: (Element, usize), 
        n_b: (Element, usize), 
        n_x: (Element, usize)
    ) -> Result<Self> {
        let n_elements = n_a.1 + n_b.1 + n_x.1;

        if positions.len() == n_elements {
            let a_site = positions[..n_a.1].iter().map(|pos| Atom{ atom_type: Some(n_a.0), position: *pos }).collect();
            let b_site = positions[n_a.1..n_a.1+n_b.1].iter().map(|pos| Atom{ atom_type: Some(n_b.0), position: *pos }).collect();
            let x_site = positions[n_a.1+n_b.1..].iter().map(|pos| Atom{ atom_type: Some(n_x.0), position: *pos }).collect();

            Ok(SupercellAtoms { a: a_site, b: b_site, x: x_site })
        } else {
            Err(Error::SizeMismatch { expected: n_elements, got: positions.len() })
        }
    }
}

pub struct AtomDisplacements {
    pub a: Vec<Cartesian>,
    pub b: Vec<Cartesian>,
    pub x: Vec<Cartesian>
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RotationVariant {
    R0,
    R90,
    R180,
    R270
}

#[derive(PartialEq, Debug, Clone)]
pub struct ReadResult {
    pub positions: Vectors,
    pub element_count: Vec<(Element, usize)>,
    pub cell_matrix: Matrix,
    pub coordinate_type: CoordinatesType,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supercell_atoms_new_succeed() {
        let sr = Cartesian(Vector::new(1.0, 1.0, 1.0));
        let ti = Cartesian(Vector::new(2.0, 2.0, 2.0));
        let o1 = Cartesian(Vector::new(3.0, 3.0, 3.0));
        let o2 = Cartesian(Vector::new(4.0, 4.0, 4.0));
        let o3 = Cartesian(Vector::new(5.0, 5.0, 5.0));

        let positions = vec![sr, ti, o1, o2, o3];

        let supercell_atoms = SupercellAtoms::new(&positions, (Element::Sr, 1), (Element::Ti, 1), (Element::O, 3)).unwrap();

        assert_eq!(supercell_atoms.a[0].atom_type.unwrap(), Element::Sr);
        assert_eq!(supercell_atoms.a[0].position, sr);

        assert_eq!(supercell_atoms.b[0].atom_type.unwrap(), Element::Ti);
        assert_eq!(supercell_atoms.b[0].position, ti);

        supercell_atoms.x.iter().zip(vec![o1, o2, o3]).for_each(|(atom, expected)| {
            assert_eq!(atom.atom_type.unwrap(), Element::O);
            assert_eq!(atom.position, expected);
        });

        assert_eq!(supercell_atoms.a.len(), 1);
        assert_eq!(supercell_atoms.b.len(), 1);
        assert_eq!(supercell_atoms.x.len(), 3);
    }

    #[test]
    fn test_supercell_atoms_new_fail() {
        let sr = Cartesian(Vector::new(1.0, 1.0, 1.0));
        let ti = Cartesian(Vector::new(2.0, 2.0, 2.0));
        let o1 = Cartesian(Vector::new(3.0, 3.0, 3.0));
        let o2 = Cartesian(Vector::new(4.0, 4.0, 4.0));
        let o3 = Cartesian(Vector::new(5.0, 5.0, 5.0));

        let positions = vec![sr, ti, o1, o2, o3];

        assert!(SupercellAtoms::new(&positions, (Element::Sr, 2), (Element::Ti, 1), (Element::O, 3)).is_err())
    }
}
