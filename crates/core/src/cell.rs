use serde::Deserialize;

use crate::prelude::*;
use crate::geometry::{get_displacements_between};
use crate::types::{AsVector, Atom, AtomDisplacements, Atoms, CartAtom, CartAtoms, Cartesian, CellAlignment, Element, Elements, UQuaternion, UVector, Vector, Vectors};

#[derive(Clone, Debug, PartialEq)]
pub struct UnitCell {
    pub cell_atoms: UnitCellAtoms,
    pub center_of_mass: Cartesian,
}

pub trait UnitCellOps: AsUnitCell {
    fn center_inplace(&mut self) -> &Self;
    fn centered_cell(&self) -> Self;
    fn displacements_from<Cell: UnitCellOps>(&self, rhs: &Cell) -> Result<AtomDisplacements>;
    fn rotate_inplace(&mut self, q: &UQuaternion) -> &Self;
    fn rotated_cell(&self, q: &UQuaternion) -> Self;
}

pub trait AsUnitCell {
    fn as_unit_cell(&self) -> &UnitCell;
    fn as_unit_cell_mut(&mut self) -> &mut UnitCell;
}

impl AsUnitCell for UnitCell {
    fn as_unit_cell(&self) -> &UnitCell {
        self
    }

    fn as_unit_cell_mut(&mut self) -> &mut UnitCell {
        self
    }
}

impl UnitCellOps for UnitCell {
    fn center_inplace(&mut self) -> &Self {
        let center_atoms = |atoms: &mut CartAtoms| {
            for atom in atoms {
                atom.position -= self.center_of_mass;
            }
        };

        center_atoms(&mut self.cell_atoms.a);
        center_atoms(&mut self.cell_atoms.b);
        center_atoms(&mut self.cell_atoms.x);

        self
    }

    fn centered_cell(&self) -> Self {
        let mut cell = self.clone();
        cell.center_inplace();
        cell
    }

    fn displacements_from<Cell: UnitCellOps>(&self, rhs: &Cell) -> Result<AtomDisplacements> {
        let a_site_displacements = get_displacements_between(
            &self.cell_atoms.a,
            &rhs.as_unit_cell().cell_atoms.a
        )?;

        let b_site_displacements = get_displacements_between(
            &self.cell_atoms.b,
            &rhs.as_unit_cell().cell_atoms.b
        )?;

        let x_site_displacements = get_displacements_between(
            &self.cell_atoms.x,
            &rhs.as_unit_cell().cell_atoms.x
        )?;

        Ok(AtomDisplacements { 
            a: a_site_displacements, 
            b: b_site_displacements, 
            x: x_site_displacements 
        })
    }

    fn rotate_inplace(&mut self, q: &UQuaternion) -> &Self {
        let rotate = |atoms: &mut CartAtoms| {
            for atom in atoms.iter_mut() {
                let new_pos =  q*atom.position.as_vector();
                atom.position = Cartesian::new(new_pos)
            }
        };

        rotate(&mut self.cell_atoms.a);
        rotate(&mut self.cell_atoms.b);
        rotate(&mut self.cell_atoms.x);

        self
    }

    fn rotated_cell(&self, q: &UQuaternion) -> Self {
        let mut current_uc = self.clone();
        current_uc.rotate_inplace(q);
        current_uc
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnitCellAtoms {
    pub a: CartAtoms,
    pub b: CartAtoms,
    pub x: CartAtoms // for future: interstitial sites as Option<Atoms>
}

#[derive(Clone, Copy, PartialEq, Debug, Deserialize)]
pub enum CellGeometry {
    Cubic,
    Tetragonal,
}

pub mod reference;
pub mod simulation;

#[cfg(test)]
mod tests {
    use super::*;
}
