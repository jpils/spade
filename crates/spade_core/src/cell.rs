use crate::prelude::*;
use crate::geometry::{get_displacements_between};
use crate::types::{Atom, Atoms, Element, UVector, UQuaternion, Vector, Vectors, AtomDisplacements, Elements};

#[derive(Clone, Debug)]
pub struct UnitCell {
    cell_atoms: UnitCellAtoms,
    center_of_mass: Vector,
    orientation: Option<UQuaternion>
}

impl UnitCell {
    fn get_orientation(&self) -> Option<UQuaternion> {
        self.orientation
    }

    fn get_centered_cell(&self) -> Self {
        let mut cell = self.clone();

        let center_atoms = |atoms: &mut Atoms| {
            for atom in atoms {
                atom.position -= cell.center_of_mass;
            }
        };

        center_atoms(&mut cell.cell_atoms.a);
        center_atoms(&mut cell.cell_atoms.b);
        center_atoms(&mut cell.cell_atoms.x);

        cell
    }

    fn get_displacements_from(&self, rhs: &Self) -> Result<AtomDisplacements> {
        let a_site_displacements = get_displacements_between(
            &self.cell_atoms.a,
            &rhs.cell_atoms.a
        )?;
        let b_site_displacements = get_displacements_between(
            &self.cell_atoms.b,
            &rhs.cell_atoms.b
        )?;
        let x_site_displacements = get_displacements_between(
            &self.cell_atoms.x,
            &rhs.cell_atoms.x
        )?;

        Ok(AtomDisplacements { a_site_displacements, b_site_displacements, x_site_displacements })
    }
}

#[derive(Clone, Debug)]
pub struct UnitCellAtoms {
    a: Atoms,
    b: Atoms,
    x: Atoms // for future: interstitial sites as Option<Atoms>
}

#[derive(Clone, Copy, Debug)]
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
