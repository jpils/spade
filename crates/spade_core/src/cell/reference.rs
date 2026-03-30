use std::hash::BuildHasher;

use crate::{geometry, prelude::*};
use super::{CellGeometry, Elements, UVector, UnitCell, Error, Vector, UnitCellAtoms, Atoms, Atom};

/// creates a unit cell with the atoms sitting at their high symmetry positions and the
/// `center_of_mass` in the origin of the coordinate system (cartesian coordinates)
#[derive(Default, Debug)]
pub struct RefCellBuilder {
    geometry: Option<CellGeometry>,
    lattice_constant: Option<f64>,
    elements: Option<Elements>,
    tilt_angle: Option<f64>,
    tilt_axis: Option<UVector>,
}

impl RefCellBuilder {
    #[must_use]
    pub fn new() -> Self {
        RefCellBuilder::default()
    }

    pub fn geometry(&mut self, geometry: CellGeometry) -> &mut Self {
        self.geometry = Some(geometry);
        self
    }

    pub fn lattice_constant(&mut self, lattice_constant: f64) -> &mut Self {
        self.lattice_constant = Some(lattice_constant);
        self
    }

    pub fn tilt_angle(&mut self, tilt_angle: f64 /* rad */) -> &mut Self {
        self.tilt_angle = Some(tilt_angle);
        self
    }

    pub fn tilt_axis(&mut self, tilt_axis: UVector) -> &mut Self {
        self.tilt_axis = Some(tilt_axis);
        self
    }

    pub fn elements(&mut self, elements: Elements) -> &mut Self {
        self.elements = Some(elements);
        self
    }

    pub fn build(&self) -> Result<UnitCell> {
        let ctx = BuildContext {
            lattice_constant: self.lattice_constant.ok_or_else(|| Error::Generic("Lattice constant is not set".into()))?,
            elements: self.elements.as_ref().ok_or_else(|| Error::Generic("Elements are not set".into()))?,
            tilt_axis_angle: self.tilt_axis.zip(self.tilt_angle)
        };

        let geometry = self.geometry.ok_or_else(|| Error::Generic("Cell geometry not set".into()))?; 
    
        match geometry {
            CellGeometry::Cubic => Ok(build_cubic(ctx)?),
            CellGeometry::Tetragonal => Ok(build_tetragonal(ctx)?),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BuildContext<'a> {
    lattice_constant: f64, 
    elements: &'a Elements,
    tilt_axis_angle: Option<(UVector, f64)>,
}

fn build_cubic(ctx: BuildContext) -> Result<UnitCell> {
    let mut cell_atoms = get_cubic_ref_atoms(ctx.lattice_constant, ctx.elements)?;
    
    if let Some(axis_angle) = ctx.tilt_axis_angle {
        geometry::rotate_x_site_atoms(&mut cell_atoms.x, axis_angle);
    }
    
    Ok(UnitCell { cell_atoms, center_of_mass: Vector::zeros(), orientation: None })
}

fn build_tetragonal(ctx: BuildContext) ->Result<UnitCell> {
    todo!()
}

fn get_cubic_ref_atoms(
    lattice_constant: f64, 
    elements: &Elements
) -> Result<UnitCellAtoms> {

    let elements = if elements.len() == 3 { 
        (elements[0], elements[1], elements[2]) 
    } else {
        return Err(Error::Generic("Expected 3 elements".into()));
    };

    let ex = Vector::x();
    let ey = Vector::y();
    let ez = Vector::z();

    // convention: -y plane counter clockwise, y plane counter clockwise
    let a_site_atoms: Atoms = vec![
        Atom { atom_type: elements.0, position: (-0.5*lattice_constant*ex - 0.5*lattice_constant*ey - 0.5*lattice_constant*ez) },
        Atom { atom_type: elements.0, position: (0.5*lattice_constant*ex - 0.5*lattice_constant*ey - 0.5*lattice_constant*ez) },
        Atom { atom_type: elements.0, position: (0.5*lattice_constant*ex - 0.5*lattice_constant*ey + 0.5*lattice_constant*ez) },
        Atom { atom_type: elements.0, position: (-0.5*lattice_constant*ex - 0.5*lattice_constant*ey + 0.5*lattice_constant*ez) },
        Atom { atom_type: elements.0, position: (-0.5*lattice_constant*ex + 0.5*lattice_constant*ey - 0.5*lattice_constant*ez) },
        Atom { atom_type: elements.0, position: (0.5*lattice_constant*ex + 0.5*lattice_constant*ey - 0.5*lattice_constant*ez) },
        Atom { atom_type: elements.0, position: (0.5*lattice_constant*ex + 0.5*lattice_constant*ey + 0.5*lattice_constant*ez) },
        Atom { atom_type: elements.0, position: (-0.5*lattice_constant*ex + 0.5*lattice_constant*ey + 0.5*lattice_constant*ez) }
    ];

    let b_site_atoms: Atoms = vec![Atom {atom_type: elements.1, position: Vector::zeros()}];

    let x_site_atoms: Atoms = vec![
        Atom { atom_type: elements.2, position: (-0.5*lattice_constant*ey) },
        Atom { atom_type: elements.2, position: (-0.5*lattice_constant*ez) },
        Atom { atom_type: elements.2, position: (-0.5*lattice_constant*ex) },
        Atom { atom_type: elements.2, position: (0.5*lattice_constant*ey) },
        Atom { atom_type: elements.2, position: (0.5*lattice_constant*ez) },
        Atom { atom_type: elements.2, position: (0.5*lattice_constant*ex) }
    ];

    Ok(UnitCellAtoms { a: a_site_atoms, b: b_site_atoms, x: x_site_atoms })
}

fn get_tetragonal_atoms() -> Result<UnitCellAtoms> {
    todo!()
}

#[cfg(test)]
mod tests;
