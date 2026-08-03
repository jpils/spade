use std::cell::Ref;

use crate::{cell::{AsUnitCell, UnitCellOps}, geometry, prelude::*, types::RotationVariant};
use super::{CellGeometry, Elements, UVector, UnitCell, Error, Vector, UnitCellAtoms, Atoms, Atom};

pub struct RefCell {
    pub(crate) unit_cell: UnitCell,
    pub(crate) variant: RotationVariant
}

impl AsUnitCell for RefCell {
    fn as_unit_cell(&self) -> &UnitCell {
        &self.unit_cell
    }

    fn as_unit_cell_mut(&mut self) -> &mut UnitCell {
        &mut self.unit_cell
    }
}

impl UnitCellOps for RefCell {
    fn center_inplace(&mut self) -> &Self {
        self.unit_cell.center_inplace();
        self
    }

    fn centered_cell(&self) -> Self {
        let uc = self.unit_cell.centered_cell();
        Self {
            unit_cell: uc,
            variant: self.variant
        }
    }

    fn displacements_from<Cell: UnitCellOps>(&self, rhs: &Cell) -> Result<crate::types::AtomDisplacements> {
        self.unit_cell.displacements_from(rhs.as_unit_cell())
    }

    fn rotate_inplace(&mut self, q: &crate::types::UQuaternion) -> &Self {
        self.unit_cell.rotate_inplace(q);
        self
    }

    fn rotated_cell(&self, q: &crate::types::UQuaternion) -> Self {
        let uc = self.unit_cell.rotated_cell(q);
        Self { 
            unit_cell: uc, 
            variant: self.variant 
        }
    }
}

/// creates a unit cell with the atoms sitting at their high symmetry positions and the
/// `center_of_mass` in the origin of the coordinate system (cartesian coordinates)
/// tilt angle not given => 0.0
/// tilt axis is only valid with nonzero tilt angle
#[derive(Default, Debug)]
pub struct RefCellBuilder {
    geometry: Option<CellGeometry>,
    lattice_constant: Option<f64>,
    elements: Option<Elements>,
    tilt_angle: Option<f64>,
    tilt_axis: Option<UVector>,
    rotation_variant: Option<RotationVariant>
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

    pub fn rotation_variant(&mut self, rotation_variant: RotationVariant) -> &mut Self {
        self.rotation_variant = Some(rotation_variant);
        self
    }

    pub fn build(&self) -> Result<RefCell> {
        let validated_ctx = self.validate_fields()?;

        match validated_ctx.geometry {
            CellGeometry::Cubic => Ok(build_cubic(validated_ctx)?),
            CellGeometry::Tetragonal => Ok(build_tetragonal(validated_ctx)?),
        }
    }
}

impl RefCellBuilder {
    fn validate_fields(&self) -> Result<BuildContext<'_>> {
        let geometry = self.geometry
            .ok_or_else(|| Error::Generic("Cell geometry not set".into()))?; 
        let lattice_constant = self.lattice_constant
            .ok_or_else(|| Error::Generic("Lattice constant is not set".into()))?;
        let elements = self.elements.as_ref()
            .ok_or_else(|| Error::Generic("Elements are not set".into()))?; 
        let rotation_variant = self.rotation_variant
            .ok_or_else(|| Error::Generic("Rotation variant not specified".into()))?;
        let tilt_angle = match self.tilt_angle {
            Some(x) => x,
            None => 0.0
        };

        if !lattice_constant.is_finite() {
            return Err(Error::Generic("Lattice constant is infinite".into()))
        } else if lattice_constant <= 0.0 {
            return Err(Error::Generic("Lattice constant must be positive".into()))
        } else if lattice_constant.is_nan() {
            return Err(Error::Generic("Lattice constant is NAN".into()))
        }

        if elements.len() != 3 {
            return Err(Error::Generic("Elements must be 3 (currently only ABO_3 supported)".into()))
        }

        if !tilt_angle.is_finite() {
            return Err(Error::Generic("Tilt angle is not finite".into()))
        } else if tilt_angle == 0.0 && let Some(_) = self.tilt_axis {
            return Err(Error::Generic("Tilt angle is 0 but tilt axis was declared".into()))
        } else if tilt_angle != 0.0 && let None = self.tilt_axis {
            return Err(Error::Generic("Tilt angle is set but no tilt axis was declared".into())) 
        } 

        let ctx = BuildContext {
            geometry,
            lattice_constant, 
            elements,
            rotation_variant,
            tilt_axis_angle: self.tilt_axis.zip(Some(tilt_angle))
        };

        Ok(ctx)
    }
}

#[derive(Clone, Copy, Debug)]
struct BuildContext<'a> {
    geometry: CellGeometry,
    lattice_constant: f64, 
    elements: &'a Elements,
    rotation_variant: RotationVariant,
    tilt_axis_angle: Option<(UVector, f64)>,
}

fn build_cubic(ctx: BuildContext) -> Result<RefCell> {
    let mut cell_atoms = get_cubic_ref_atoms(ctx.lattice_constant, ctx.elements)?;
    
    if let Some(axis_angle) = ctx.tilt_axis_angle {
        geometry::rotate_x_site_atoms(&mut cell_atoms.x, axis_angle);
    }

    let uc = UnitCell { cell_atoms, center_of_mass: Vector::zeros() };

    Ok(RefCell {
        unit_cell: uc,
        variant: ctx.rotation_variant
    })
}

fn build_tetragonal(ctx: BuildContext) ->Result<RefCell> {
    todo!()
}

fn get_cubic_ref_atoms(
    lattice_constant: f64, 
    elements: &Elements
) -> Result<UnitCellAtoms> {

    let elements = (elements[0], elements[1], elements[2]);
    let ex = Vector::x();
    let ey = Vector::y();
    let ez = Vector::z();

    // convention: -y plane counter clockwise, y plane counter clockwise
    let a_site_atoms: Atoms = vec![
        Atom { atom_type: Some(elements.0), position: (-0.5*lattice_constant*ex - 0.5*lattice_constant*ey - 0.5*lattice_constant*ez) },
        Atom { atom_type: Some(elements.0), position: (0.5*lattice_constant*ex - 0.5*lattice_constant*ey - 0.5*lattice_constant*ez) },
        Atom { atom_type: Some(elements.0), position: (0.5*lattice_constant*ex - 0.5*lattice_constant*ey + 0.5*lattice_constant*ez) },
        Atom { atom_type: Some(elements.0), position: (-0.5*lattice_constant*ex - 0.5*lattice_constant*ey + 0.5*lattice_constant*ez) },
        Atom { atom_type: Some(elements.0), position: (-0.5*lattice_constant*ex + 0.5*lattice_constant*ey - 0.5*lattice_constant*ez) },
        Atom { atom_type: Some(elements.0), position: (0.5*lattice_constant*ex + 0.5*lattice_constant*ey - 0.5*lattice_constant*ez) },
        Atom { atom_type: Some(elements.0), position: (0.5*lattice_constant*ex + 0.5*lattice_constant*ey + 0.5*lattice_constant*ez) },
        Atom { atom_type: Some(elements.0), position: (-0.5*lattice_constant*ex + 0.5*lattice_constant*ey + 0.5*lattice_constant*ez) }
    ];

    let b_site_atoms: Atoms = vec![Atom {atom_type: Some(elements.1), position: Vector::zeros()}];

    let x_site_atoms: Atoms = vec![
        Atom { atom_type: Some(elements.2), position: (-0.5*lattice_constant*ey) },
        Atom { atom_type: Some(elements.2), position: (-0.5*lattice_constant*ez) },
        Atom { atom_type: Some(elements.2), position: (-0.5*lattice_constant*ex) },
        Atom { atom_type: Some(elements.2), position: (0.5*lattice_constant*ey) },
        Atom { atom_type: Some(elements.2), position: (0.5*lattice_constant*ez) },
        Atom { atom_type: Some(elements.2), position: (0.5*lattice_constant*ex) }
    ];

    Ok(UnitCellAtoms { a: a_site_atoms, b: b_site_atoms, x: x_site_atoms })
}

fn get_tetragonal_atoms() -> Result<UnitCellAtoms> {
    todo!()
}

fn permute_inplace(cell_atoms: &mut UnitCellAtoms, rotation_variant: RotationVariant) {
    match rotation_variant {
        RotationVariant::R0 => return,
        RotationVariant::R90 => {

        },
        RotationVariant::R180 => {

        },
        RotationVariant::R270 => {

        },
    }
    todo!()
}

#[cfg(test)]
mod tests;
