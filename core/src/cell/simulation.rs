use crate::{geometry, prelude::*, types::{DWSide, DWType, Elements, Matrix}};
use super::{CellGeometry, UnitCellAtoms, Atoms, Vector, UQuaternion, Element, Vectors, UnitCell};

#[derive(Default, Debug)]
struct MDCellBuilder {
    /// atoms in direct coordinates
    geometry: Option<CellGeometry>,
    elements: Option<Elements>,
    atoms: Option<Atoms>,
    cell_matrix: Option<Matrix>,
    dw_type: Option<DWType>,
    orientation: Option<UQuaternion>
}

impl MDCellBuilder {
    #[must_use]
    pub fn new() -> Self {
       MDCellBuilder::default()
    }

    pub fn geometry(&mut self, geometry: CellGeometry) -> &mut Self {
        self.geometry = Some(geometry);
        self
    }

    pub fn elements(&mut self, elements: Elements) -> &mut Self {
        self.elements = Some(elements);
        self
    }

    pub fn atoms(&mut self, atoms: Atoms) -> &mut Self {
        self.atoms = Some(atoms);
        self
    }

    pub fn cell_matrix(&mut self, cell_matrix: Matrix) -> &mut Self {
        self.cell_matrix = Some(cell_matrix);
        self
    }

    pub fn dw_type(&mut self, dw_type: DWType) -> &mut Self {
        self.dw_type = Some(dw_type);
        self
    }

    pub fn orientation(&mut self, orientation: UQuaternion) -> &mut Self {
       self.orientation = Some(orientation);
       self
    }

    /// outputs unit cell with atom positions in cartesian coordinates
    pub fn build(&self) -> Result<UnitCell> {
        let ctx = BuildContext {
            atoms:  self.atoms.as_ref().ok_or_else(|| Error::Generic("Atom positions not set".into()))?,
            elements: self.elements.as_ref().ok_or_else(|| Error::Generic("Elements mapping not set".into()))?,
            cell_matrix: self.cell_matrix.as_ref().ok_or_else(|| Error::Generic("Cell matrix not set".into()))?,
            dw_type: self.dw_type.ok_or_else(|| Error::Generic("DW type is not set".into()))?,
            orientation: self.orientation
        };

        let geometry = self.geometry.ok_or_else(|| Error::Generic("Cell Geometry not set".into()))?;

        match geometry {
            CellGeometry::Cubic => Ok(cubic::build_cell(ctx)?),
            CellGeometry::Tetragonal => Ok(tetragonal::build_cell(ctx)?),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BuildContext<'a> {
    atoms: &'a Atoms,
    elements: &'a Elements,
    cell_matrix: &'a Matrix,
    dw_type: DWType,
    orientation: Option<UQuaternion>,
}

mod cubic;
mod tetragonal;

#[cfg(test)] 
mod tests {
    use nalgebra::Unit;

    use super::*;
    use crate::types::{Atom};
    use crate::cell::reference::{self, RefCellBuilder};

    #[test]
    fn test_md_cell_builder_setters() {
        let builder = prepare_builder();

        let atoms = atoms();
        let cell_matrix = Matrix::from_diagonal(&Vector::new(1.0, 1.0, 1.0));
        let orientation =UQuaternion::from_axis_angle(
                &Unit::new_normalize(Vector::new(0.0, 1.0, 0.0)), 
                0.0
            );
        let geometry = CellGeometry::Cubic;
        let elements = vec![Element::Sr, Element::Ti, Element::O];
        let dw_type = DWType::HT;

        assert_eq!(builder.geometry.unwrap(), geometry);
        assert_eq!(builder.elements.unwrap(), elements);
        assert_eq!(builder.atoms.unwrap(), atoms);
        assert_eq!(builder.cell_matrix.unwrap(), cell_matrix);
        assert_eq!(builder.dw_type.unwrap(), dw_type);
        assert_eq!(builder.orientation.unwrap(), orientation);
    }

    fn atoms() -> Atoms {
        let a = 1.0;
        let mut atoms = Atoms::new();
        atoms.push(Atom { atom_type: Element::Ti, position: Vector::zeros()});

        atoms.push(Atom { atom_type: Element::Sr, position: Vector::new(-a/2.0, -a/2.0, -a/2.0)});
        atoms.push(Atom { atom_type: Element::Sr, position: Vector::new(-a/2.0,  a/2.0, -a/2.0)});
        atoms.push(Atom { atom_type: Element::Sr, position: Vector::new( a/2.0, -a/2.0, -a/2.0)});
        atoms.push(Atom { atom_type: Element::Sr, position: Vector::new( a/2.0,  a/2.0, -a/2.0)});
        atoms.push(Atom { atom_type: Element::Sr, position: Vector::new( a/2.0, -a/2.0,  a/2.0)});
        atoms.push(Atom { atom_type: Element::Sr, position: Vector::new( a/2.0,  a/2.0,  a/2.0)});
        atoms.push(Atom { atom_type: Element::Sr, position: Vector::new(-a/2.0, -a/2.0,  a/2.0)});
        atoms.push(Atom { atom_type: Element::Sr, position: Vector::new(-a/2.0,  a/2.0,  a/2.0)});

        atoms.push(Atom { atom_type: Element::O, position: Vector::new(0.0, -a/2.0, 0.0)});
        atoms.push(Atom { atom_type: Element::O, position: Vector::new(0.0,  a/2.0, 0.0)});
        atoms.push(Atom { atom_type: Element::O, position: Vector::new(0.0, 0.0, -a/2.0)});
        atoms.push(Atom { atom_type: Element::O, position: Vector::new(0.0, 0.0,  a/2.0)});
        atoms.push(Atom { atom_type: Element::O, position: Vector::new(-a/2.0, 0.0, 0.0)});
        atoms.push(Atom { atom_type: Element::O, position: Vector::new( a/2.0, 0.0, 0.0)});

        atoms
    }

    fn prepare_builder() -> MDCellBuilder {
        let atoms = atoms();
        let cell_matrix = Matrix::from_diagonal(&Vector::new(1.0, 1.0, 1.0));
        let orientation =UQuaternion::from_axis_angle(
                &Unit::new_normalize(Vector::new(0.0, 1.0, 0.0)), 
                0.0
            );
        let geometry = CellGeometry::Cubic;
        let elements = vec![Element::Sr, Element::Ti, Element::O];
        let dw_type = DWType::HT;

        let mut builder = MDCellBuilder::new();
        builder
            .geometry(geometry)
            .elements(elements.clone())
            .atoms(atoms)
            .cell_matrix(cell_matrix)
            .dw_type(dw_type)
            .orientation(orientation);

        builder
    }

    fn get_reference() -> UnitCell {
        todo!() 
    }
}
