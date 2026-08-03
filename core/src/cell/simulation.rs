use crate::{cell::{AsUnitCell, UnitCellOps}, geometry, prelude::*, types::{CellAlignment, DWSide, DWType, Elements, Matrix}};
use super::{CellGeometry, UnitCellAtoms, Atoms, Vector, UQuaternion, Element, Vectors, UnitCell};

#[derive(Clone, Debug)]
pub struct SimCell(pub(crate) UnitCell);

impl AsUnitCell for SimCell {
    fn as_unit_cell(&self) -> &UnitCell {
        &self.0
    }

    fn as_unit_cell_mut(&mut self) -> &mut UnitCell {
        &mut self.0
    }
}

impl UnitCellOps for SimCell {
    fn center_inplace(&mut self) -> &Self {
        self.0.center_inplace();
        self
    }

    fn centered_cell(&self) -> Self {
        let uc = self.0.centered_cell();
        Self(uc)
    }

    fn displacements_from<Cell: UnitCellOps>(&self, rhs: &Cell) -> Result<crate::types::AtomDisplacements> {
        self.0.displacements_from(rhs.as_unit_cell())
    }

    fn rotate_inplace(&mut self, q: &crate::types::UQuaternion) -> &Self {
        self.0.rotate_inplace(q);
        self
    }

    fn rotated_cell(&self, q: &crate::types::UQuaternion) -> Self {
        let uc = self.0.rotated_cell(q);
        Self(uc)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FittedSimCell {
    pub(crate) unit_cell: UnitCell,
    pub(crate) alignment: CellAlignment
}

impl AsUnitCell for FittedSimCell {
    fn as_unit_cell(&self) -> &UnitCell {
        &self.unit_cell
    }
    
    fn as_unit_cell_mut(&mut self) -> &mut UnitCell {
        &mut self.unit_cell
    }
}

impl UnitCellOps for FittedSimCell {
    fn center_inplace(&mut self) -> &Self {
        self.unit_cell.center_inplace();
        self
    }

    fn centered_cell(&self) -> Self {
        let uc = self.unit_cell.centered_cell();
        Self {
            unit_cell: uc,
            alignment: self.alignment
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
            alignment: self.alignment 
        }
    }
}

#[derive(Default, Debug)]
struct SimCellBuilder {
    /// atoms in direct coordinates
    geometry: Option<CellGeometry>,
    elements: Option<Elements>,
    atoms: Option<Atoms>,
    cell_matrix: Option<Matrix>,
    dw_type: Option<DWType>,
}

impl SimCellBuilder {
    #[must_use]
    pub fn new() -> Self {
       SimCellBuilder::default()
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

    /// outputs unit cell with atom positions in cartesian coordinates
    pub fn build(&self) -> Result<SimCell> {
        let validated_ctx = self.validate_fields()?;

        match validated_ctx.geometry {
            CellGeometry::Cubic => Ok(cubic::build_cell(validated_ctx)?),
            CellGeometry::Tetragonal => Ok(tetragonal::build_cell(validated_ctx)?),
        }
    }
}

impl SimCellBuilder {
    fn validate_fields(&self) -> Result<BuildContext<'_>> {
        let geometry = self.geometry.ok_or_else(|| Error::Generic("Cell geometry not set".into()))?; 
        let elements = self.elements.as_ref().ok_or_else(|| Error::Generic("Elements are not set".into()))?; 
        let atoms = self.atoms.as_ref().ok_or_else(|| Error::Generic("Atoms are not set".into()))?; 
        let cell_matrix = self.cell_matrix.as_ref().ok_or_else(|| Error::Generic("Cell matrix is not set".into()))?; 
        let dw_type = self.dw_type.ok_or_else(|| Error::Generic("DW type is not set".into()))?; 

        if elements.len() != 3 {
            return Err(Error::Generic("Elements must be 3 (currently only ABO_3 supported)".into()));
        }

        match geometry {
            CellGeometry::Cubic => {
                if atoms.len() != 15 {
                    return Err(
                        Error::Generic(
                            format!("Atom count for building cubic md cell is {}, expected: 15", atoms.len())
                    ));
                }
            },
            CellGeometry::Tetragonal => return Err(Error::Generic("Tetragonal cells currently not supported".into()))
        }

       let ctx = BuildContext {
            geometry,
            atoms,
            elements,
            cell_matrix,
            dw_type,
        };

       Ok(ctx)
    }
}

#[derive(Clone, Copy, Debug)]
struct BuildContext<'a> {
    geometry: CellGeometry,
    atoms: &'a Atoms,
    elements: &'a Elements,
    cell_matrix: &'a Matrix,
    dw_type: DWType,
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
    }

    #[test]
    fn test_md_cell_builder_validates_complete_cubic_context() {
        let builder = prepare_builder();

        assert!(builder.validate_fields().is_ok());
    }

    #[test]
    fn test_md_cell_builder_errors_when_geometry_missing() {
        let mut builder = prepare_builder();
        builder.geometry = None;

        assert!(builder.validate_fields().is_err());
    }

    #[test]
    fn test_md_cell_builder_errors_when_elements_missing() {
        let mut builder = prepare_builder();
        builder.elements = None;

        assert!(builder.validate_fields().is_err());
    }

    #[test]
    fn test_md_cell_builder_errors_when_atoms_missing() {
        let mut builder = prepare_builder();
        builder.atoms = None;

        assert!(builder.validate_fields().is_err());
    }

    #[test]
    fn test_md_cell_builder_errors_when_cell_matrix_missing() {
        let mut builder = prepare_builder();
        builder.cell_matrix = None;

        assert!(builder.validate_fields().is_err());
    }

    #[test]
    fn test_md_cell_builder_errors_when_dw_type_missing() {
        let mut builder = prepare_builder();
        builder.dw_type = None;

        assert!(builder.validate_fields().is_err());
    }

    #[test]
    fn test_md_cell_builder_errors_when_element_count_is_not_three() {
        let mut builder = prepare_builder();
        builder.elements = Some(vec![Element::Sr, Element::Ti]);

        assert!(builder.validate_fields().is_err());
    }

    #[test]
    fn test_md_cell_builder_errors_when_cubic_atom_count_is_not_fifteen() {
        let mut builder = prepare_builder();
        builder.atoms = Some(vec![Atom { atom_type: Some(Element::Ti), position: Vector::zeros() }]);

        assert!(builder.validate_fields().is_err());
    }

    fn atoms() -> Atoms {
        let a = 1.0;
        let mut atoms = Atoms::new();
        atoms.push(Atom { atom_type: Some(Element::Ti), position: Vector::zeros()});

        atoms.push(Atom { atom_type: Some(Element::Sr), position: Vector::new(-a/2.0, -a/2.0, -a/2.0)});
        atoms.push(Atom { atom_type: Some(Element::Sr), position: Vector::new(-a/2.0,  a/2.0, -a/2.0)});
        atoms.push(Atom { atom_type: Some(Element::Sr), position: Vector::new( a/2.0, -a/2.0, -a/2.0)});
        atoms.push(Atom { atom_type: Some(Element::Sr), position: Vector::new( a/2.0,  a/2.0, -a/2.0)});
        atoms.push(Atom { atom_type: Some(Element::Sr), position: Vector::new( a/2.0, -a/2.0,  a/2.0)});
        atoms.push(Atom { atom_type: Some(Element::Sr), position: Vector::new( a/2.0,  a/2.0,  a/2.0)});
        atoms.push(Atom { atom_type: Some(Element::Sr), position: Vector::new(-a/2.0, -a/2.0,  a/2.0)});
        atoms.push(Atom { atom_type: Some(Element::Sr), position: Vector::new(-a/2.0,  a/2.0,  a/2.0)});

        atoms.push(Atom { atom_type: Some(Element::O), position: Vector::new(0.0, -a/2.0, 0.0)});
        atoms.push(Atom { atom_type: Some(Element::O), position: Vector::new(0.0,  a/2.0, 0.0)});
        atoms.push(Atom { atom_type: Some(Element::O), position: Vector::new(0.0, 0.0, -a/2.0)});
        atoms.push(Atom { atom_type: Some(Element::O), position: Vector::new(0.0, 0.0,  a/2.0)});
        atoms.push(Atom { atom_type: Some(Element::O), position: Vector::new(-a/2.0, 0.0, 0.0)});
        atoms.push(Atom { atom_type: Some(Element::O), position: Vector::new( a/2.0, 0.0, 0.0)});

        atoms
    }

    fn prepare_builder() -> SimCellBuilder {
        let atoms = atoms();
        let cell_matrix = Matrix::from_diagonal(&Vector::new(1.0, 1.0, 1.0));
        let orientation =UQuaternion::from_axis_angle(
                &Unit::new_normalize(Vector::new(0.0, 1.0, 0.0)), 
                0.0
            );
        let geometry = CellGeometry::Cubic;
        let elements = vec![Element::Sr, Element::Ti, Element::O];
        let dw_type = DWType::HT;

        let mut builder = SimCellBuilder::new();
        builder
            .geometry(geometry)
            .elements(elements.clone())
            .atoms(atoms)
            .cell_matrix(cell_matrix)
            .dw_type(dw_type);

        builder
    }

    fn get_reference() -> UnitCell {
        todo!() 
    }
}
