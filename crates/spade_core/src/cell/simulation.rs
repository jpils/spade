use crate::{geometry, prelude::*, types::{DWType, Elements, Matrix}};
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
            atoms:  self.atoms.as_ref().ok_or_else(|| Error::Generic("Atom positions are not set".into()))?,
            elements: self.elements.as_ref().ok_or_else(|| Error::Generic("Elements for mapping not set".into()))?,
            cell_matrix: self.cell_matrix.as_ref().ok_or_else(|| Error::Generic("Cell matrix is not set".into()))?,
            dw_type: self.dw_type.ok_or_else(|| Error::Generic("DW type is not set".into()))?,
            orientation: self.orientation
        };

        let geometry = self.geometry.ok_or_else(|| Error::Generic("Cell Geometry is not set".into()))?;

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

mod cubic {
    use super::{geometry, BuildContext, Result, UnitCell, UnitCellAtoms, Elements, Atoms, Vector, DWType, Matrix};
    use crate::{error::Error, types::{Atom, DWSide}};

    pub(super) fn build_cell(ctx: BuildContext) -> Result<UnitCell> {
        let atoms_dir = ctx.atoms.clone();

        let com = compute_com(&atoms_dir)?;
        let com_cart = geometry::direct_to_cartesian(com, ctx.cell_matrix);

        let atoms_com_frame = com_frame_cart(atoms_dir, com, ctx.cell_matrix);
        let atoms_split = split_atoms(atoms_com_frame, ctx.elements)?;
        let (a_site, b_site, x_site) = sort_atoms(atoms_split, com_cart, ctx.dw_type)?;

        let uc_atoms = UnitCellAtoms {
            a: global_frame(a_site, com),
            b: global_frame(b_site, com),
            x: global_frame(x_site, com)
        };

        Ok(UnitCell {
            cell_atoms: uc_atoms,
            center_of_mass: com_cart,
            orientation: ctx.orientation
        })
    }

    fn global_frame(mut atoms: Atoms, com: Vector) -> Atoms {
        atoms.iter_mut()
            .map(|atom| atom.position += com);
        atoms
    }

    fn compute_com(atoms: &Atoms) -> Result<Vector> {
        let (weighted_sum, total_mass) = atoms
            .iter()
            .try_fold((Vector::zeros(), 0.0), |(acc_weighted_sum, acc_total_mass), atom| {
                let mass = atom.atom_type
                    .get_mass()
                    .ok_or_else(|| Error::Generic("Atom type mass unknown".into()))?;
                Ok::<_, Error>((acc_weighted_sum + mass * atom.position, acc_total_mass + mass))
            })?;

        Ok(weighted_sum/total_mass)
    }

    fn split_atoms(atoms: Atoms, elements: &Elements) -> Result<(Atoms, Atoms, Atoms)> {
        let mut a_site_atoms: Atoms = Vec::with_capacity(8);
        let mut b_site_atoms: Atoms = Vec::with_capacity(1);
        let mut x_site_atoms: Atoms = Vec::with_capacity(6);

        for atom in atoms {
            match atom.atom_type {
                t if t == elements[0] => a_site_atoms.push(atom),
                t if t == elements[1] => b_site_atoms.push(atom),
                t if t == elements[2] => x_site_atoms.push(atom),
                _ => return Err(Error::Generic("Atom type not found in elements provided".into()))
            }
        }

        Ok((a_site_atoms, b_site_atoms, x_site_atoms))
    }

    fn sort_atoms(
        atoms_split: (Atoms, Atoms, Atoms), 
        com: Vector, 
        dw_type: DWType
    ) -> Result<(Atoms, Atoms, Atoms)> {

        let (a_site, b_site, x_site) = atoms_split;

        match dw_type {
            DWType::HT | DWType::HH => {
                let a_site_sorted = sort_a_site_atoms_twin(a_site, com)?;
                let x_site_sorted = sort_x_site_atoms_twin(x_site, com)?;

                Ok((a_site_sorted, b_site, x_site_sorted))
            },
            DWType::APB => todo!()
        }
    }

    fn sort_a_site_atoms_twin(atoms: Atoms, com: Vector) -> Result<Atoms> {
        let (front, back) = split_front_back(atoms, com);
        
        let atoms_angle_front: Vec<(Atom, f64)> = get_atom_angle_pairs(front);
        let atoms_angle_back: Vec<(Atom, f64)>  = get_atom_angle_pairs(back);

        let sorted_atoms_front = get_a_site_sequence(atoms_angle_front)?;
        let sorted_atoms_back = get_a_site_sequence(atoms_angle_back)?;

        let mut sorted_atoms = Vec::with_capacity(sorted_atoms_back.len() + sorted_atoms_front.len());
        sorted_atoms.extend(sorted_atoms_front);
        sorted_atoms.extend(sorted_atoms_back);

        Ok(sorted_atoms)
    }

    fn sort_x_site_atoms_twin(atoms: Atoms, com: Vector) -> Result<Atoms> {
        let x_pairs = find_opposite(atoms, com)?;
        let sorted_atoms = get_x_site_sequence(x_pairs)?;
        Ok(sorted_atoms)
    }

    /// find opposite atom by finding the largest scalar product from the reference atom to the
    /// other atoms
    fn find_opposite(mut atoms: Atoms, com: Vector) -> Result<Vec<(Atom, Atom)>> {
        let mut x_pairs: Vec<(Atom, Atom)> = Vec::with_capacity(atoms.len()/2);

        while !atoms.is_empty() {
            let current_atom = atoms.swap_remove(0);
            let current_vec = (current_atom.position - com).normalize();

            let cosines = atoms.iter()
                .map(|atom| {
                    let vec = (atom.position - com).normalize();
                    let cos = current_vec.dot(&vec);
                    cos.abs()
                })
                .collect::<Vec<_>>();

            let opposite_id = cosines.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(id, _)| id)
                .ok_or_else(|| Error::Generic("Couldn't find maximum".into()))?;

            let opposite_atom = atoms.swap_remove(opposite_id);
            x_pairs.push((current_atom, opposite_atom));
        }

        Ok(x_pairs)
    }

    fn get_x_site_sequence(mut pairs: Vec<(Atom, Atom)>) -> Result<Atoms> {
        let y_pair_id = get_pair_id(&pairs, Vector::y())?;
        let y_pair = pairs.swap_remove(y_pair_id);

        let x_pair_id = get_pair_id(&pairs, Vector::new(1.0, 0.0, -1.0))?;
        let x_pair = pairs.swap_remove(x_pair_id);

        let z_pair = pairs.remove(0);

        let x_site_atoms = Atoms::with_capacity(pairs.len()*2);

        let mut x_site_atoms = vec![y_pair.0, x_pair.0, z_pair.0, y_pair.1, x_pair.1, z_pair.1];

        Ok(x_site_atoms)
    }

    fn get_pair_id(mut pairs: &Vec<(Atom, Atom)>, axis: Vector) -> Result<usize> {
        pairs.iter()
            .enumerate()
            .map(|(id, (a, b))| {
                let vec = b.position - a.position;
                let vec = vec.normalize();
                let projection = vec.dot(&axis);
                (id, projection.abs())
            })
            .max_by(|(id1, x), (id2, y)| x.total_cmp(y))
            .ok_or_else(|| Error::Generic("Couldn't find maximum".into()))
            .map(|(id, _)| id)
    }

    fn get_a_site_sequence(atoms: Vec<(Atom, f64)>) -> Result<Atoms> {
        let rightmost_idx = atoms
            .iter()
            .enumerate()
            .max_by(|(_, (atom1, _)), (_, (atom2, _))| atom1.position[0].total_cmp(&atom2.position[0]))
            .map(|(id, _)| id) 
            .ok_or_else(|| Error::Generic("atoms empty".into()))?;

        let mut atoms = atoms;
        let rightmost_atom = atoms.swap_remove(rightmost_idx);
        atoms.sort_by(|pair1, pair2| pair1.1.total_cmp(&pair2.1));
        atoms.push(rightmost_atom);
        atoms.rotate_left(1);

        let sorted_atoms = atoms.into_iter()
            .map(|(atom, _)| atom)
            .collect();

        Ok(sorted_atoms)
    }

    fn get_atom_angle_pairs(atoms: Atoms) -> Vec<(Atom, f64)> {
        atoms
            .into_iter()
            .map(|atom| (atom, geometry::compute_x_angle(&atom.position)))
            .collect::<Vec<_>>()
    }

    fn split_front_back(atoms: Atoms, com: Vector) -> (Atoms, Atoms) {
        let mut front = Atoms::with_capacity(4); // -y plane
        let mut back = Atoms::with_capacity(4);

        for atom in atoms {
            if atom.position[1] < com[1] {
                front.push(atom);
            } else {
                back.push(atom);
            }
        }

        (front, back)
    }

    fn com_frame_cart(mut atoms: Atoms, com: Vector, cell_matrix: &Matrix) -> Atoms {
        for atom in &mut atoms {
            let centered_pos = atom.position - com;
            let wrapped_centered_pos = geometry::wrap_vector(&centered_pos);
            let wrapped_centered_pos = geometry::direct_to_cartesian(wrapped_centered_pos, cell_matrix);
            atom.position = wrapped_centered_pos;
        }

        atoms
    }
}

mod tetragonal {
    use super::{BuildContext, Result, UnitCell};
    pub(super) fn build_cell(ctx: BuildContext) -> Result<UnitCell> {
        todo!()
    }
}

#[cfg(test)] 
mod tests;
