use super::{geometry, BuildContext, Result, UnitCell, UnitCellAtoms, Elements, Atoms, Vector, DWType, Matrix};
use crate::{cell::simulation::SimCell, error::Error, types::{AsVector, Atom, CartAtom, CartAtoms, Cartesian, DWSide, Direct, DirectAtoms, UQuaternion}};

pub(super) fn build_cell(ctx: BuildContext) -> Result<SimCell> {
    let atoms_dir = ctx.atoms.clone();

    let com = compute_com(&atoms_dir)?;
    let com_cart = geometry::direct_to_cartesian(com, ctx.cell_matrix);

    let atoms_com_frame = com_frame_cart(atoms_dir, com, ctx.cell_matrix)?;
    let atoms_split = split_atoms(atoms_com_frame, ctx.elements)?;
    let (a_site, b_site, x_site) = sort_atoms(atoms_split, ctx.dw_type)?;

    let uc_atoms = UnitCellAtoms {
        a: global_frame(a_site, com_cart),
        b: global_frame(b_site, com_cart),
        x: global_frame(x_site, com_cart)
    };

    let uc = UnitCell {
        cell_atoms: uc_atoms,
        center_of_mass: com_cart,
    };

    Ok(SimCell(uc))
}

pub(super) fn reorder_x_sites(
    atoms: &mut CartAtoms,
    b_pos: Cartesian,
    q: &UQuaternion,
) -> Result<()> {
    if atoms.len() != 6 {
        return Err(Error::Generic("Expected six X-site atoms".into()));
    }

    // Keep the signed y pair at indices 0 and 3.
    let slots = [1, 2, 4, 5];

    // Transform B-X displacements into the fitted local frame.
    let local = slots.map(|i| {
        let displacement =
            atoms[i].position.as_vector() - b_pos.as_vector();

        q.inverse_transform_vector(&displacement)
    });

    if local.iter().any(|v| v.norm_squared() < 1e-20) {
        return Err(Error::Generic("Zero-length B-X bond".into()));
    }

    // Reference order of the four remaining sites.
    let directions = [
        -Vector::z(),
        -Vector::x(),
         Vector::z(),
         Vector::x(),
    ];

    let mut best = [0, 1, 2, 3];
    let mut best_score = f64::NEG_INFINITY;

    for a in 0..4 {
        for b in 0..4 {
            if b == a { continue; }

            for c in 0..4 {
                if c == a || c == b { continue; }

                let d = 6 - a - b - c;
                let permutation = [a, b, c, d];

                let score: f64 = permutation
                    .iter()
                    .enumerate()
                    .map(|(j, &i)| {
                        local[i].normalize().dot(&directions[j])
                    })
                    .sum();

                if score > best_score {
                    best_score = score;
                    best = permutation;
                }
            }
        }
    }

    // Reorder the existing atoms without changing their positions.
    let original = slots.map(|i| atoms[i]);

    for (slot, index) in slots.into_iter().zip(best) {
        atoms[slot] = original[index];
    }

    Ok(())
}


fn global_frame(mut atoms: CartAtoms, com: Cartesian) -> CartAtoms {
    for atom in &mut atoms {
        atom.position += com;
    }

    atoms
}

fn compute_com(atoms: &DirectAtoms) -> Result<Direct> {
    let (weighted_sum, total_mass) = atoms
        .iter()
        .try_fold((Direct::new(Vector::zeros())?, 0.0), |(acc_weighted_sum, acc_total_mass), atom| {
            let mass = atom.atom_type
                .ok_or_else(|| Error::Generic("Atom type is set to None".into()))?
                .get_mass();
            Ok::<_, Error>((acc_weighted_sum + atom.position * mass, acc_total_mass + mass))
        })?;

    Ok(weighted_sum/total_mass)
}

fn split_atoms(atoms: CartAtoms, elements: &Elements) -> Result<(CartAtoms, CartAtoms, CartAtoms)> {
    let mut a_site_atoms: CartAtoms = Vec::with_capacity(8);
    let mut b_site_atoms: CartAtoms = Vec::with_capacity(1);
    let mut x_site_atoms: CartAtoms = Vec::with_capacity(6);

    for atom in atoms {
        match atom.atom_type {
            Some(t) if t == elements[0] => a_site_atoms.push(atom),
            Some(t) if t == elements[1] => b_site_atoms.push(atom),
            Some(t) if t == elements[2] => x_site_atoms.push(atom),
            None => return Err(Error::Generic("Atom type is set to None".into())),
            _ => return Err(Error::Generic("Atom type not found in elements provided".into()))
        }
    }

    Ok((a_site_atoms, b_site_atoms, x_site_atoms))
}

///Atoms in COM frame
fn sort_atoms(
    atoms_split: (CartAtoms, CartAtoms, CartAtoms), 
    dw_type: DWType
) -> Result<(CartAtoms, CartAtoms, CartAtoms)> {

    let (a_site, b_site, x_site) = atoms_split;

    match dw_type {
        DWType::HT | DWType::HH => {
            let a_site_sorted = sort_a_site_atoms_twin(a_site)?;
            let x_site_sorted = sort_x_site_atoms_twin(x_site)?;

            Ok((a_site_sorted, b_site, x_site_sorted))
        },
        DWType::Apb => {
            let a_site_sorted = sort_a_site_atoms_apb(a_site)?;
            let x_site_sorted = sort_x_site_atoms_apb(x_site)?;

            Ok((a_site_sorted, b_site, x_site_sorted))
        }
    }
}

fn sort_a_site_atoms_twin(atoms: CartAtoms) -> Result<CartAtoms> {
    let (front, back) = split_front_back(atoms);
    
    let atoms_angle_front: Vec<(CartAtom, f64)> = get_atom_angle_pairs(front);
    let atoms_angle_back: Vec<(CartAtom, f64)>  = get_atom_angle_pairs(back);

    let sorted_atoms_front = get_a_site_sequence(atoms_angle_front)?;
    let sorted_atoms_back = get_a_site_sequence(atoms_angle_back)?;

    let mut sorted_atoms = Vec::with_capacity(sorted_atoms_back.len() + sorted_atoms_front.len());
    sorted_atoms.extend(sorted_atoms_front);
    sorted_atoms.extend(sorted_atoms_back);

    Ok(sorted_atoms)
}

fn sort_x_site_atoms_twin(atoms: CartAtoms) -> Result<CartAtoms> {
    let x_pairs = find_opposite(atoms)?;
    let sorted_atoms = get_x_site_sequence(x_pairs)?;
    Ok(sorted_atoms)
}

#[allow(clippy::needless_pass_by_value)] // TODO get rid of this
fn sort_a_site_atoms_apb(atoms: CartAtoms) -> Result<CartAtoms> {
    todo!()
}

#[allow(clippy::needless_pass_by_value)] // TODO get rid of this
fn sort_x_site_atoms_apb(atoms: CartAtoms) -> Result<CartAtoms> {
    todo!()
}

/// find opposite atom by finding the largest scalar product from the reference atom to the
/// other atoms
fn find_opposite(mut atoms: CartAtoms) -> Result<Vec<(CartAtom, CartAtom)>> {
    let mut x_pairs: Vec<(CartAtom, CartAtom)> = Vec::with_capacity(atoms.len()/2);

    while !atoms.is_empty() {
        let current_atom = atoms.swap_remove(0);
        let current_vec = current_atom.position.as_vector().normalize();

        let cosines = atoms.iter()
            .map(|atom| {
                let vec = atom.position.as_vector().normalize();
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

fn get_x_site_sequence(mut pairs: Vec<(CartAtom, CartAtom)>) -> Result<CartAtoms> {
    let y_pair_id = get_pair_id(&pairs, Cartesian::new(Vector::y()))?;
    let y_pair = pairs.swap_remove(y_pair_id);

    let x_pair_id = get_pair_id(&pairs, Cartesian::new(Vector::new(1.0, 0.0, -1.0)))?;
    let x_pair = pairs.swap_remove(x_pair_id);

    let z_pair = pairs.remove(0);

    let mut x_site_atoms = vec![y_pair.0, x_pair.0, z_pair.0, y_pair.1, x_pair.1, z_pair.1];

    Ok(x_site_atoms)
}

fn get_pair_id(mut pairs: &Vec<(CartAtom, CartAtom)>, axis: Cartesian) -> Result<usize> {
    pairs.iter()
        .enumerate()
        .map(|(id, (a, b))| {
            let vec = b.position - a.position;
            let vec = vec.as_vector().normalize();
            let projection = vec.dot(axis.as_vector());
            (id, projection.abs())
        })
        .max_by(|(id1, x), (id2, y)| x.total_cmp(y))
        .ok_or_else(|| Error::Generic("Couldn't find maximum".into()))
        .map(|(id, _)| id)
}

fn get_a_site_sequence(atoms: Vec<(CartAtom, f64)>) -> Result<CartAtoms> {
    let rightmost_idx = atoms
        .iter()
        .enumerate()
        .max_by(|(_, (atom1, _)), (_, (atom2, _))| atom1.position.as_vector().x.total_cmp(&atom2.position.as_vector().x))
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

fn get_atom_angle_pairs(atoms: CartAtoms) -> Vec<(CartAtom, f64)> {
    atoms
        .into_iter()
        .map(|atom| (atom, geometry::compute_x_angle(&atom.position)))
        .collect::<Vec<_>>()
}

fn split_front_back(atoms: CartAtoms) -> (CartAtoms, CartAtoms) {
    let mut front = Atoms::with_capacity(4); // -y plane
    let mut back = Atoms::with_capacity(4);

    for atom in atoms {
        if atom.position.as_vector().y.is_sign_negative() {
            front.push(atom);
        } else {
            back.push(atom);
        }
    }

    (front, back)
}

fn com_frame_cart(atoms: DirectAtoms, com: Direct, cell_matrix: &Matrix) -> Result<CartAtoms> {
    let mut atoms_cart: CartAtoms = Vec::with_capacity(atoms.len());

    for atom in atoms {
        let centered_pos = atom.position - com;
        let wrapped_centered_pos = geometry::wrap_vector(&centered_pos)?;
        let wrapped_centered_pos = geometry::direct_to_cartesian(wrapped_centered_pos, cell_matrix);

        let atom_cart = CartAtom {
            atom_type: atom.atom_type,
            position: wrapped_centered_pos
        };

        atoms_cart.push(atom_cart);
    }

    Ok(atoms_cart)
}

#[cfg(test)]
mod tests;
