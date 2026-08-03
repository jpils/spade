use super::*;
use crate::{cell::{self, CellGeometry, UnitCellOps, reference::RefCellBuilder}, types::{Atom, Atoms, Element, Elements, RotationVariant, UQuaternion}};
use nalgebra::Unit;
use std::f64::consts::PI;

#[test]
fn test_build_cell() {
    let (mut atoms, cell_matrix, geometry, elements, dw_type) = get_build_ctx_data();
    let q = UQuaternion::from_axis_angle(&Unit::new_normalize(Vector::new(0.0, 1.0, 0.0)), PI/4.0);
    atoms
        .iter_mut()
        .for_each(|atom| atom.position = q*atom.position);

    let ctx = BuildContext {
        geometry: geometry,
        atoms: &atoms,
        elements: &elements,
        cell_matrix: &cell_matrix,
        dw_type
    };

    let cell = build_cell(ctx).unwrap();

    let mut ref_cell = get_reference();
    ref_cell.rotate_inplace(&q);

    let (a1, b1, x1) = (&cell.0.cell_atoms.a, &cell.0.cell_atoms.b, &cell.0.cell_atoms.x);
    let (a2, b2, x2) = (&ref_cell.cell_atoms.a, &ref_cell.cell_atoms.b, &ref_cell.cell_atoms.x);

    for (&lhs, &rhs) in a1.iter().zip(a2) {
        assert!(approx_eq(lhs, rhs));
    }

    for (&lhs, &rhs) in b1.iter().zip(b2) {
        assert!(approx_eq(lhs, rhs));
    }

    assert!(approx_eq(x1[0], x2[0]));
    assert!(approx_eq(x1[3], x2[3]));
}

#[test]
fn test_global_frame() {
    let com = Vector::new(1.0, 0.0, 0.0);
    let vec = Vector::new(1.0, 1.0, 1.0);
    let atoms = vec![Atom { atom_type: None, position: vec }];

    let global = global_frame(atoms, com)[0];
    let ref_atom = Vector::new(2.0, 1.0, 1.0);
    assert_eq!(global.position, ref_atom);
}

#[test]
fn test_compute_com() {
    let atoms = atoms_cart();
    let a_site = atoms
        .iter()
        .filter(|atom| atom.atom_type == Some(Element::Sr))
        .collect::<Vec<_>>();

    let com = compute_com(&atoms).unwrap();
    let eps = Vector::new(f64::EPSILON, f64::EPSILON, f64::EPSILON);

    assert!(com > com-eps && com < com+eps);
}

#[test]
fn test_split_atoms() {
    let atoms = atoms_cart();
    let elements = vec![Element::Sr, Element::Ti, Element::O];
    let (a_site, b_site, x_site) = split_atoms(atoms, &elements).unwrap();

    assert!(a_site.iter().all(|atom| atom.atom_type == Some(elements[0])));
    assert!(b_site.iter().all(|atom| atom.atom_type == Some(elements[1])));
    assert!(x_site.iter().all(|atom| atom.atom_type == Some(elements[2])));
}

#[test]
#[should_panic]
fn test_split_atoms_fail() {
    let atoms = atoms_cart();
    let elements = vec![Element::Sr, Element::Ti];
    let (a_site, b_site, x_site) = split_atoms(atoms, &elements).unwrap();
}

#[test]
fn test_sort_atoms_twin() {
    let atoms = atoms_cart();
    let elements = vec![Element::Sr, Element::Ti, Element::O];
    let atoms_split = split_atoms(atoms, &elements).unwrap();
    let com = Vector::zeros();

    let (mut a_atoms, mut b_atoms, mut x_atoms) = sort_atoms(atoms_split, com, DWType::HT).unwrap();

    let a = 1.0;
    let mut a_site = vec![
        Atom { atom_type: Some(Element::Sr), position: Vector::new(-a/2.0, -a/2.0, -a/2.0)},
        Atom { atom_type: Some(Element::Sr), position: Vector::new( a/2.0, -a/2.0, -a/2.0)},
        Atom { atom_type: Some(Element::Sr), position: Vector::new( a/2.0, -a/2.0,  a/2.0)},
        Atom { atom_type: Some(Element::Sr), position: Vector::new(-a/2.0, -a/2.0,  a/2.0)},
        Atom { atom_type: Some(Element::Sr), position: Vector::new(-a/2.0,  a/2.0, -a/2.0)},
        Atom { atom_type: Some(Element::Sr), position: Vector::new( a/2.0,  a/2.0, -a/2.0)},
        Atom { atom_type: Some(Element::Sr), position: Vector::new( a/2.0,  a/2.0,  a/2.0)},
        Atom { atom_type: Some(Element::Sr), position: Vector::new(-a/2.0,  a/2.0,  a/2.0)}
    ];

    let mut x_site = vec![
        Atom { atom_type: Some(Element::O), position: Vector::new(0.0, -a/2.0, 0.0)},
        Atom { atom_type: Some(Element::O), position: Vector::new(0.0, 0.0, -a/2.0)},
        Atom { atom_type: Some(Element::O), position: Vector::new(-a/2.0, 0.0, 0.0)},
        Atom { atom_type: Some(Element::O), position: Vector::new(0.0,  a/2.0, 0.0)},
        Atom { atom_type: Some(Element::O), position: Vector::new(0.0, 0.0,  a/2.0)},
        Atom { atom_type: Some(Element::O), position: Vector::new( a/2.0, 0.0, 0.0)},
    ];

    // rotate into 45 degree rotated frame
    let q = UQuaternion::from_axis_angle(&Unit::new_normalize(Vector::new(0.0, 1.0, 0.0)), PI/4.0);
    a_site
        .iter_mut()
        .for_each(|atom| atom.position = q.clone()*atom.position);
    a_atoms
        .iter_mut()
        .for_each(|atom| atom.position = q.clone()*atom.position);

    x_site
        .iter_mut()
        .for_each(|atom| atom.position = q.clone()*atom.position);
    x_atoms
        .iter_mut()
        .for_each(|atom| atom.position = q*atom.position);
    
    for (&lhs, rhs) in a_site.iter().zip(a_atoms) {
        assert!(approx_eq(lhs, rhs), "A site error");
    }

    // at the time of UC construction only the pair along y side 
    // can be classified confidently
    assert!(approx_eq(x_atoms[0], x_site[0]), "X site error");
    assert!(approx_eq(x_atoms[3], x_site[3]), "X site error");
}

fn approx_eq(lhs: Atom, rhs: Atom) -> bool {
    let diff = lhs.position - rhs.position;
    if diff.norm_squared() <= f64::EPSILON {
        true
    } else {
        false
    }
}

#[test]
#[should_panic]
fn test_sort_atoms_apb() {
    todo!()
}

fn get_build_ctx_data() -> (Atoms, Matrix, CellGeometry, Elements, DWType) {
    let atoms = atoms_dir();
    let cell_matrix = Matrix::from_diagonal(&Vector::new(2.0, 2.0, 2.0));
    let geometry = CellGeometry::Cubic;
    let elements = vec![Element::Sr, Element::Ti, Element::O];
    let dw_type = DWType::HT;

    (atoms, cell_matrix, geometry, elements, dw_type)
}

fn atoms_cart() -> Atoms {
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

fn atoms_dir() -> Atoms {
    let a = 1.0;
    let mut atoms = Atoms::new();
    
    atoms.push(Atom { atom_type: Some(Element::Ti), position: Vector::zeros()});

    atoms.push(Atom { atom_type: Some(Element::Sr), position: Vector::new(-0.25, -0.25, -0.25)});
    atoms.push(Atom { atom_type: Some(Element::Sr), position: Vector::new(-0.25,  0.25, -0.25)});
    atoms.push(Atom { atom_type: Some(Element::Sr), position: Vector::new( 0.25, -0.25, -0.25)});
    atoms.push(Atom { atom_type: Some(Element::Sr), position: Vector::new( 0.25,  0.25, -0.25)});
    atoms.push(Atom { atom_type: Some(Element::Sr), position: Vector::new( 0.25, -0.25,  0.25)});
    atoms.push(Atom { atom_type: Some(Element::Sr), position: Vector::new( 0.25,  0.25,  0.25)});
    atoms.push(Atom { atom_type: Some(Element::Sr), position: Vector::new(-0.25, -0.25,  0.25)});
    atoms.push(Atom { atom_type: Some(Element::Sr), position: Vector::new(-0.25,  0.25,  0.25)});

    atoms.push(Atom { atom_type: Some(Element::O), position: Vector::new(0.0, -0.25, 0.0)});
    atoms.push(Atom { atom_type: Some(Element::O), position: Vector::new(0.0,  0.25, 0.0)});
    atoms.push(Atom { atom_type: Some(Element::O), position: Vector::new(0.0, 0.0, -0.25)});
    atoms.push(Atom { atom_type: Some(Element::O), position: Vector::new(0.0, 0.0,  0.25)});
    atoms.push(Atom { atom_type: Some(Element::O), position: Vector::new(-0.25, 0.0, 0.0)});
    atoms.push(Atom { atom_type: Some(Element::O), position: Vector::new( 0.25, 0.0, 0.0)});

    atoms
}

fn get_reference() -> UnitCell {
    let a = 1.0;
    let cell_matrix = Matrix::from_diagonal(&Vector::new(2.0, 2.0, 2.0));
    let geometry = CellGeometry::Cubic;
    let elements = vec![Element::Sr, Element::Ti, Element::O];

    let cell = RefCellBuilder::new()
        .elements(elements)
        .geometry(geometry)
        .lattice_constant(a)
        .rotation_variant(RotationVariant::R0)
        .build()
        .unwrap();
    
    cell.unit_cell
}
