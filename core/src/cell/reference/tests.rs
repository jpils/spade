use std::f64::consts::PI;

use nalgebra::RealField;

use super::*;
use crate::types::{Element, Elements};

#[test]
fn test_build_no_tilts() {
    let ctx = Context::cubic();
    let cell = RefCellBuilder::new()
        .elements(ctx.elements.clone())
        .lattice_constant(ctx.lattice_const)
        .geometry(ctx.geometry)
        .build().unwrap();

    let a_atoms = cell.cell_atoms.a;
    let b_atoms = cell.cell_atoms.b;
    let x_atoms = cell.cell_atoms.x;

    check_a_atoms_position(&a_atoms);
    check_b_atoms_position(&b_atoms);
    check_x_atoms_position_untilted(&x_atoms);

    check_atoms_element(&a_atoms, ctx.elements[0]);
    check_atoms_element(&b_atoms, ctx.elements[1]);
    check_atoms_element(&x_atoms, ctx.elements[2]);

    assert_eq!(cell.center_of_mass, Vector::zeros())
}

#[test]
fn test_build_with_tilts() {
    let ctx = Context::cubic();
    let cell = RefCellBuilder::new()
        .elements(ctx.elements.clone())
        .lattice_constant(ctx.lattice_const)
        .geometry(ctx.geometry)
        .tilt_axis(ctx.tilt_axis)
        .tilt_angle(ctx.tilt_angle)
        .build().unwrap();

    let a_atoms = cell.cell_atoms.a;
    let b_atoms = cell.cell_atoms.b;
    let x_atoms = cell.cell_atoms.x;

    check_a_atoms_position(&a_atoms);
    check_b_atoms_position(&b_atoms);
    check_x_atoms_position_tilted(&x_atoms);

    check_atoms_element(&a_atoms, ctx.elements[0]);
    check_atoms_element(&b_atoms, ctx.elements[1]);
    check_atoms_element(&x_atoms, ctx.elements[2]);

    assert_eq!(cell.center_of_mass, Vector::zeros())
}

#[test]
#[should_panic]
fn test_build_invalid_geometry() {
    let ctx = Context::tetragonal();
    let cell = RefCellBuilder::new()
        .elements(ctx.elements.clone())
        .lattice_constant(ctx.lattice_const)
        .geometry(ctx.geometry)
        .tilt_axis(ctx.tilt_axis)
        .tilt_angle(ctx.tilt_angle)
        .build().unwrap();
}

struct Context {
    geometry: CellGeometry,
    lattice_const: f64,
    elements: Elements,
    tilt_angle: f64,
    tilt_axis: UVector
}

impl Context {
    fn cubic() -> Context {
        Context { 
            geometry: CellGeometry::Cubic, 
            lattice_const: 1.0, 
            elements: vec![Element::Sr, Element::Ti, Element::O],
            tilt_angle: PI/2.0,
            tilt_axis: UVector::new_normalize(Vector::new(0.0, 0.0, 1.0))
        }
    }

    fn tetragonal() -> Context {
        Context { 
            geometry: CellGeometry::Tetragonal, 
            lattice_const: 1.0, 
            elements: vec![Element::Sr, Element::Ti, Element::O],
            tilt_angle: PI/2.0,
            tilt_axis: UVector::new_normalize(Vector::new(0.0, 0.0, 1.0))
        }
    }
}

fn check_a_atoms_position(a_atoms: &Atoms) {
    // -y plane
    assert!(a_atoms[0].position.x.is_sign_negative());
    assert!(a_atoms[0].position.y.is_sign_negative());
    assert!(a_atoms[0].position.z.is_sign_negative());

    assert!(a_atoms[1].position.x.is_sign_positive());
    assert!(a_atoms[1].position.y.is_sign_negative());
    assert!(a_atoms[1].position.z.is_sign_negative());

    assert!(a_atoms[2].position.x.is_sign_positive());
    assert!(a_atoms[2].position.y.is_sign_negative());
    assert!(a_atoms[2].position.z.is_sign_positive());

    assert!(a_atoms[3].position.x.is_sign_negative());
    assert!(a_atoms[3].position.y.is_sign_negative());
    assert!(a_atoms[3].position.z.is_sign_positive());

    // +y plane
    assert!(a_atoms[4].position.x.is_sign_negative());
    assert!(a_atoms[4].position.y.is_sign_positive());
    assert!(a_atoms[4].position.z.is_sign_negative());

    assert!(a_atoms[5].position.x.is_sign_positive());
    assert!(a_atoms[5].position.y.is_sign_positive());
    assert!(a_atoms[5].position.z.is_sign_negative());

    assert!(a_atoms[6].position.x.is_sign_positive());
    assert!(a_atoms[6].position.y.is_sign_positive());
    assert!(a_atoms[6].position.z.is_sign_positive());

    assert!(a_atoms[7].position.x.is_sign_negative());
    assert!(a_atoms[7].position.y.is_sign_positive());
    assert!(a_atoms[7].position.z.is_sign_positive());
}

fn check_b_atoms_position(b_atoms: &Atoms) {
    assert_eq!(b_atoms[0].position, Vector::zeros())
}

fn check_x_atoms_position_untilted(x_atoms: &Atoms) {
    // front
    assert_eq!(x_atoms[0].position.x, 0.0);
    assert!(x_atoms[0].position.y.is_sign_negative());
    assert_eq!(x_atoms[0].position.z, 0.0);
    //bottom
    assert_eq!(x_atoms[1].position.x, 0.0);
    assert_eq!(x_atoms[1].position.y, 0.0);
    assert!(x_atoms[1].position.z.is_sign_negative());
    //left
    assert!(x_atoms[2].position.x.is_sign_negative());
    assert_eq!(x_atoms[2].position.y, 0.0);
    assert_eq!(x_atoms[2].position.z, 0.0);
    // back
    assert_eq!(x_atoms[3].position.x, 0.0);
    assert!(x_atoms[3].position.y.is_sign_positive());
    assert_eq!(x_atoms[3].position.z, 0.0);
    //top
    assert_eq!(x_atoms[4].position.x, 0.0);
    assert_eq!(x_atoms[4].position.y, 0.0);
    assert!(x_atoms[4].position.z.is_sign_positive());
    //right
    assert!(x_atoms[5].position.x.is_sign_positive());
    assert_eq!(x_atoms[5].position.y, 0.0);
    assert_eq!(x_atoms[5].position.z, 0.0);

}

fn check_x_atoms_position_tilted(x_atoms: &Atoms) {
    let eps = 1e-12;

    // x_atoms[0] (originally front: [0.0, -0.5, 0.0] -> rotated: [0.5, 0.0, 0.0])
    assert!((x_atoms[0].position.x - 0.5).abs() < eps);
    assert!(x_atoms[0].position.y.abs() < eps);
    assert!(x_atoms[0].position.z.abs() < eps);

    // x_atoms[1] (originally bottom: [0.0, 0.0, -0.5] -> rotated: [0.0, 0.0, -0.5])
    assert!(x_atoms[1].position.x.abs() < eps);
    assert!(x_atoms[1].position.y.abs() < eps);
    assert!((x_atoms[1].position.z - (-0.5)).abs() < eps);

    // x_atoms[2] (originally left: [-0.5, 0.0, 0.0] -> rotated: [0.0, -0.5, 0.0])
    assert!(x_atoms[2].position.x.abs() < eps);
    assert!((x_atoms[2].position.y - (-0.5)).abs() < eps);
    assert!(x_atoms[2].position.z.abs() < eps);

    // x_atoms[3] (originally back: [0.0, 0.5, 0.0] -> rotated: [-0.5, 0.0, 0.0])
    assert!((x_atoms[3].position.x - (-0.5)).abs() < eps);
    assert!(x_atoms[3].position.y.abs() < eps);
    assert!(x_atoms[3].position.z.abs() < eps);

    // x_atoms[4] (originally top: [0.0, 0.0, 0.5] -> rotated: [0.0, 0.0, 0.5])
    assert!(x_atoms[4].position.x.abs() < eps);
    assert!(x_atoms[4].position.y.abs() < eps);
    assert!((x_atoms[4].position.z - 0.5).abs() < eps);

    // x_atoms[5] (originally right: [0.5, 0.0, 0.0] -> rotated: [0.0, 0.5, 0.0])
    assert!(x_atoms[5].position.x.abs() < eps);
    assert!((x_atoms[5].position.y - 0.5).abs() < eps);
    assert!(x_atoms[5].position.z.abs() < eps);
}

fn check_atoms_element(atoms: &Atoms, element: Element) {
    for atom in atoms.iter().enumerate() {
        assert_eq!(atom.1.atom_type, element, "(idx, atom): {:#?}", atom)
    }
}
