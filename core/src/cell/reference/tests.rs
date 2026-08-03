use std::f64::consts::PI;

use super::*;
use crate::types::{Element, Elements, RotationVariant};

#[test]
fn test_build_no_tilts() {
    let ctx = Context::cubic();
    let cell = RefCellBuilder::new()
        .elements(ctx.elements.clone())
        .lattice_constant(ctx.lattice_const)
        .geometry(ctx.geometry)
        .rotation_variant(ctx.rotation_variant)
        .build().unwrap();

    let a_atoms = cell.unit_cell.cell_atoms.a;
    let b_atoms = cell.unit_cell.cell_atoms.b;
    let x_atoms = cell.unit_cell.cell_atoms.x;

    check_a_atoms_position(&a_atoms);
    check_b_atoms_position(&b_atoms);
    check_x_atoms_position_untilted(&x_atoms);

    check_atoms_element(&a_atoms, ctx.elements[0]);
    check_atoms_element(&b_atoms, ctx.elements[1]);
    check_atoms_element(&x_atoms, ctx.elements[2]);

    assert_eq!(cell.unit_cell.center_of_mass, Vector::zeros());
    assert_eq!(cell.variant, ctx.rotation_variant)
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
        .rotation_variant(ctx.rotation_variant)
        .build().unwrap();

    let a_atoms = cell.unit_cell.cell_atoms.a;
    let b_atoms = cell.unit_cell.cell_atoms.b;
    let x_atoms = cell.unit_cell.cell_atoms.x;

    check_a_atoms_position(&a_atoms);
    check_b_atoms_position(&b_atoms);
    check_x_atoms_position_tilted(&x_atoms);

    check_atoms_element(&a_atoms, ctx.elements[0]);
    check_atoms_element(&b_atoms, ctx.elements[1]);
    check_atoms_element(&x_atoms, ctx.elements[2]);

    assert_eq!(cell.unit_cell.center_of_mass, Vector::zeros());
    assert_eq!(cell.variant, ctx.rotation_variant)
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
        .rotation_variant(ctx.rotation_variant)
        .build().unwrap();
}

#[test]
fn test_build_errors_when_geometry_missing() {
    let ctx = Context::cubic();
    let res = RefCellBuilder::new()
        .elements(ctx.elements)
        .lattice_constant(ctx.lattice_const)
        .rotation_variant(ctx.rotation_variant)
        .build();

    assert!(res.is_err());
}

#[test]
fn test_build_errors_when_lattice_constant_missing() {
    let ctx = Context::cubic();
    let res = RefCellBuilder::new()
        .elements(ctx.elements)
        .geometry(ctx.geometry)
        .rotation_variant(ctx.rotation_variant)
        .build();

    assert!(res.is_err());
}

#[test]
fn test_build_errors_when_elements_missing() {
    let ctx = Context::cubic();
    let res = RefCellBuilder::new()
        .lattice_constant(ctx.lattice_const)
        .geometry(ctx.geometry)
        .rotation_variant(ctx.rotation_variant)
        .build();

    assert!(res.is_err());
}

#[test]
fn test_build_errors_when_rotation_variant_missing() {
    let ctx = Context::cubic();
    let res = RefCellBuilder::new()
        .elements(ctx.elements)
        .lattice_constant(ctx.lattice_const)
        .geometry(ctx.geometry)
        .build();

    assert!(res.is_err());
}

#[test]
fn test_build_errors_when_element_count_is_not_three() {
    let ctx = Context::cubic();
    let res = RefCellBuilder::new()
        .elements(vec![Element::Sr, Element::Ti])
        .lattice_constant(ctx.lattice_const)
        .geometry(ctx.geometry)
        .rotation_variant(ctx.rotation_variant)
        .build();

    assert!(res.is_err());
}

#[test]
fn test_build_errors_when_lattice_constant_is_zero() {
    let res = valid_cubic_builder_with_lattice(0.0).build();

    assert!(res.is_err());
}

#[test]
fn test_build_errors_when_lattice_constant_is_negative() {
    let res = valid_cubic_builder_with_lattice(-1.0).build();

    assert!(res.is_err());
}

#[test]
fn test_build_errors_when_lattice_constant_is_nan() {
    let res = valid_cubic_builder_with_lattice(f64::NAN).build();

    assert!(res.is_err());
}

#[test]
fn test_build_errors_when_lattice_constant_is_infinite() {
    let res = valid_cubic_builder_with_lattice(f64::INFINITY).build();

    assert!(res.is_err());
}

#[test]
fn test_build_accepts_zero_tilt_angle_without_axis() {
    let ctx = Context::cubic();
    let res = RefCellBuilder::new()
        .elements(ctx.elements)
        .lattice_constant(ctx.lattice_const)
        .geometry(ctx.geometry)
        .rotation_variant(ctx.rotation_variant)
        .tilt_angle(0.0)
        .build();

    assert!(res.is_ok());
}

#[test]
fn test_build_errors_when_tilt_axis_is_set_without_angle() {
    let ctx = Context::cubic();
    let res = RefCellBuilder::new()
        .elements(ctx.elements)
        .lattice_constant(ctx.lattice_const)
        .geometry(ctx.geometry)
        .rotation_variant(ctx.rotation_variant)
        .tilt_axis(ctx.tilt_axis)
        .build();

    assert!(res.is_err());
}

#[test]
fn test_build_errors_when_tilt_axis_is_set_with_zero_angle() {
    let ctx = Context::cubic();
    let res = RefCellBuilder::new()
        .elements(ctx.elements)
        .lattice_constant(ctx.lattice_const)
        .geometry(ctx.geometry)
        .rotation_variant(ctx.rotation_variant)
        .tilt_axis(ctx.tilt_axis)
        .tilt_angle(0.0)
        .build();

    assert!(res.is_err());
}

#[test]
fn test_build_errors_when_nonzero_tilt_angle_has_no_axis() {
    let ctx = Context::cubic();
    let res = RefCellBuilder::new()
        .elements(ctx.elements)
        .lattice_constant(ctx.lattice_const)
        .geometry(ctx.geometry)
        .rotation_variant(ctx.rotation_variant)
        .tilt_angle(ctx.tilt_angle)
        .build();

    assert!(res.is_err());
}

#[test]
fn test_build_errors_when_tilt_angle_is_nan() {
    let ctx = Context::cubic();
    let res = RefCellBuilder::new()
        .elements(ctx.elements)
        .lattice_constant(ctx.lattice_const)
        .geometry(ctx.geometry)
        .rotation_variant(ctx.rotation_variant)
        .tilt_axis(ctx.tilt_axis)
        .tilt_angle(f64::NAN)
        .build();

    assert!(res.is_err());
}

#[test]
fn test_build_errors_when_tilt_angle_is_infinite() {
    let ctx = Context::cubic();
    let res = RefCellBuilder::new()
        .elements(ctx.elements)
        .lattice_constant(ctx.lattice_const)
        .geometry(ctx.geometry)
        .rotation_variant(ctx.rotation_variant)
        .tilt_axis(ctx.tilt_axis)
        .tilt_angle(f64::INFINITY)
        .build();

    assert!(res.is_err());
}

fn valid_cubic_builder_with_lattice(lattice_const: f64) -> RefCellBuilder {
    let ctx = Context::cubic();
    let mut builder = RefCellBuilder::new();
    builder
        .elements(ctx.elements)
        .lattice_constant(lattice_const)
        .geometry(ctx.geometry)
        .rotation_variant(ctx.rotation_variant);
    builder
}

struct Context {
    geometry: CellGeometry,
    lattice_const: f64,
    elements: Elements,
    tilt_angle: f64,
    tilt_axis: UVector,
    rotation_variant: RotationVariant,
}

impl Context {
    fn cubic() -> Context {
        Context { 
            geometry: CellGeometry::Cubic, 
            lattice_const: 1.0, 
            elements: vec![Element::Sr, Element::Ti, Element::O],
            tilt_angle: PI/2.0,
            tilt_axis: UVector::new_normalize(Vector::new(0.0, 0.0, 1.0)),
            rotation_variant: RotationVariant::R0,
        }
    }

    fn tetragonal() -> Context {
        Context { 
            geometry: CellGeometry::Tetragonal, 
            lattice_const: 1.0, 
            elements: vec![Element::Sr, Element::Ti, Element::O],
            tilt_angle: PI/2.0,
            tilt_axis: UVector::new_normalize(Vector::new(0.0, 0.0, 1.0)),
            rotation_variant: RotationVariant::R0,
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
        assert_eq!(atom.1.atom_type.unwrap(), element, "(idx, atom): {:#?}", atom)
    }
}
