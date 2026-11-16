use std::{cmp::min_by, path::PathBuf};

use crate::args::AnalyzeArgs;
use crate::config::ConfigSpec;

use anyhow::{Context, Error, Ok, Result};
use serde::de;
use spade_core::{cell::{AsUnitCell, CellGeometry, UnitCellOps, reference::RefCellBuilder, simulation::{FittedSimCell, SimCell, SimCellBuilder}}, classification::{descriptor::{BDescriptor, BDescriptors, DescribedCell, calculate_descriptors}, kmeans::{Clusters, descriptor_closest_to_domain_center, k_means}}, geometry::{self, NearestNeighbors, cartesian_to_direct, n_nearest_neighbors}, optimization::{GradientDescentCtx, gradient_descent}, parser::Parser, types::{AsVector, Atom, Atoms, CartAtom, CartAtoms, Cartesian, CellAlignment, CoordinatesType, DWType, Direct, DirectAtom, DirectAtoms, Elements, Matrix, PermutationVariant, ReadResult, SupercellAtoms, UQuaternion, Vector, Vectors}};
use std::f64::consts::{FRAC_PI_2, PI, TAU};

pub(crate) fn run(
    spec: &ConfigSpec, 
    args: &AnalyzeArgs
) -> Result<()> {

    let data = read_data(&spec.io.file, spec.io.format)?;
    
    for frame in data {
        let positions = positions_to_direct(&frame)?;

        let supercell_atoms = SupercellAtoms::<Direct>::new(
            &positions, 
            frame.element_count[0], 
            frame.element_count[1],
            frame.element_count[2]
        ).with_context(|| "Failed to construct SupercellAtoms")?;

        let nearest_neighbors = get_nearest_neighbors(
            &supercell_atoms, 
            &frame.cell_matrix
        )?;

        let sim_cells = build_simulation_cells(
            &frame,
            &nearest_neighbors,
            spec.cell.geometry,
            spec.cell.dw_type
        )?;

        let (domain_seed_ids, described_cells) = find_domain_seeds_ids(
            &nearest_neighbors.b_b_nns, 
            &nearest_neighbors.b_x_nns,
            sim_cells,
            spec.analysis.fitting.n_clusters.unwrap_or(2)
        )?;

        let fitted_cells = fit_sim_cells(
            described_cells, 
            domain_seed_ids, 
            spec
        )?;
    }

    Ok(())
}

struct NNs {
    b_a_nns: Vec<NearestNeighbors>,
    b_b_nns: Vec<NearestNeighbors>,
    b_x_nns: Vec<NearestNeighbors>
}

fn get_nearest_neighbors(
    supercell_atoms: &SupercellAtoms<Direct>,
    cell_matrix: &Matrix
) -> Result<NNs> {

    let b_a_nns = n_nearest_neighbors(
        &supercell_atoms.b, 
        &supercell_atoms.a, 
        8, 
        cell_matrix)?;

    let b_b_nns = n_nearest_neighbors(
        &supercell_atoms.b, 
        &supercell_atoms.b, 
        6, 
        cell_matrix)?;

    let b_x_nns = n_nearest_neighbors(
        &supercell_atoms.b, 
        &supercell_atoms.x, 
        6, 
        cell_matrix)?;

    Ok(NNs {
        b_a_nns,
        b_b_nns,
        b_x_nns
    })
}

fn read_data(
    file: &PathBuf, 
    format: Parser
)-> Result<Vec<ReadResult>> {
    format.read(file)
        .with_context(|| format!(
            "Failed to read input data from {}", 
            file.display())
        )
}

fn build_simulation_cells(
    frame: &ReadResult,
    nearest_neighbors: &NNs,
    geometry: CellGeometry,
    dw_type: DWType,
) -> Result<Vec<SimCell>> {

    let b_a_nns = &nearest_neighbors.b_a_nns;
    let b_x_nns = &nearest_neighbors.b_x_nns;

    let uc_atoms = grouped_uc_atoms(b_a_nns, b_x_nns, &frame.cell_matrix)?;

    let elements: Elements = frame.element_count.iter()
        .map(|&(elem, _)| elem)
        .collect();

    let mut sim_cell_builder = SimCellBuilder::new();
    let mut sim_cells: Vec<SimCell> = Vec::with_capacity(uc_atoms.len());

    for atoms in uc_atoms {
        sim_cell_builder
            .atoms(atoms.clone())
            .elements(elements.clone())
            .geometry(geometry)
            .dw_type(dw_type);
        
        let sim_cell = sim_cell_builder.build()
            .with_context(|| format!("Failed to build SimulationCell atoms {:#?}", atoms))?;

        sim_cells.push(sim_cell)
    }

    Ok(sim_cells)
}

fn positions_to_direct(frame: &ReadResult) -> Result<Vec<Direct>> {
    let positions: Vec<Direct> = match frame.coordinate_type {
        CoordinatesType::Direct => {
            frame.positions.iter()
                .map(|pos| Direct::new(*pos))
                .collect::<spade_core::prelude::Result<Vec<_>>>()?
        },
        CoordinatesType::Cartesian => {
            let mut positions_direct = Vec::with_capacity(frame.positions.len());
            for pos in frame.positions.clone() {
                let pos_cart = Cartesian::new(pos);
                let pos_direct = cartesian_to_direct(pos_cart, &frame.cell_matrix)
                    .with_context(|| 
                        "Failed to convert positions from cartesian to direct!"
                    )?;
                positions_direct.push(pos_direct);
            }

            positions_direct
        }
    };

    Ok(positions)
}

fn grouped_uc_atoms(
    b_a_nns: &Vec<NearestNeighbors>, 
    b_x_nns: &Vec<NearestNeighbors>, 
    cell_matrix: &Matrix
) -> Result<Vec<DirectAtoms>> {
    b_a_nns.iter()
        .zip(b_x_nns.iter())
        .map(|(b_a_nn, b_x_nn)| {
            let mut atoms_chained: DirectAtoms = b_a_nn.neighbors.iter()
                .chain(b_x_nn.neighbors.iter())
                .map(|neighbor| {
                    let atom_cart = neighbor.atom_abs;
                    let pos_direct = cartesian_to_direct(atom_cart.position, cell_matrix)?;

                    Ok(DirectAtom {
                        atom_type: atom_cart.atom_type,
                        position: pos_direct
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            let reference_pos_direct = cartesian_to_direct(b_a_nn.reference.position, cell_matrix)?;
            atoms_chained.push(DirectAtom {
                atom_type: b_a_nn.reference.atom_type,
                position: reference_pos_direct
            });

            Ok(atoms_chained)
        })
        .collect::<Result<Vec<_>>>()
}

fn find_domain_seeds_ids(
    b_b_nns: &Vec<NearestNeighbors>,
    b_x_nns: &Vec<NearestNeighbors>,
    sim_cells: Vec<SimCell>,
    n_clusters: usize
) -> Result<(Vec<usize>, Clusters)> {

    let descriptors = calculate_descriptors(b_b_nns, b_x_nns)?;
    let described_cells = descriptors.iter()
        .zip(sim_cells)
        .map(|(desc, cell)| DescribedCell {b_descriptor: *desc, cell: cell})
        .collect::<Vec<_>>();

    let clusters = k_means(described_cells, n_clusters);

    let bulk_centers = clusters.iter()
        .map(|cluster| {
            let  min_x = cluster.described_cells.iter()
                .min_by(|&a, &b| 
                    a.b_descriptor.reference_b_atom.position.as_vector().x.total_cmp(
                    &b.b_descriptor.reference_b_atom.position.as_vector().x)
                )
                .map(|desc| desc.b_descriptor.reference_b_atom.position.as_vector().x)?;
            let max_x = cluster.described_cells.iter()
                .max_by(|&a, &b| 
                    a.b_descriptor.reference_b_atom.position.as_vector().x.total_cmp(
                    &b.b_descriptor.reference_b_atom.position.as_vector().x)
                )
                .map(|desc| desc.b_descriptor.reference_b_atom.position.as_vector().x)?;

            Some((max_x - min_x) / 2.0)
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| anyhow::anyhow!("Finding Domain seeds: Clusters are empty"))?;

    let mut bulk_center_cell_ids: Vec<usize> = Vec::with_capacity(bulk_centers.len());
    for (center, cluster) in bulk_centers.iter().zip(clusters.iter()) {
        let dists: Vec<(f64, usize)> = cluster.described_cells.iter()
            .enumerate()
            .map(|(id, described_cell)| {
                let dist = (center - described_cell.cell.as_unit_cell().cell_atoms.b[0].position.as_vector().x)
                    .abs();

                (dist, id)
            })
            .collect();
        
        let (_, min_cell_id) = dists.iter()
            .min_by(|&(a_dist, a_id), &(b_dist, b_id)| a_dist.total_cmp(b_dist))
            .ok_or_else(|| anyhow::anyhow!("Finding Domain seeds: dists is empty"))?;

        bulk_center_cell_ids.push(*min_cell_id);
    }

    Ok((bulk_center_cell_ids, clusters))
}

fn find_cell_id_to_descriptor(
    sim_cells: &Vec<SimCell>, 
    descriptor: &BDescriptor
) -> Option<usize> {

    sim_cells.iter()
        .position(|cell| 
            cell.as_unit_cell().cell_atoms.b[0] == descriptor.reference_b_atom
        )
}

fn fit_sim_cells(
    clusters: Clusters, 
    seeds_ids: Vec<usize>, 
    spec: &ConfigSpec
) -> Result<Vec<FittedSimCell>> {

    match spec.cell.geometry {
        CellGeometry::Cubic => fit_cubic_sim_cells(clusters, seeds_ids, spec),
        CellGeometry::Tetragonal => todo!()
    }
}

fn fit_cubic_sim_cells(
    clusters: Clusters, 
    seeds_ids: Vec<usize>,
    spec: &ConfigSpec
) -> Result<Vec<FittedSimCell>> {
    
    match spec.cell.dw_type {
        DWType::HT | DWType::HH => fit_cubic_twin(clusters, seeds_ids, spec),
        DWType::Apb => fit_cubic_apb(clusters, seeds_ids)
    }
}

fn fit_cubic_twin(
    clusters: Clusters, 
    seeds_ids: Vec<usize>, 
    spec: &ConfigSpec
) -> Result<Vec<FittedSimCell>> {

    anyhow::ensure!(clusters.len() == 2, "Twin fitting requires 2 clusters");

    let seed_cell = &clusters[0].described_cells[seeds_ids[0]].cell;
    let (first_perm, q1) = find_first_permutation(seed_cell, spec)?;
    let (second_perm, q2) = find_second_permutation(&q1, first_perm, spec.cell.dw_type);

    let n_cells = clusters.iter().fold(0, |acc, cluster| acc + cluster.described_cells.len());
    let mut fitted_cells: Vec<FittedSimCell> = Vec::with_capacity(n_cells);

    for ((cluster, permutation), init_q) in clusters.into_iter().zip([first_perm, second_perm]).zip([q1, q2]) {
        let ref_cell = RefCellBuilder::new()
            .geometry(spec.cell.geometry)
            .permutation_variant(permutation)
            .lattice_constant(spec.cell.lattice_constant)
            .elements(spec.cell.elements.clone().into())
            .build()?;

        for cell in cluster.described_cells {
            let grad_ctx = GradientDescentCtx {
                reference_uc: &ref_cell,
                centered_simulation_uc: &cell.cell.centered_cell(),
                lattice_const: spec.cell.lattice_constant,
                init_quaternion: init_q,
                max_iter_gradient_descent: spec.analysis.fitting.max_iter
                    .unwrap_or(100),
                max_iter_line_search: spec.analysis.fitting.max_iter_line_search
                    .unwrap_or(10),
                step_size: spec.analysis.fitting.step_size
                    .unwrap_or(0.01),
                max_step_size_adjustments: spec.analysis.fitting.max_step_size_adjustments
                    .unwrap_or(4)
            };
            
            let q = gradient_descent(&grad_ctx)?;

            let mut cell = cell;
            cell.cell.reorder_x_sites(&q)?;

            let cell_alignment = CellAlignment {
                unit_quaternion: q,
                ref_variant: permutation,
            };

            let fitted_cell = FittedSimCell {
                unit_cell: cell.cell.0,
                alignment: cell_alignment
            };

            fitted_cells.push(fitted_cell);
        }
    }

    Ok(fitted_cells)
}

pub fn find_second_permutation(
    quaternion: &UQuaternion,
    variant: PermutationVariant,
    dw_type: DWType,
) -> (PermutationVariant, UQuaternion) {

    if dw_type == DWType::Apb {
        return (PermutationVariant::R0, UQuaternion::identity());
    }

    let z_rotated = quaternion.transform_vector(&Vector::z());
    let angle = z_rotated.z.atan2(z_rotated.x).rem_euclid(TAU);

    let quadrant_2_or_4 =
        (FRAC_PI_2 < angle && angle < PI)
        || (3.0 * FRAC_PI_2 < angle && angle < TAU);

    let positive_rotation = match dw_type {
        DWType::HT => !quadrant_2_or_4,
        DWType::HH => quadrant_2_or_4,
        DWType::Apb => unreachable!(),
    };

    let rotation_angle = if positive_rotation {
        FRAC_PI_2
    } else {
        -FRAC_PI_2
    };

    let rot_y = UQuaternion::from_axis_angle(
        &Vector::y_axis(),
        rotation_angle,
    );

    let second_quaternion = quaternion * rot_y;

    let second_variant = match (variant, positive_rotation) {
        (PermutationVariant::R0, true) => PermutationVariant::R90,
        (PermutationVariant::R90, true) => PermutationVariant::R180,
        (PermutationVariant::R180, true) => PermutationVariant::R270,
        (PermutationVariant::R270, true) => PermutationVariant::R0,

        (PermutationVariant::R0, false) => PermutationVariant::R270,
        (PermutationVariant::R90, false) => PermutationVariant::R0,
        (PermutationVariant::R180, false) => PermutationVariant::R90,
        (PermutationVariant::R270, false) => PermutationVariant::R180,
    };

    (second_variant, second_quaternion)
}

fn find_first_permutation(seed_cell: &SimCell, spec: &ConfigSpec) -> Result<(PermutationVariant, UQuaternion)> {
    let get_permutation_variant = |id: usize| -> Result<PermutationVariant> {
        let variant = match id {
            0 => PermutationVariant::R0,
            1 => PermutationVariant::R90,
            2 => PermutationVariant::R180,
            3 => PermutationVariant::R270,
            _ => unreachable!("There are only 4 penmutation variants")
        };

        Ok(variant)
    };

    let perm_dists = (0..4).into_iter()
        .map(|id| {
            let ref_cell = RefCellBuilder::new()
                .geometry(spec.cell.geometry)
                .permutation_variant(get_permutation_variant(id)?)
                .lattice_constant(spec.cell.lattice_constant)
                .elements(spec.cell.elements.clone().into())
                .build()?;

            let grad_ctx = GradientDescentCtx {
                reference_uc: &ref_cell,
                centered_simulation_uc: &seed_cell.centered_cell(),
                lattice_const: spec.cell.lattice_constant,
                init_quaternion: UQuaternion::identity(),
                max_iter_gradient_descent: spec.analysis.fitting.max_iter
                    .unwrap_or(100),
                max_iter_line_search: spec.analysis.fitting.max_iter_line_search
                    .unwrap_or(10),
                step_size: spec.analysis.fitting.step_size
                    .unwrap_or(0.01),
                max_step_size_adjustments: spec.analysis.fitting.max_step_size_adjustments
                    .unwrap_or(4)
            };

            let uqaternion = gradient_descent(&grad_ctx)?;
            let rotated_ref_cell = ref_cell.rotated_cell(&uqaternion);
            let displacements = seed_cell.as_unit_cell()
                .displacements_from(rotated_ref_cell.as_unit_cell())?;

            let avg_y_dist = displacements.x[0].as_vector().norm_squared()
                + displacements.x[3].as_vector().norm_squared() 
                / 2.0;

            Ok((id, avg_y_dist, uqaternion))
        })
        .collect::<Result<Vec<_>>>()?;

    let (id, min, uquaternion) = perm_dists.iter()
        .min_by(|(_, a_dist, _), (_, b_dist, _)| a_dist.total_cmp(b_dist))
        .ok_or_else(|| anyhow::anyhow!("perm_dists is empty"))?;

    match id {
        0 => Ok((PermutationVariant::R0, *uquaternion)),
        1 => Ok((PermutationVariant::R90,*uquaternion)),
        2 => Ok((PermutationVariant::R180, *uquaternion)),
        3 => Ok((PermutationVariant::R270, *uquaternion)),
        _ => unreachable!("There are only 4 penmutation variants")
    }
}

fn fit_cubic_apb(clusters: Clusters, seeds_ids: Vec<usize>) -> Result<Vec<FittedSimCell>> {
    todo!()
}
