use std::path::PathBuf;

use crate::args::AnalyzeArgs;
use crate::config::ConfigSpec;

use anyhow::{Context, Error, Ok, Result};
use spade_core::{cell::{AsUnitCell, CellGeometry, simulation::{FittedSimCell, SimCell, SimCellBuilder}}, classification::{descriptor::{BDescriptor, BbDescriptors, calculate_descriptors}, kmeans::{Clusters, descriptor_closest_to_center, k_means}}, geometry::{self, NearestNeighbors, cartesian_to_direct, n_nearest_neighbors}, parser::Parser, types::{AsVector, Atom, Atoms, CartAtom, CartAtoms, Cartesian, CoordinatesType, DWType, Direct, DirectAtom, DirectAtoms, Elements, Matrix, ReadResult, SupercellAtoms, Vectors}};

pub(crate) fn run(spec: &ConfigSpec, args: &AnalyzeArgs) -> Result<()> {
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

        let domain_seed_ids = find_domain_seeds_ids(
            &nearest_neighbors.b_b_nns, 
            &nearest_neighbors.b_x_nns,
            &sim_cells,
            spec.analysis.fitting.n_clusters.unwrap_or(2)
        )?;

        let fitted_cells = fit_sim_cells(
            sim_cells, 
            domain_seed_ids, 
            spec.cell.geometry, 
            spec.cell.dw_type
        )?;
    }

    Ok(())
}

struct NNs {
    b_a_nns: Vec<NearestNeighbors>,
    b_b_nns: Vec<NearestNeighbors>,
    b_x_nns: Vec<NearestNeighbors>
}

fn get_nearest_neighbors(supercell_atoms: &SupercellAtoms<Direct>, cell_matrix: &Matrix) -> Result<NNs> {
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

fn read_data(file: &PathBuf, format: Parser) -> Result<Vec<ReadResult>> {
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
    sim_cells: &Vec<SimCell>,
    n_clusters: usize
) -> Result<Vec<usize>> {

    let bb_descriptors = calculate_descriptors(b_b_nns, b_x_nns)?;
    let clusters = k_means(&bb_descriptors, n_clusters);

    // change to x values instead
    let bulk_bounds = clusters.iter()
        .map(|cluster| {
            let (min_id, _) = cluster.bb_descriptors.iter()
                .enumerate()
                .min_by(|a, b| 
                    a.1.reference_b_atom.position.as_vector().x.total_cmp(
                    &b.1.reference_b_atom.position.as_vector().x)
                )?;
            let (max_id, _) = cluster.bb_descriptors.iter()
                .enumerate()
                .max_by(|a, b| 
                    a.1.reference_b_atom.position.as_vector().x.total_cmp(
                    &b.1.reference_b_atom.position.as_vector().x)
                )?;
            Some((min_id, max_id))
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| anyhow::anyhow!("Clusters are empty"));

    // find uc in the center of this
    todo!()
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
    sim_cells: Vec<SimCell>, 
    seeds_ids: Vec<usize>, 
    geometry: CellGeometry,
    dw_type: DWType
) -> Result<Vec<FittedSimCell>> {

    match geometry {
        CellGeometry::Cubic => fit_cubic_sim_cells(sim_cells, seeds_ids, dw_type),
        CellGeometry::Tetragonal => todo!()
    };

    todo!()
}

fn fit_cubic_sim_cells(
    sim_cells: Vec<SimCell>, 
    seeds_ids: Vec<usize>,
    dw_type: DWType
) -> Result<FittedSimCell> {
    
    match dw_type {
        DWType::HT => fit_cubic_ht(sim_cells, seeds_ids),
        DWType::HH => fit_cubic_hh(sim_cells, seeds_ids),
        DWType::Apb => fit_cubic_apb(sim_cells, seeds_ids)
    };
    
    todo!()
}

fn fit_cubic_ht(sim_cells: Vec<SimCell>, seeds_ids: Vec<usize>) {
    todo!()
}

fn fit_cubic_hh(sim_cells: Vec<SimCell>, seeds_ids: Vec<usize>) {
    todo!()
}

fn fit_cubic_apb(sim_cells: Vec<SimCell>, seeds_ids: Vec<usize>) {
    todo!()
}
