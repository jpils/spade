use std::path::PathBuf;

use crate::args::AnalyzeArgs;
use crate::config::ConfigSpec;

use anyhow::{Context, Ok, Result};
use spade_core::{cell::{CellGeometry, simulation::{SimCell, SimCellBuilder}}, geometry::{self, cartesian_to_direct, n_nearest_neighbors}, parser::Parser, types::{AsVector, Atoms, CartAtoms, Cartesian, CoordinatesType, DWType, Direct, DirectAtom, DirectAtoms, Elements, ReadResult, SupercellAtoms, Vectors}};

pub(crate) fn run(spec: &ConfigSpec, args: &AnalyzeArgs) -> Result<()> {
    let data = read_data(&spec.io.file, spec.io.format)?;
    
    for frame in data {
        let cells = build_simulation_cells(
            &frame,
            spec.cell.geometry,
            spec.cell.dw_type
        );
    }

    Ok(())
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
    geometry: CellGeometry,
    dw_type: DWType,
) -> Result<Vec<SimCell>> {

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

    let supercell_atoms = SupercellAtoms::<Direct>::new(
        &positions, 
        frame.element_count[0], 
        frame.element_count[1],
        frame.element_count[2]
    ).with_context(|| "Failed to construct SupercellAtoms")?;

    let b_a_nns = n_nearest_neighbors(
        &supercell_atoms.b, 
        &supercell_atoms.a, 
        8, 
        &frame.cell_matrix)?;

    let b_x_nns = n_nearest_neighbors(
        &supercell_atoms.b, 
        &supercell_atoms.x, 
        6, 
        &frame.cell_matrix)?;

    let uc_atoms = b_a_nns.iter()
        .zip(b_x_nns.iter())
        .map(|(b_a_nn, b_x_nn)| {
            let mut atoms_chained: DirectAtoms = b_a_nn.neighbors.iter()
                .chain(b_x_nn.neighbors.iter())
                .map(|neighbor| {
                    let atom_cart = neighbor.atom_abs.clone();
                    let pos_direct = cartesian_to_direct(atom_cart.position, &frame.cell_matrix)?;

                    Ok(DirectAtom {
                        atom_type: atom_cart.atom_type,
                        position: pos_direct
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            let reference_pos_direct = cartesian_to_direct(b_a_nn.reference.position, &frame.cell_matrix)?;
            atoms_chained.push(DirectAtom {
                atom_type: b_a_nn.reference.atom_type,
                position: reference_pos_direct
            });

            Ok(atoms_chained)
        })
        .collect::<Result<Vec<_>>>()?;

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
