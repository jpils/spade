#![allow(unused)]

use spade_core::{error::Error, geometry, parser::Parser, types::{Atom, Atoms, Matrix, ReadResultVariant, SupercellAtoms, Vectors}};
use nalgebra::{SVector, DMatrix};
use ndarray::{Array2};
use std::error::Error as err;
use std::fs::File;
use csv::Writer;
mod descriptor;
mod kmeans;

use descriptor::*;
use kmeans::*;

fn save_clusters_to_csv(
    clusters: &Vec<Descriptors>,   // each cluster is a Vec<Descriptor>
    file_path: &str,
) -> Result<(), Box<dyn err>> {
    let mut wtr = Writer::from_path(file_path)?;

    let n_features = clusters[0][0].len();
    let header: Vec<String> = std::iter::once("cluster_id".to_owned())
        .chain((0..n_features).map(|i| format!("d{}", i+1)))
        .collect();
    wtr.write_record(&header)?;

    for (cluster_id, cluster) in clusters.iter().enumerate() {
        for point in cluster {
            let mut record = vec![cluster_id.to_string()];
            // `point.iter()` works if Descriptor derefs to `[f64]` or implements Iterator
            record.extend(
                point.iter().map(|val| val.to_string())
            );
            wtr.write_record(&record)?;
        }
    }

    wtr.flush()?;
    Ok(())
}

fn to_ndarray(data: &Descriptors) -> Array2<f64> {
    let flat: Vec<f64> = data.iter().flat_map(|desc| desc.iter().copied()).collect();
    Array2::from_shape_vec((data.len(), 18), flat).unwrap()
}

fn perform_pca_nalgebra(data: &Array2<f64>, k: usize) -> DMatrix<f64> {
    // 1. Convert ndarray::Array2 to nalgebra::DMatrix
    // ndarray is row-major, so we use from_row_slice
    let (rows, cols) = (data.nrows(), data.ncols());
    let mut matrix = DMatrix::from_row_slice(rows, cols, data.as_slice().unwrap());

    // 2. Center the data (subtract mean from each column)
    for i in 0..matrix.ncols() {
        let col_mean = matrix.column(i).mean();
        matrix.column_mut(i).add_scalar_mut(-col_mean);
    }

    // 3. Singular Value Decomposition
    // We want V_t (the right-singular vectors) which are the Principal Components
    let svd = matrix.clone().svd(true, true);
    let v_t = svd.v_t.expect("SVD decomposition failed");

    // 4. Project the data
    // We take the first 'k' components (rows of V_t)
    let components = v_t.rows(0, k).transpose();
    
    // Result = Centered Data * Top K Components
    matrix * components
}

fn save_pca_to_csv(matrix: &nalgebra::DMatrix<f64>, file_path: &str) -> Result<(), Box<dyn err>> {
    let mut wtr = Writer::from_path(file_path)?;

    // 1. Write Header
    wtr.write_record(&["PC1", "PC2", "PC3"])?;

    // 2. Iterate through rows (the 800 samples)
    for row in matrix.row_iter() {
        // Convert the nalgebra row view into a record
        let record: Vec<String> = row.iter()
            .map(|val| val.to_string())
            .collect();
        wtr.write_record(record)?;
    }

    wtr.flush()?;
    Ok(())
}

fn main() -> Result<(), Error> {
    let parser = Parser::Poscar;    
    if let ReadResultVariant::ReadResult { res } = parser.read("./POSCAR_HT_2")? {
        let a_num = res.element_count[0].1;
        let b_num = res.element_count[1].1;
        let x_num = res.element_count[2].1;

        let atoms = SupercellAtoms::new(&res.positions, a_num, b_num, x_num)?;
        let b_atoms = atoms.b_site;
        let b_x = n_nearest_neighbors(&b_atoms, atoms.x_site, 6, &res.cell_matrix);
        let b_b = n_nearest_neighbors(&b_atoms, b_atoms.clone(), 1, &res.cell_matrix);

        let atom_desc = calculate_descriptors(b_b, b_x);
        let descriptors = atom_desc.iter()
            .map(|(_, desc)| *desc)
            .collect::<Vec<_>>();

        let clusters = k_means(descriptors, 2);
        let res = save_clusters_to_csv(&clusters, "./kmeans.csv").unwrap();
    }
    Ok(())
}
