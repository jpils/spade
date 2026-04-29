use spade_core::{error::Error, geometry, parser::Parser, types::{Atom, Atoms, Matrix, ReadResultVariant, SupercellAtoms, Vectors}};
use nalgebra::{SVector, Vector};

pub(crate) type Descriptor = SVector<f64, 18>;
pub(crate) type Descriptors = Vec<SVector<f64, 18>>;
pub(crate) type NN = Vec<(Atom, Vec<(Atom, f64)>)>;

#[derive(Debug, Clone)]
pub struct NearestNeighbor {
    reference: Atom,
    neighbors: Vec<(Atom, Atom, f64)>
}


/// find n nearest neighbors for atom a with relative positions
pub(crate) fn n_nearest_neighbors(a: &Atoms, b: Atoms, n: usize, cell_matrix: &Matrix) -> Vec<NearestNeighbor> {
    let mut distances: Vec<NearestNeighbor> = Vec::new();
    for atom_a in a {
        let atom_distances = b.iter()
            .map(|&current| { 
                let d_unwrapped_dir = atom_a.position - current.position;
                let min_img = geometry::wrap_vector(&d_unwrapped_dir);
                let d_wrapped_cart = geometry::direct_to_cartesian(min_img, cell_matrix);
                let dist = d_wrapped_cart.norm_squared();
                let current_abs_cart = Atom {
                    atom_type: current.atom_type,
                    position: geometry::direct_to_cartesian(current.position, cell_matrix)
                };
                let current_rel_cart = Atom {
                    atom_type: current.atom_type,
                    position: d_wrapped_cart
                };
                (current_abs_cart, current_rel_cart, dist)
            })
            .collect::<Vec<_>>();

        let mut atom_distances = atom_distances;
        atom_distances.sort_by(|(_, _, a), (_, _, b)| a.total_cmp(&b));
        
        let n_nearest = &atom_distances[..n];
        let n_nearest = if *a == b { // quick fix to discard nn calculation with itself TODO
                                     // optimize later
            &atom_distances[1..n+1]
        } else {
            &atom_distances[..n]
        };
         
        let atom_a_cart_pos = geometry::direct_to_cartesian(atom_a.position, cell_matrix);
        let atom_a = Atom {
            atom_type: atom_a.atom_type,
            position: atom_a_cart_pos
        };
        let nn = NearestNeighbor {
            reference: atom_a,
            neighbors: n_nearest.into()
        };
        distances.push(nn);
    }
    distances
}

pub(crate) fn calculate_descriptors(mut b_b: Vec<NearestNeighbor>, b_x: Vec<NearestNeighbor>) -> Vec<(Atom, Descriptor)> {
    let mut descriptors: Vec<(Atom, Descriptor)> = Vec::with_capacity(b_b.len());
    while let Some(b) = b_b.pop() {
        let current = b.reference;
        let neighbor = b.neighbors[0].0;

        let current_x = b_x.iter().find(|nn| nn.reference == current).unwrap();
        let neighbor_x = b_x.iter().find(|nn| nn.reference == neighbor).unwrap();

        let current_desc = get_domain_descriptor(current_x.to_owned());
        let neighbor_desc = get_domain_descriptor(neighbor_x.to_owned());
        let avg_desc = (current_desc + neighbor_desc)/2.0;

        descriptors.push((current, avg_desc));
    }
    descriptors
}

fn get_domain_descriptor(b_nns: NearestNeighbor) -> Descriptor {
    let vectors = b_nns.neighbors.iter()
        .map(|(_, neighbor, _)| neighbor.position)
        .collect::<Vec<_>>();
    let sorted_vectors = sort_x(vectors);
    
    let mut arr = [0.0; 18];
    for (vec_id, vec) in sorted_vectors.into_iter().enumerate() {
        let arr_id = vec_id * 3;
        arr[arr_id] = vec.x;
        arr[arr_id + 1] = vec.y;
        arr[arr_id + 2] = vec.z;
    }

    Descriptor::from(arr)
}

fn sort_x(positions: Vectors) -> Vectors { // do something with the y pair
    let (y_top_id, &y_top) = positions.iter().enumerate().max_by(|a, b| a.1.y.total_cmp(&b.1.y)).unwrap();

    let mut positions = positions.clone();
    positions.swap_remove(y_top_id);
    let (y_bottom_id, &y_bottom) = positions.iter().enumerate().min_by(|a, b| a.1.y.total_cmp(&b.1.y)).unwrap();
    positions.swap_remove(y_bottom_id);

    let mut pos_angle = positions.into_iter()
        .map(|pos| {
           let angle = geometry::compute_x_angle(&pos);
           (pos, angle)
        })
        .collect::<Vec<_>>();

    pos_angle.sort_by(|a, b| a.1.total_cmp(&b.1));
    
    let mut pos = pos_angle.into_iter()
        .map(|(pos, _)| pos)
        .collect::<Vec<_>>();

    pos.insert(0, y_top.to_owned());
    pos.insert(1, y_bottom.to_owned());
    pos
}
