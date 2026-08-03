use crate::{observables::op::Op, types::{AsVector, DWType, Vector}};
use crate::prelude::{Error, Result};

pub(super) fn dw_shift(op_profile: &[Op], dw_type: DWType) -> usize {
    let tanh_profile_entry = match dw_type {
        DWType::HH => 0,
        DWType::HT | DWType::Apb => 2
    };

    todo!()
}

fn dw_center_ids(op_profile: &[Op]) -> usize {
    todo!()
}

fn fit_tanh() {
    todo!()
}

fn reposition_tanh(profile: &mut [Op]) {
    todo!()
}

fn find_first_plateau(data: &[f64], window_size: usize) -> Result<(usize, usize)> {
    todo!()
}

#[allow(clippy::cast_precision_loss)]
fn window_average(data: &[f64], window_bounds: (usize, usize)) -> f64 {
    let data_len = data.len();
    let (lower, upper) = window_bounds;

    let split1 = &data[lower..data_len];
    let split2 = &data[0..upper];

    let data: Vec<f64> = split1.iter()
        .chain(split2)
        .cloned()
        .collect();

    data.iter()
        .fold(0.0, |acc, val| acc + val)
}

fn wrapped_index(idx: isize, data_len: usize) -> usize {
    assert!(data_len != 0, "Data is empty");
    
    if idx < 0 {
        let rem = idx.unsigned_abs() % data_len;
        if rem == 0 { 0 } else { data_len - rem }
    } else {
        (idx as usize) % data_len
    }
}

fn rolling_avg_crossover(
    data: &[f64], 
    window_size_left: usize,
    window_size_right: usize
) -> Vec<(usize, f64)> {

    let data_len = data.len();
    let mut differences: Vec<(usize, f64)> = Vec::with_capacity(data.len());

    for i in 0..data_len as isize {
        // splitting into 2 arrays because index wrapping could break array slicing
        // -> arr[8..2]
        let lower = wrapped_index(i - window_size_left as isize, data_len);
        let upper = wrapped_index(i - 1, data_len);
        let left_avg = window_average(data, (lower, upper));

        let lower = wrapped_index(i, data_len);
        let upper = wrapped_index(i + window_size_right as isize, data_len);
        let right_avg = window_average(data, (lower, upper));

        /*
        let lower = wrapped_index(i - window_size_left as isize, data_len);
        let upper = wrapped_index(i - 1, data_len);
        let window_avg_left = average(&data[lower..upper]);

        let lower = wrapped_index(i, data_len);
        let upper = wrapped_index(i + window_size_right as isize, data_len);
        let window_avg_right = average(&data[lower..upper]);

        let diff = (i as usize, window_avg_right - window_avg_left);
        differences.push(diff);
        */
        todo!()
    }

    differences
}

#[allow(clippy::cast_precision_loss)]
fn forward_diff(data: &[f64]) -> Result<f64> {
    let first = data.first().ok_or_else(|| 
        Error::Generic("Data is empty!".into()))?;
    let last = data.last().ok_or_else(|| 
        Error::Generic("Data is empty!".into()))?;

    Ok((last - first) / data.len() as f64)
}

#[allow(clippy::cast_precision_loss)]
fn rolling_average(profile: &[Op], max_window_size: usize) -> Vec<Op> {
    assert!(max_window_size <= profile.len(), "Window size is larger than profile.");

    let mut averages: Vec<Op> = Vec::with_capacity(profile.len());
    for i in 0..profile.len() {
        let window_size = if i <= profile.len() - max_window_size {
            max_window_size
        } else {
            profile.len() - i
        };

        let avg = &profile[i..i+window_size]
            .iter()
            .fold(Vector::zeros(), |acc, phi| acc + phi.as_vector());

        let op = Op::new(avg / window_size as f64);
        averages.push(op);
    }

    averages
}
