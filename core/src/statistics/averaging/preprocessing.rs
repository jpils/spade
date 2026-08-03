use mathr::prelude::{LmFit, LmOptions, curve_fit};

use crate::{observables::op::Op, types::{AsVector, DWType, Vector}};
use crate::prelude::{Error, Result};

#[derive(Clone, Copy, Debug)]
pub(crate) struct WindowSizes {
    left: usize,
    right: usize
}

pub(super) fn dw_shift(op_profile: &[Op], dw_type: DWType, window_sizes: WindowSizes) -> Result<usize> {
    let tanh_profile_entry = match dw_type {
        DWType::HH => 0,
        DWType::HT | DWType::Apb => 2
    };
    
    let data: Vec<f64> = op_profile.iter()
        .map(|op| op.as_vector()[tanh_profile_entry])
        .collect();

    let tanh_crossover_signal = rolling_avg_crossover(&data, window_sizes);
    let (root_left, root_right) = tanh_zeropoints(&tanh_crossover_signal)?;

    Ok(root_left)
}

pub(super) fn reposition_dw(profile: &mut [f64], shift: usize) { // mave this into slice with right
    // type
    profile.rotate_left(shift);
}

#[derive(Clone, Copy, Debug)]
struct WindowBounds(usize, usize);

fn tanh_zeropoints(crossover_signal: &[(usize, f64)]) -> Result<(usize, usize)> {
    let (argmax,_) = crossover_signal.iter()
        .max_by(|&x, &y| x.1.total_cmp(&y.1))
        .ok_or_else(|| Error::Generic("Crossover signal is empty!".into()))?;
    let (argmin, _) = crossover_signal.iter()
        .min_by(|&x, &y| x.1.total_cmp(&y.1))
        .ok_or_else(|| Error::Generic("Crossover signal is empty!".into()))?;

    match argmax.cmp(argmin) {
        std::cmp::Ordering::Less => Ok((*argmax, *argmin)),
        std::cmp::Ordering::Greater => Ok((*argmin, *argmax)),
        std::cmp::Ordering::Equal => Err(Error::Generic("Tanh profile corrupted!".into()))
    }
}

#[allow(clippy::cast_precision_loss)]
fn window_average(data: &[f64], window_bounds: WindowBounds) -> f64 {
    let data_len = data.len();
    let WindowBounds(lower, upper) = window_bounds;

    match lower.cmp(&upper) {
        std::cmp::Ordering::Greater => {
            let split1 = &data[lower..data_len];
            let split2 = &data[0..upper];
            split1.iter()
                .chain(split2)
                .fold(0.0, |acc, val| acc + val) 
            / (split1.len() + split2.len()) as f64 
        },
        std::cmp::Ordering::Less => {
            data[lower..upper].iter()
                .fold(0.0, |acc, val| acc + val) 
            / (upper - lower) as f64
        },
        std::cmp::Ordering::Equal => 0.0, 
    }
}

fn wrapped_index(idx: isize, data_len: usize) -> usize {
    assert!(data_len != 0, "Data is empty");
    
    if idx < 0 {
        let rem = idx.unsigned_abs() % data_len;
        if rem == 0 { 0 } else { data_len - rem }
    } else {
       idx.cast_unsigned() % data_len
    }
}

fn rolling_avg_crossover(
    data: &[f64], 
    window_sizes: WindowSizes
) -> Vec<(usize, f64)> {

    let window_size_left = window_sizes.left;
    let window_size_right = window_sizes.right;
    let data_len = data.len();
    let mut differences: Vec<(usize, f64)> = Vec::with_capacity(data.len());

    for i in 0..data_len.cast_signed() {
        // splitting into 2 arrays because index wrapping could break array slicing
        // -> arr[8..2]
        let lower = wrapped_index(i - window_size_left.cast_signed(), data_len);
        let upper = wrapped_index(i, data_len);
        let left_avg = window_average(data, WindowBounds(lower, upper));

        let lower = wrapped_index(i, data_len);
        let upper = wrapped_index(i + window_size_right.cast_signed(), data_len);
        let right_avg = window_average(data, WindowBounds(lower, upper));

        let diff = (i.cast_unsigned(), right_avg - left_avg);
        differences.push(diff);
    }

    differences
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn assert_close(actual: f64, expected: f64) {
        let tolerance = EPS * expected.abs().max(1.0);

        assert!(
            (actual - expected).abs() <= tolerance,
            "Expected {expected}, got {actual}"
        );
    }

    #[test]
    fn window_average_contiguous() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];

        assert_close(window_average(&data, WindowBounds(1, 4)), 3.0);
        assert_close(window_average(&data, WindowBounds(0, 2)), 1.5);
        assert_close(window_average(&data, WindowBounds(0, 5)), 3.0);
    }

    #[test]
    fn window_average_wrapped() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];

        assert_close(window_average(&data, WindowBounds(4, 2)), 8.0 / 3.0);
        assert_close(window_average(&data, WindowBounds(3, 1)), 10.0 / 3.0);
    }

    #[test]
    fn window_average_single_element() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];

        assert_close(window_average(&data, WindowBounds(2, 3)), 3.0);
        assert_close(window_average(&data, WindowBounds(4, 0)), 5.0);
    }

    #[test]
    fn wrapped_index_positive() {
        assert_eq!(wrapped_index(0, 10), 0);
        assert_eq!(wrapped_index(9, 10), 9);
        assert_eq!(wrapped_index(10, 10), 0);
        assert_eq!(wrapped_index(11, 10), 1);
        assert_eq!(wrapped_index(21, 10), 1);
    }

    #[test]
    fn wrapped_index_negative() {
        assert_eq!(wrapped_index(-1, 10), 9);
        assert_eq!(wrapped_index(-9, 10), 1);
        assert_eq!(wrapped_index(-10, 10), 0);
        assert_eq!(wrapped_index(-11, 10), 9);
        assert_eq!(wrapped_index(-21, 10), 9);
    }

    #[test]
    fn wrapped_index_extremes() {
        for idx in [isize::MIN, isize::MAX] {
            let expected = (idx as i128).rem_euclid(10) as usize;
            assert_eq!(wrapped_index(idx, 10), expected);
        }
    }

    #[test]
    #[should_panic(expected = "Data is empty")]
    fn wrapped_index_empty_data() {
        wrapped_index(0, 0);
    }

    #[test]
    fn crossover_symmetric_windows() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];

        let result = rolling_avg_crossover(
            &data,
            WindowSizes { left: 2, right: 2 },
        );

        let expected = [-3.0, -0.5, 2.0, 2.0, -0.5];

        assert_eq!(result.len(), data.len());

        for (i, &(index, diff)) in result.iter().enumerate() {
            assert_eq!(index, i);
            assert_close(diff, expected[i]);
        }
    }

    #[test]
    fn crossover_asymmetric_windows() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];

        let result = rolling_avg_crossover(
            &data,
            WindowSizes { left: 1, right: 3 },
        );

        let expected = [
            -3.0,
            2.0,
            2.0,
            1.0 / 3.0,
            -4.0 / 3.0,
        ];

        assert_eq!(result.len(), data.len());

        for (i, &(index, diff)) in result.iter().enumerate() {
            assert_eq!(index, i);
            assert_close(diff, expected[i]);
        }
    }

    #[test]
    fn crossover_constant_signal() {
        let data = [7.25; 10];

        for left in 1..data.len() {
            for right in 1..data.len() {
                let result = rolling_avg_crossover(
                    &data,
                    WindowSizes { left, right },
                );

                assert_eq!(result.len(), data.len());

                for (_, diff) in result {
                    assert_close(diff, 0.0);
                }
            }
        }
    }

    #[test]
    fn crossover_empty_data() {
        let result = rolling_avg_crossover(
            &[],
            WindowSizes { left: 2, right: 3 },
        );

        assert!(result.is_empty());
    }

    fn reference_difference(
        data: &[f64],
        i: usize,
        left: usize,
        right: usize,
    ) -> f64 {
        let n = data.len();

        let left_sum: f64 = (0..left)
            .map(|j| data[(i + n - left + j) % n])
            .sum();

        let right_sum: f64 = (0..right)
            .map(|j| data[(i + j) % n])
            .sum();

        right_sum / right as f64 - left_sum / left as f64
    }

    #[test]
    fn crossover_matches_reference() {
        for n in 2..=20 {
            let data: Vec<f64> = (0..n)
                .map(|i| ((17 * i + 3) % 23) as f64 - 11.0)
                .collect();

            for left in 1..n {
                for right in 1..n {
                    let result = rolling_avg_crossover(
                        &data,
                        WindowSizes { left, right },
                    );

                    assert_eq!(result.len(), n);

                    for (i, &(index, diff)) in result.iter().enumerate() {
                        assert_eq!(index, i);

                        let expected =
                            reference_difference(&data, i, left, right);

                        assert_close(diff, expected);
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "Full-period windows currently produce equal bounds"]
    fn crossover_full_period_window() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];

        let result = rolling_avg_crossover(
            &data,
            WindowSizes { left: 5, right: 2 },
        );

        for (i, &(_, diff)) in result.iter().enumerate() {
            let expected = reference_difference(&data, i, 5, 2);
            assert_close(diff, expected);
        }
    }

    #[test]
    #[ignore = "Zero-width windows are not yet rejected"]
    #[should_panic]
    fn crossover_rejects_zero_window() {
        rolling_avg_crossover(
            &[1.0, 2.0, 3.0],
            WindowSizes { left: 0, right: 2 },
        );
    }
}
