use mathr::prelude::{LmFit, LmOptions, curve_fit};

use crate::{cell::simulation::FittedSimCell, observables::{Observable, op::Op}, types::{AsVector, DWType, Vector}};
use crate::prelude::{Error, Result};
use crate::observables::BinnedObservable;

#[derive(Clone, Copy, Debug)]
pub struct WindowSizes {
    left: usize,
    right: usize
}

pub(super) fn dw_shift(op_profile: &BinnedObservable<Op>, dw_type: DWType, window_sizes: WindowSizes) -> Result<usize> {
    let tanh_profile_entry = match dw_type {
        DWType::HH => 0,
        DWType::HT | DWType::Apb => 2
    };
    
    let data: Vec<f64> = op_profile.observable.iter()
        .map(|op| op.as_vector()[tanh_profile_entry])
        .collect();

    let tanh_crossover_signal = rolling_avg_crossover(&data, window_sizes);
    let (root_left, root_right) = tanh_zeropoints(&tanh_crossover_signal)?;

    Ok(root_left)
}

pub(super) fn shift_obs<Obs>(profile: &mut BinnedObservable<Obs>, shift: usize) 
    where Obs: Observable
{
    profile.observable.rotate_left(shift);
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

        for (i, &(_, diff)) in result.iter().enumerate() {
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

    #[test]
    fn tanh_zeropoints_finds_ordered_extrema() {
        let signal = [
            (0, 0.2),
            (1, -3.0),
            (2, 0.5),
            (3, 4.0),
            (4, -1.0),
        ];

        assert_eq!(tanh_zeropoints(&signal).unwrap(), (1, 3));
    }

    #[test]
    fn tanh_zeropoints_handles_reversed_extrema() {
        let signal = [
            (0, 4.0),
            (1, 1.0),
            (2, -3.0),
            (3, 0.0),
        ];

        assert_eq!(tanh_zeropoints(&signal).unwrap(), (0, 2));
    }

    #[test]
    fn tanh_zeropoints_rejects_empty_signal() {
        assert!(tanh_zeropoints(&[]).is_err());
    }

    #[test]
    fn tanh_zeropoints_rejects_single_element() {
        assert!(tanh_zeropoints(&[(0, 1.0)]).is_err());
    }

    #[test]
    #[ignore = "Flat crossover signals are not yet rejected"]
    fn tanh_zeropoints_rejects_constant_signal() {
        let signal: Vec<(usize, f64)> =
            (0..64).map(|i| (i, 0.0)).collect();

        assert!(tanh_zeropoints(&signal).is_err());
    }

    #[test]
    fn dw_shift_finds_expected_wall_position() {
        let profile = two_wall_profile(64);

        let shift = dw_shift(
            &profile,
            DWType::HH,
            WindowSizes { left: 5, right: 5 },
        )
        .unwrap();

        assert!(shift.abs_diff(12) <= 2, "shift={shift}");
    }

    #[test]
    fn dw_shift_selects_correct_component() {
        let profile = two_wall_profile(64);
        let windows = WindowSizes { left: 5, right: 5 };

        let hh = dw_shift(
            &profile,
            DWType::HH,
            windows,
        )
        .unwrap();

        let ht = dw_shift(
            &profile,
            DWType::HT,
            windows,
        )
        .unwrap();

        let apb = dw_shift(
            &profile,
            DWType::Apb,
            windows,
        )
        .unwrap();

        assert!(hh.abs_diff(12) <= 2, "HH shift={hh}");
        assert!(ht.abs_diff(24) <= 2, "HT shift={ht}");
        assert!(apb.abs_diff(24) <= 2, "APB shift={apb}");
    }

    #[test]
    fn dw_shift_rejects_empty_profile() {
        let profile = BinnedObservable::<Op> {
            bins: vec![],
            observable: vec![],
        };

        assert!(
            dw_shift(
                &profile,
                DWType::HH,
                WindowSizes { left: 2, right: 2 },
            )
            .is_err()
        );
    }

    #[test]
    fn shift_obs_rotates_only_observables() {
        let mut profile = BinnedObservable {
            bins: vec![0.1, 1.1, 2.1, 3.1, 4.1],
            observable: (1..=5)
                .map(|i| Op::new(Vector::new(i as f64, 0.0, 0.0)))
                .collect(),
        };

        shift_obs(&mut profile, 2);

        assert_eq!(profile.bins, vec![0.1, 1.1, 2.1, 3.1, 4.1]);

        let expected = [3.0, 4.0, 5.0, 1.0, 2.0];

        for (op, expected) in profile.observable.iter().zip(expected) {
            assert_close(op.as_vector()[0], expected);
        }
    }

    #[test]
    fn shift_obs_zero_shift_preserves_profile() {
        let mut profile = two_wall_profile(64);

        let original: Vec<Vector> = profile
            .observable
            .iter()
            .map(|op| *op.as_vector())
            .collect();

        shift_obs(&mut profile, 0);

        for (op, expected) in profile.observable.iter().zip(&original) {
            assert!(
                (op.as_vector() - expected).norm() <= EPS
            );
        }
    }

    #[test]
    fn dw_shift_tracks_translation_without_boundary_crossing() {
        let n = 64;
        let translation = 5;
        let windows = WindowSizes { left: 5, right: 5 };

        let original = two_wall_profile(n);
        let translated = translated_profile(n, translation);

        let original_shift =
            dw_shift(&original, DWType::HH, windows).unwrap();

        let translated_shift =
            dw_shift(&translated, DWType::HH, windows).unwrap();

        assert_eq!(
            translated_shift,
            (original_shift + translation) % n
        );
    }

    #[test]
    fn preprocessing_aligns_translated_profiles() {
        let n = 64;
        let translation = 5;
        let windows = WindowSizes { left: 5, right: 5 };

        let mut original = two_wall_profile(n);
        let mut translated = translated_profile(n, translation);

        let original_shift =
            dw_shift(&original, DWType::HH, windows).unwrap();

        let translated_shift =
            dw_shift(&translated, DWType::HH, windows).unwrap();

        shift_obs(&mut original, original_shift);
        shift_obs(&mut translated, translated_shift);

        assert_eq!(original.bins, translated.bins);

        for (lhs, rhs) in original
            .observable
            .iter()
            .zip(&translated.observable)
        {
            assert!(
                (lhs.as_vector() - rhs.as_vector()).norm() <= EPS
            );
        }
    }

    #[test]
    fn block_average_preserves_aligned_profile() {
        let n = 64;
        let windows = WindowSizes { left: 5, right: 5 };

        let mut original = two_wall_profile(n);
        let mut translated = translated_profile(n, 5);

        let shift_original =
            dw_shift(&original, DWType::HH, windows).unwrap();

        let shift_translated =
            dw_shift(&translated, DWType::HH, windows).unwrap();

        shift_obs(&mut original, shift_original);
        shift_obs(&mut translated, shift_translated);

        let expected_bins = original.bins.clone();

        let expected_obs: Vec<Vector> = original
            .observable
            .iter()
            .map(|op| *op.as_vector())
            .collect();

        let averaged = super::super::block_average(
            &[original, translated],
            2,
        )
        .unwrap();

        assert_eq!(averaged.bins.len(), n);
        assert_eq!(averaged.observable.len(), n);

        for (actual, expected) in averaged.bins.iter().zip(&expected_bins) {
            assert_close(*actual, *expected);
        }

        for (actual, expected) in averaged
            .observable
            .iter()
            .zip(&expected_obs)
        {
            assert!(
                (actual.as_vector() - expected).norm() <= EPS
            );
        }
    }

    #[test]
    #[ignore = "Sorting extrema by index loses wall identity across PBC"]
    fn dw_shift_preserves_wall_identity_across_boundary() {
        let n = 64;
        let translation = 24;
        let windows = WindowSizes { left: 5, right: 5 };

        let original = two_wall_profile(n);
        let translated = translated_profile(n, translation);

        let original_shift =
            dw_shift(&original, DWType::HH, windows).unwrap();

        let translated_shift =
            dw_shift(&translated, DWType::HH, windows).unwrap();

        assert_eq!(
            translated_shift,
            (original_shift + translation) % n
        );
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

    fn two_wall_profile(n: usize) -> BinnedObservable<Op> {
        let bins: Vec<f64> = (0..n)
            .map(|i| i as f64)
            .collect();

        let observable = (0..n)
            .map(|i| {
                let x = periodic_tanh_profile(i, 12.0, 44.0);
                let z = periodic_tanh_profile(i, 24.0, 54.0);

                Op::new(Vector::new(x, 0.0, z))
            })
            .collect();

        BinnedObservable { bins, observable }
    }

    fn translated_profile(
        n: usize,
        translation: usize,
    ) -> BinnedObservable<Op> {
        let mut profile = two_wall_profile(n);
        profile.observable.rotate_right(translation);
        profile
    }

    fn periodic_tanh_profile(
        i: usize,
        rising: f64,
        falling: f64,
    ) -> f64 {
        let x = i as f64;

        ((x - rising) / 2.0).tanh()
            - ((x - falling) / 2.0).tanh()
            - 1.0
    }
}
