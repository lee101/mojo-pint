import numpy as np

from mojopint._lib import (
    _PARALLEL_THRESHOLD,
    _convert_f64,
    add_converted_arrays,
    combine_dimension_arrays,
    convert_array,
    convert_array_inplace,
    dimensions_equal_many,
)


def test_conversion_simd_tail_matches_numpy():
    for size in (0, 1, 3, 4, 5, 17):
        values = np.linspace(-3.0, 7.0, size)
        assert np.allclose(
            convert_array(values, 2.5, -1.25),
            values * 2.5 - 1.25,
            rtol=1e-15,
            atol=1e-15,
        )


def test_conversion_handles_noncontiguous_and_readonly_inputs_safely():
    values = np.arange(30.0).reshape(5, 6)[:, ::2]
    values.flags.writeable = False
    result = convert_array(values, 2.0, 1.0)
    assert result.flags.c_contiguous
    assert np.array_equal(result, values * 2.0 + 1.0)
    assert convert_array_inplace(values, 2.0, 1.0) is not values


def test_conversion_does_not_discard_complex_or_extended_precision():
    complex_values = np.array([1 + 2j, 3 - 4j])
    assert np.array_equal(
        convert_array(complex_values, 2.0, 1.0),
        complex_values * 2.0 + 1.0,
    )
    extended = np.array([1.0], dtype=np.longdouble)
    result = convert_array(extended, np.longdouble("1.1"), 0.0)
    assert result.dtype == np.longdouble


def test_private_conversion_boundary_rejects_invalid_buffers():
    good = np.arange(4.0)
    with np.testing.assert_raises(TypeError):
        _convert_f64(good.astype(np.float32), good, 1.0, 0.0)
    with np.testing.assert_raises(ValueError):
        _convert_f64(good, good[:3], 1.0, 0.0)
    with np.testing.assert_raises(ValueError):
        _convert_f64(good[::2], np.empty(2), 1.0, 0.0)


def test_array_addition_empty_and_complex_paths():
    empty = np.array([], dtype=np.float64)
    assert add_converted_arrays(empty, empty, 2.0, 1.0).shape == (0,)
    left = np.array([1 + 2j])
    right = np.array([3 - 1j])
    assert np.array_equal(
        add_converted_arrays(left, right, 2.0, 1.0),
        left + (right * 2.0 + 1.0),
    )


def test_array_addition_parallel_threshold_and_worker_tails_match_numpy():
    for size in (
        _PARALLEL_THRESHOLD - 1,
        _PARALLEL_THRESHOLD,
        _PARALLEL_THRESHOLD + 3,
    ):
        left = np.arange(size, dtype=np.float64)
        right = left[::-1].copy()
        expected = left - (right * 0.25 - 2.0)
        assert np.array_equal(
            add_converted_arrays(left, right, 0.25, -2.0, -1.0),
            expected,
        )


def test_conversion_parallel_threshold_and_worker_tails_match_numpy():
    for size in (
        _PARALLEL_THRESHOLD - 1,
        _PARALLEL_THRESHOLD,
        _PARALLEL_THRESHOLD + 3,
    ):
        values = np.arange(size, dtype=np.float64)
        assert np.array_equal(convert_array(values, 0.5, 0.0), values * 0.5)


def test_dimension_combine_kernel_matches_numpy():
    rng = np.random.default_rng(4)
    left = rng.integers(-3, 4, size=(10_003, 7)).astype(float)
    right = rng.integers(-3, 4, size=(10_003, 7)).astype(float)
    assert np.array_equal(combine_dimension_arrays(left, right), left + right)
    assert np.array_equal(
        combine_dimension_arrays(left, right, -1), left - right
    )


def test_dimension_equality_kernel_matches_numpy():
    rng = np.random.default_rng(5)
    left = rng.integers(-3, 4, size=(20_001, 7)).astype(float)
    right = left.copy()
    right[::17, 3] += 1
    expected = np.all(left == right, axis=1)
    result = dimensions_equal_many(left, right)
    assert result.dtype == np.bool_
    assert np.array_equal(result, expected)


def test_empty_dimension_batches_do_not_cross_null_pointers():
    empty = np.empty((0, 7))
    assert combine_dimension_arrays(empty, empty).shape == (0, 7)
    result = dimensions_equal_many(empty, empty)
    assert result.shape == (0,)
    assert result.dtype == np.bool_
