# tests/test_tfa_interpolate_spline.py

import pytest
import tensorflow as tf
from wf_psf.utils.interpolation import tfa_interpolate_spline_rbf as interpolate_spline


@pytest.fixture
def simple_1d_data():
    train_points = tf.constant([[0.0], [1.0], [2.0]], dtype=tf.float32)
    train_values = tf.constant([[0.0], [1.0], [4.0]], dtype=tf.float32)
    query_points = tf.constant([[0.5], [1.5]], dtype=tf.float32)
    return train_points, train_values, query_points


def test_output_shape(simple_1d_data):
    train_points, train_values, query_points = simple_1d_data
    with tf.device("/CPU:0"):
        result = interpolate_spline(
            train_points=tf.expand_dims(train_points, axis=0),
            train_values=tf.expand_dims(train_values, axis=0),
            query_points=tf.expand_dims(query_points, axis=0),
            order=2,
            regularization_weight=0.0,
        )
    # Expect shape: [1, n_query, n_values]
    assert result.shape == (1, 2, 1)


def test_differentiability(simple_1d_data):
    train_points, train_values, query_points = simple_1d_data
    query = tf.Variable(query_points, dtype=tf.float32)
    with tf.device("/CPU:0"):
        with tf.GradientTape() as tape:
            result = interpolate_spline(
                train_points=tf.expand_dims(train_points, axis=0),
                train_values=tf.expand_dims(train_values, axis=0),
                query_points=tf.expand_dims(query, axis=0),
                order=2,
                regularization_weight=0.0,
            )
            loss = tf.reduce_sum(result)
        grad = tape.gradient(loss, query)

    assert grad is not None
    assert grad.shape == query.shape


@pytest.mark.parametrize("order", [1, 2, 3])
def test_order_variants(simple_1d_data, order):
    train_points, train_values, query_points = simple_1d_data
    with tf.device("/CPU:0"):
        result = interpolate_spline(
            train_points=tf.expand_dims(train_points, axis=0),
            train_values=tf.expand_dims(train_values, axis=0),
            query_points=tf.expand_dims(query_points, axis=0),
            order=order,
            regularization_weight=0.0,
        )
    assert result.shape == (1, 2, 1)
