import tensorflow as tf
import numpy as np
import pandas as pd

def nan_mean(x, axis=None):
    valid_mask = tf.math.logical_not(tf.math.is_nan(x))
    sum = tf.math.reduce_sum(tf.where(valid_mask, x, tf.zeros_like(x)), 
                             axis=axis, keepdims=True)
    weight = tf.math.reduce_sum(tf.cast(valid_mask, tf.float32),
                                axis=axis, keepdims=True)
    return sum / weight


def norm(coord, ref_lm_idx):
    diff = coord - coord[:,:, ref_lm_idx]
    mean_diff = tf.math.sqrt(nan_mean(tf.math.square(diff), axis=-1))
    norm = diff / mean_diff
    return norm


def preproc_norm (inputs, max_len, ref_lm_idx):

    lm_dim = inputs.shape[-1]

    # get xy coordinates
    xy_mask = tf.range(0, lm_dim) % 3 != 2
    xy_coord = tf.boolean_mask(inputs, xy_mask, axis=-1)

    # remove frames where all columns a nan
    nan_mask  = tf.math.is_nan(xy_coord)
    all_nan_mask = tf.math.reduce_all(nan_mask, axis=-1, keepdims=True)
    xy_coord = tf.boolean_mask(xy_coord, tf.math.logical_not(all_nan_mask), 
                               axis=1)
    
    xy_lm_dim = xy_coord.shape[-1]

    x_mask = tf.range(0, xy_lm_dim) % 2 == 0
    x_coord = tf.boolean_mask(inputs, x_mask, axis=-1)
    x_norm = norm(x_coord, ref_lm_idx)

    y_mask = tf.range(0, xy_lm_dim) % 2 == 1
    y_coord = tf.boolean_mask(inputs, y_mask, axis=-1)
    y_norm = norm(y_coord, ref_lm_idx)

    norm_coord = tf.concat([x_norm, y_norm], axis=-1)
    # convert nan to 0
    norm_coord = tf.where(tf.math.is_nan(norm_coord), 
                          tf.zeros_like(norm_coord), norm_coord)
    
    if tf.shape(norm_coord)[1] < max_len:
        pad_len = max_len - tf.shape(norm_coord)[1]
        norm_coord = tf.pad(norm_coord, [[0,0], [0, pad_len], [0,0]])
    else:
        norm_coord = norm_coord[:, :max_len, :]

    return norm_coord





class Preprocess(tf.keras.layers.Layer):
    def __init__(self, max_len, ref_lm_idx, preproc_method = 'norm',  **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.ref_lm_idx = ref_lm_idx

    def call(self, inputs):
        if tf.rank(inputs) == 2:
            inputs = inputs[None, ...]
        # inputs: [batch_size, seq_len, landmark_dim]
        # Assuming the landmarks have been selected already

        if preproc_norm == 'norm':
            x = preproc_norm(inputs, self.max_len, self.ref_lm_idx)
        

        




        
        