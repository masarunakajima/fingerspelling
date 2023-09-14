# study: Make a study directory with all the configuration needed to
# run analysis for methbase.
#
# MIT License
#
# Copyright (c) 2023 Masaru Nakajima
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Defines a transformer model for 2 dimensional input (time x features) 
and sequence of classes as ouput for ASL finger spelling detection.
"""

# from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding


def pos_encode(seq_len, dim):
    """
    Generate sinusoidal positional embedding.
    """
    range_even = tf.range(dim, dtype=tf.float32)
    range_even = tf.divide(range_even, 2)
    range_even = tf.cast(range_even, dtype=tf.int32)
    range_even = tf.cast(range_even, dtype=tf.float32)
    power = tf.math.divide(range_even, dim)
    denom = tf.math.pow(10000, power)
    denom = tf.reshape(denom, (1, dim))
    pos = tf.range(seq_len, dtype=tf.float32)
    pos = tf.reshape(pos, (seq_len, 1))
    arg = tf.divide(pos, denom)
    cos_mask = tf.cast(tf.math.mod(tf.range(seq_len), 2), dtype=tf.bool)
    cos_mask = tf.reshape(cos_mask, (seq_len, 1))
    sin_mask = tf.logical_not(cos_mask)
    sin = tf.where(sin_mask, tf.math.sin(arg), 0)
    cos = tf.where(cos_mask, tf.math.cos(arg), 0)

    return tf.math.add(sin, cos)


# class Embedding1D(Layer):
    # """
    # Embedding layer for 1D sequence of tokens
    # """

    # def __init__(self, seq_len, num_class, num_hid=64):
        # super().__init__()


# def get_study_title(study_data_frame):
# """
# Get the study title from the first row of a data frame.
# """
# try:
# return study_data_frame.iloc[0].study_title
# except AttributeError as attrib_err:
# attrib_err.add_note(  # pylint: disable=E1101
# "'study_title' not found in dataframe"
# )
# raise attrib_err
