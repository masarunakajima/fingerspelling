import os
from os import path
import json
import pprint
import tensorflow as tf
# import matplotlib.pyplot as plt
import pandas as pd
import re
import glob
import tqdm



NUM_LMKS = 543



# Function to create a tf.train.Feature for an integer value
def _int_feature(value_list):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))

# Function to create a tf.train.Feature for a float value
def _float_feature(value_list):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))

# Function to create a tf.train.Feature for a byte string value
def _bytes_feature(value_list):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value_list))






def get_selected_column_idx(all_features, selected_lmks, axes):
    selected_features = []
    for lmk in selected_lmks:
        for axis in axes:
            selected_features.append(f'{axis}_{lmk}')
    selected_column_idx = [i for i in range(len(all_features)) if 
                        all_features[i] in selected_features]
    return selected_column_idx

def get_refs(inputs, ref_lmk_id):
    # assume inputs is numpy array of shape [1, seq_len, NUM_LMKS*3] or [seq_len, NUM_LMKS*3]
    if tf.rank(inputs) == 2:
        inputs = inputs[None, ...]
    ref_x = inputs[:, :, [ref_lmk_id]]
    ref_x = tf.where(tf.math.is_nan(ref_x), tf.zeros_like(ref_x), ref_x)
    ref_y = inputs[:, :, [ref_lmk_id+NUM_LMKS]]
    ref_y = tf.where(tf.math.is_nan(ref_y), tf.zeros_like(ref_y), ref_y)
    ref_z = inputs[:, :, [ref_lmk_id+NUM_LMKS*2]]
    ref_z = tf.where(tf.math.is_nan(ref_z), tf.zeros_like(ref_z), ref_z)
    refs = [ref_x, ref_y, ref_z]
    # nan in refs are filled with 0.0
    return refs

def nan_mean(x, axis=None):
    valid_mask = tf.math.logical_not(tf.math.is_nan(x))
    sum = tf.math.reduce_sum(tf.where(valid_mask, x, tf.zeros_like(x)), 
                             axis=axis, keepdims=True)
    weight = tf.math.reduce_sum(tf.cast(valid_mask, tf.float32),
                                axis=axis, keepdims=True)
    return sum / weight


def norm(coord, ref_coord):
    diff = coord - ref_coord
    mean_diff = tf.math.sqrt(nan_mean(tf.math.square(diff), axis=-1))
    norm = diff / mean_diff
    return norm

def normalize(coord, refs, axes):
    axes2num = {'x': 0, 'y': 1, 'z': 2}
    dim = coord.shape[-1]
    num_axes = len(axes)
    step = dim // num_axes

    axis_num = [axes2num[ax] for ax in axes]
    ref_coord = [refs[i] for i in axis_num]
    
    normed_coords = []
    for k in range(num_axes):
        # get k-th axis 
        normed_coord = coord[:,:,k*step:(k+1)*step] 
        normed_coord = norm(normed_coord, ref_coord[k])
        normed_coords.append(normed_coord)
    normed_coord = tf.concat(normed_coords, axis=-1)

    return normed_coord

def remove_rows_with_all_nan(coord):
    # mask for time steps with at least one non-nan value (dim = 1)
    mask = tf.math.logical_not(tf.reduce_all(tf.math.is_nan(coord), axis=-1))[0]
    coord = tf.boolean_mask(coord, mask, axis=1)
    return coord
        
def pad(coord, max_len):
    seq_len = tf.shape(coord)[1]
    if seq_len > max_len:
        coord = coord[:, :max_len, :]
    else:
        pad_len = max_len - seq_len
        coord = tf.pad(coord, [[0,0], [0, pad_len], [0,0]])
    return coord

def fill_na(coord):
    
    coord = tf.where(tf.math.is_nan(coord), 0.0, coord)
    return coord 


def preproc_norm (inputs, **kwargs):
    # assume inputs is numpy array of shape [1, seq_len, NUM_LMKS*3] or [seq_len, NUM_LMKS*3]
    if tf.rank(inputs) == 2:
        inputs = inputs[None, ...]

    ref_lmrk_id = kwargs.get('ref_lmrk_id', 0)
    axes = kwargs.get('axes', ['x', 'y'])
    selected_lmks = kwargs.get('selected_lmks', [])
    all_features = kwargs.get('all_features', [])
    max_len = kwargs.get('max_len', 300)

    refs = get_refs(inputs, ref_lmrk_id)
    selected_column_idx = get_selected_column_idx(all_features, selected_lmks, axes)
    coord = tf.gather(inputs, selected_column_idx, axis=-1)
    coord = remove_rows_with_all_nan(coord)
    normed_coord = normalize(coord, refs, axes)
    padded_coord = pad(normed_coord, max_len)
    output = tf.where(tf.math.is_nan(padded_coord), 0.0, padded_coord)

    return output


def preproc_hub(inputs, **kwargs):
    preproc_method = kwargs.get('preproc_method', 'norm')
    if preproc_method == 'norm':
        output = preproc_norm(inputs, **kwargs)
    return output


def create_tf_example(df, df_label, **kwargs):
    sequence_id = df.index[0]
    label = df_label[df_label['sequence_id'] == sequence_id]['phrase'].values[0]
    pattern = r"(x_|y_|z_)"
    coords = [col for col in df.columns if re.search(pattern, col) ]
    data = df[coords].values
    
    preproc_data = preproc_hub(data, **kwargs)
    
    seq_len = preproc_data.shape[1]

    lm_dim = len(preproc_data)
    
    flat_data  = data.flatten()

    feature_dict = {'sequence_id': _int_feature([sequence_id]),
                    'seq_len': _int_feature([seq_len]),
                    'lm_dim': _int_feature([lm_dim]),
                    'data': _float_feature(flat_data),
                    'label': _bytes_feature([label.encode()])}

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def write_to_tfrecord(prquet_file, df_label, writer, **kwargs):
    df = pd.read_parquet(prquet_file)
    sequence_ids = df.index.unique()
    for sequence_id in sequence_ids:
        sub_df = df.loc[[sequence_id]]
        example = create_tf_example(sub_df, df_label, **kwargs)
        writer.write(example.SerializeToString())


if __name__ == '__main__':


    data_dir = 'data'
    
    tfrecords_root = 'tfrecords'
    # creata a new tfrecords dir with new number
    n_record_dirs = len(glob.glob(path.join(tfrecords_root, '*')))
    tfrecords_dir = path.join(tfrecords_root, f'record_{n_record_dirs}')
    os.makedirs(tfrecords_dir, exist_ok=True)

    tfrecords_path = path.join(tfrecords_dir, 'train.tfrecords')
    preproc_args_path = path.join(tfrecords_dir, 'preproc_args.json')

    parquet_dir = path.join(data_dir, 'train_landmarks')
    parquet_paths = glob.glob(path.join(parquet_dir, '*.parquet'))
    label_path = path.join(data_dir, 'train.csv')

    df_label = pd.read_csv(label_path)

    n_hand_lmrks = 21
    n_pose_lmrks = 33
    handedness = ['left', 'right']
    selected_lmks = []
    for hand in handedness:
        for i in range(n_hand_lmrks):
            selected_lmks.append(f'{hand}_hand_{i}')
    for i in range(n_pose_lmrks):
        selected_lmks.append(f'pose_{i}')

    features_path = 'all_features.json'
    with open(features_path, 'r') as f:
        all_features = json.load(f)['all_features']
    axes = ['x', 'y']
    # selected_column_idx = get_selected_column_idx(all_features, selected_lmks, coords)

    max_len = 300
    ref_lmrk_id = 489  # x_right_hand_0
    preproc_method = 'norm'

    preproc_args = {'max_len': max_len,
                    'ref_lmrk_id': ref_lmrk_id,
                    'preproc_method': preproc_method,
                    'selected_lmks': selected_lmks,
                    'axes': axes, 
                    'all_features': all_features}


    
    with tf.io.TFRecordWriter(tfrecords_path) as writer:
        for parquet_path in tqdm.tqdm(parquet_paths):
            write_to_tfrecord(parquet_path, df_label, writer, **preproc_args)




