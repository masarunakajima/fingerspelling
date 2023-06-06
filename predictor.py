

from model import Transformer, prep_dataset, get_tokenizer
import tensorflow as tf
import numpy as np
import glob
from os import path
import pandas as pd
import json


selected_columns = ['x_right_hand_0', 'y_right_hand_0', 'z_right_hand_0',
                    'x_right_hand_1', 'y_right_hand_1', 'z_right_hand_1',
                    'x_right_hand_2', 'y_right_hand_2', 'z_right_hand_2',
                    'x_right_hand_3', 'y_right_hand_3', 'z_right_hand_3',
                    'x_right_hand_4', 'y_right_hand_4', 'z_right_hand_4',
                    'x_right_hand_5', 'y_right_hand_5', 'z_right_hand_5',
                    'x_right_hand_6', 'y_right_hand_6', 'z_right_hand_6',
                    'x_right_hand_7', 'y_right_hand_7', 'z_right_hand_7',
                    'x_right_hand_8', 'y_right_hand_8', 'z_right_hand_8',
                    'x_right_hand_9', 'y_right_hand_9', 'z_right_hand_9',
                    'x_right_hand_10', 'y_right_hand_10', 'z_right_hand_10',
                    'x_right_hand_11', 'y_right_hand_11', 'z_right_hand_11',
                    'x_right_hand_12', 'y_right_hand_12', 'z_right_hand_12',
                    'x_right_hand_13', 'y_right_hand_13', 'z_right_hand_13',
                    'x_right_hand_14', 'y_right_hand_14', 'z_right_hand_14',
                    'x_right_hand_15', 'y_right_hand_15', 'z_right_hand_15',
                    'x_right_hand_16', 'y_right_hand_16', 'z_right_hand_16',
                    'x_right_hand_17', 'y_right_hand_17', 'z_right_hand_17',
                    'x_right_hand_18', 'y_right_hand_18', 'z_right_hand_18',
                    'x_right_hand_19', 'y_right_hand_19', 'z_right_hand_19',
                    'x_right_hand_20', 'y_right_hand_20', 'z_right_hand_20',
                    ]



def get_model_params(model_path):
    with open(path.join(model_path, "model_param.json"), 'r') as f:
        params = json.load(f)
    return params

def get_history(model_path):
    with open(path.join(model_path, "history.json"), 'r') as f:
        history = json.load(f)
    return history








class FingerGenModel(tf.keras.Model):
    def __init__(self, transformer, max_output_len, SOS_token, EOS_token):
        super(FingerGenModel, self).__init__()
        self.transformer = transformer
        self.max_output_len = max_output_len
        self.sos = SOS_token
        self.eos = EOS_token


    def call(self, inputs):

        x = tf.cast(x, tf.float32)[None]
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        x = tf.image.resize(x, (tf.shape(x)[0], self.transformer.seq_len))
        # print(x.shape)
        dec_input = tf.expand_dims([SOS_token], 0)
        logits = self.transformer([x, dec_input], training=False)
        # generate tensor full of nan
        
        return logits[0, :, 1:-2]



if __name__ == "__main__":

    # Verify GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus))

    # Enable all available GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)





    model_path = "models/transformer_8"
    model_params = get_model_params(model_path)
    history = get_history(model_path)
    

    data_path = 'data/train_landmarks'
    parquet_files = glob.glob(path.join(data_path, '*.parquet'))


    label_path = 'train.csv'
    c2p_path = 'character_to_prediction_index.json'

    weights_path = path.join(model_path, 'weights.h5')

    input_length = model_params['seq_len']
    input_dim = len(selected_columns)
    output_length = 32
    max_output_len = 33

    tokenizer = get_tokenizer(c2p_path)
    # print(tokenizer.index_word)



    SOS_token = tokenizer.word_index['<SOS>']
    EOS_token = tokenizer.word_index['<EOS>']


    dataset = prep_dataset([parquet_files[0]], label_path, 
                            input_length, output_length, selected_columns, tokenizer)
    

    df = pd.read_parquet(parquet_files[0], columns=selected_columns)
    labels = pd.read_csv(label_path)
    sequence_ids = df.index.unique()
    sequence_id = sequence_ids[0]

    # get rows of df with sequence_id
    mat = df.loc[sequence_id, :].values

    phrase = labels.loc[labels['sequence_id'] == sequence_id, 'phrase'].values[0]
    
    # get one sample from dataset   
    sample = next(iter(dataset))

    enc_input = sample[0][0][tf.newaxis, ...]
    dec_input = sample[0][1][tf.newaxis, ...]
    label = sample[1][tf.newaxis, ...]

    # with strategy.scope():
    model = Transformer(**model_params)
    
    model((enc_input, dec_input), training=False)
    
    # input_shape = ((None, input_length, input_dim))

    model.load_weights(weights_path)

    finger_gen = FingerGenModel(model, max_output_len, SOS_token, EOS_token)
    preds = finger_gen(inputs = tf.convert_to_tensor(mat))



    converter = tf.lite.TFLiteConverter.from_keras_model(finger_gen)
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    # converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()

    working_dir = "."

    tflite_model_path = path.join(working_dir, 'model.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)





