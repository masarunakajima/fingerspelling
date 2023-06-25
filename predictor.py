

from model import Transformer, prep_dataset, get_tokenizer, masked_accuracy, masked_loss
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


class FingerGenModelSeq(tf.keras.Model):
    def __init__(self, transformer, max_output_len, SOS_token, EOS_token):
        super(FingerGenModelSeq, self).__init__()
        self.transformer = transformer
        self.max_output_len = max_output_len
        self.sos = SOS_token
        self.eos = EOS_token




    def call(self, inputs):

        x = tf.cast(inputs, tf.float32)[None]
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        x = tf.image.resize(x, (tf.shape(x)[0], self.transformer.seq_len))
        # print(x.shape)
        ta = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False)
        ta.write(0, tf.expand_dims(self.sos, 0))
        dec_input = tf.transpose(ta.stack())
        # logits = self.transformer([x, dec_input], training=False)
        # dec_input = tf.expand_dims([SOS_token], 0)
        logits = self.transformer([x, dec_input], training=False)


        # @tf.function
        # def tensor_iteration():
        #     x = tf.constant([1, 2, 3, 4, 5])
        #     sum = tf.constant(0)

        #     for i in tf.range(tf.size(x)):
        #         sum += x[i]

        #     return sum

        return logits[0, :, 1:-2]
        # return ta





class FingerGenModel(tf.keras.Model):
    def __init__(self, transformer, max_output_len, SOS_token, EOS_token, vocab_size):
        super(FingerGenModel, self).__init__()
        self.transformer = transformer
        self.max_output_len = max_output_len
        self.sos = SOS_token
        self.eos = EOS_token

        class_mask = np.ones((1,1,vocab_size))
        class_mask[0,0,0] = 0
        class_mask[0,0,-1] = 0
        self.class_mask = tf.constant(class_mask, dtype=tf.float32)


        

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, len(selected_columns)],
                                                dtype=tf.float32, name='inputs')])
    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)[None]
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        x = tf.image.resize(x, (tf.shape(x)[0], self.transformer.seq_len))


        seq_mask = np.zeros((self.max_output_len))
        seq_mask[0] = 1
        seq_mask = tf.constant(seq_mask, dtype=tf.float32)

        dec_input = np.zeros((1, self.max_output_len))
        dec_input[0, 0] = self.sos
        dec_input = tf.constant(dec_input, dtype=tf.int32)

        logits = self.transformer([x, dec_input], training=False)
        logits *= self.class_mask

        for i in tf.range(self.max_output_len-1):
            # set second dimension 1 up to i 

            logits = self.transformer([x, dec_input], training=False)
            logits *= self.class_mask
            # ith_logits = logits[0, i, :]
            # preds = tf.argmax(ith_logits, -1, output_type=tf.int32)
            max_token = tf.argmax(logits[0, i, :], -1, output_type=tf.int32)
            if max_token == self.eos:
                break
            seq_mask = tf.where(tf.equal(tf.range(self.max_output_len), i),
                                tf.ones_like(seq_mask),seq_mask)
            dec_input = tf.where(tf.equal(tf.range(self.max_output_len), i+1),
                                max_token, dec_input)
            # print(dec_input[0,:i])

        # get the logits where seq_mask is 1
        outputs = logits[0]
        outputs = outputs[seq_mask ==1]
        outputs = outputs[:,1:-2]

        return {'outputs': outputs, 'dec_input': dec_input[0]}



if __name__ == "__main__":

    # Verify GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus))

    # Enable all available GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)





    model_path = "models/transformer_15"
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

    tokenizer = get_tokenizer(c2p_path)
    vocab_size = len(tokenizer.word_index) 
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
    model.compile(
        loss=masked_loss,
        metrics=[masked_accuracy])
    
    dataset = dataset.batch(128)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    loss, acc = model.evaluate(dataset, verbose=1)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))


    finger_gen = FingerGenModel(
        transformer=model, 
        max_output_len = output_length, 
        SOS_token = SOS_token, EOS_token = EOS_token, 
        vocab_size=vocab_size)
    preds = finger_gen(inputs = tf.convert_to_tensor(mat))



    converter = tf.lite.TFLiteConverter.from_keras_model(finger_gen)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()

    working_dir = "."

    tflite_model_path = path.join(working_dir, 'model.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    interpreter = tf.lite.Interpreter(tflite_model_path)
    REQUIRED_SIGNATURE = "serving_default"
    REQUIRED_OUTPUT = "outputs"

    with open (c2p_path, "r") as f:
        character_map = json.load(f)

    rev_character_map = {v: k for k, v in character_map.items()}

    found_signatures = list(interpreter.get_signature_list().keys())

    prediction_fn = interpreter.get_signature_runner(REQUIRED_SIGNATURE)
    frames = mat

    for sequence_id in sequence_ids[:10]:

        mat = df.loc[sequence_id, :].values
        phrase = labels.loc[labels['sequence_id'] == sequence_id, 'phrase'].values[0]
        output = prediction_fn(inputs = mat)
        preds = np.argmax(output[REQUIRED_OUTPUT], axis=1)
        prediction_str = "".join([rev_character_map.get(s,"") for s in preds])
        print("Phrase     :", phrase)
        print("Prediction :",prediction_str)
        dec_input = output['dec_input']
        print(dec_input)
        print("")



    dataset = prep_dataset([parquet_files[0]], label_path,
                            input_length, output_length, selected_columns, tokenizer)
    for data in dataset.take(4):
        enc_input = data[0][0][tf.newaxis, ...]
        dec_input = data[0][1][tf.newaxis, ...]
        dec_input = tf.ones_like(dec_input) * SOS_token
        label = data[1][tf.newaxis, ...]
        output = model((enc_input, dec_input), training=False)
        token = tf.argmax(output[0, :, :], -1, output_type=tf.int32)
        print(token)
        accuracy = masked_accuracy(label, output)
        print(accuracy)

        





