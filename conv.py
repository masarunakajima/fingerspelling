from model import *


    
class conv1DBlockLn(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, activation, use_bias, name, 
                 dropout_rate=0.2):
        super(conv1DBlockLn, self).__init__(name=name)
        self.conv1d = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                             strides=strides, padding=padding, 
                                             activation=activation, use_bias=use_bias, 
                                             name=name)
        self.ln = tf.keras.layers.LayerNormalization(name=name)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name=name)
        self.pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid', 
                                         name=name)
    def call(self, inputs, training=False):
        x = self.conv1d(inputs)
        x = self.ln(x, training=training)
        x = self.pool(x)
        return x


def get_model_mask(output_len, channels, exp, filters, kernel_size, vocab_size, dropout_rate=0.2):
    input_len = output_len * 2 ** exp
    inputs = tf.keras.Input(shape=(input_len, channels), name='input')
    x = inputs
    for i in range(exp):
        x = conv1DBlockLn(filters=filters, kernel_size=kernel_size, strides=1, padding='same',
                        activation='relu', use_bias=False, 
                        dropout_rate = dropout_rate, name=f'conv1d_{i}')(x)
    # x = EncoderLayer(d_model=filters, num_heads=8, dff=128*2, dropout_rate=dropout_rate)(x)
    x = tf.keras.layers.Dense(units=filters,  activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.1 )(x)
    x = tf.keras.layers.Dense(units=filters,  activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.8)(x)
    x = tf.keras.layers.Dense(units=1, activation='sigmoid', name='sigmoid')(x)
    x = tf.squeeze(x, axis=-1)

    return tf.keras.Model(inputs=inputs, outputs=x, name='conv1d_model_mask')

# # main function
if __name__ == '__main__':

    n_hand_lmrks = 21
    handedness = ['left', 'right']
    selected_columns = []
    for hand in handedness:
        for i in range(n_hand_lmrks):
            selected_columns.append(f'x_{hand}_hand_{i}')
            selected_columns.append(f'y_{hand}_hand_{i}')
            # selected_columns.append(f'{hand}_{i}')
    n_pose_lmrks = 33
    for i in range(n_pose_lmrks):
        selected_columns.append(f'x_pose_{i}')
        selected_columns.append(f'y_pose_{i}')
        # selected_columns.append(f'pose_{i}')

    model_dir = "models/mask_model"
    data_path = 'data/train_landmarks'
    parquet_files = glob.glob(path.join(data_path, '*.parquet'))
    valid_files = [parquet_files[0]]
    train_files = parquet_files[1:]


    label_path = 'train.csv'
    c2p_path = 'character_to_prediction_index.json'

    tokenizer = get_tokenizer(c2p_path)

    # set parameters
    # input_len = 1234230
    channels = len(selected_columns)
    filters = 128
    kernel_size = 10
    output_len = 32
    exp = 4
    dropout_rate = 0.1
    input_len = output_len * (2 ** exp)
    vocab_size = len(tokenizer.word_index)
    batch_size = 128
    epochs = 5
    runs_per_epoch = 5
    file_steps =  10

    meta_param = {'selected_columns': selected_columns,}    
    
    model_param = {
        'output_len': output_len,
        'channels': channels,
        'exp': exp,
        'filters': filters,
        'kernel_size': kernel_size,
        'vocab_size': vocab_size,
        'dropout_rate': dropout_rate,
    }

    # get model
    model = get_model_mask(**model_param)
    model.summary()

    learning_rate = CustomSchedule(channels)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                        epsilon=1e-9)
    
    model.compile(optimizer=optimizer, loss=binary_loss,
                  metrics=[binary_accuracy])
    
    train_generator = lambda: mask_data_generator(train_files, label_path, input_len, 
                                         output_len, selected_columns, batch_size, 
                                         file_steps)
    train_dataset = tf.data.Dataset.from_generator(train_generator,
                                                output_types=(tf.float32, tf.float32),
                                                output_shapes=((None, input_len, channels),
                                                                (None, output_len)))
 
    train_steps = get_num_steps(train_files, label_path, batch_size, file_steps)
    valid_generator = lambda: mask_data_generator(valid_files, label_path, input_len,
                                            output_len, selected_columns, batch_size,
                                            file_steps)
    valid_dataset = tf.data.Dataset.from_generator(valid_generator,
                                                output_types=(tf.float32, tf.float32),
                                                output_shapes=((None, input_len, channels),
                                                                (None, output_len)))
    
    valid_steps = get_num_steps(valid_files, label_path, batch_size, file_steps)
    print(train_steps)
    print(valid_steps)
    
    history = model.fit(train_dataset, validation_data = valid_dataset, epochs=epochs,
                        steps_per_epoch=train_steps, validation_steps=valid_steps)
    print("after fit")

    model_paths = glob.glob(path.join(model_dir, '*'))
    model_nums = [int(model_path.split('_')[-1]) for model_path in model_paths]
    if len(model_nums) == 0:
        max_num = 0
    else:
        max_num = max(model_nums)
    new_model_dir = path.join(model_dir, 'model_{}'.format(max_num+1))
    os.makedirs(new_model_dir, exist_ok=True)
    model.save_weights(path.join(new_model_dir, 'weights.h5'))
    with open(path.join(new_model_dir, 'model_param.json'), 'w') as f:
        json.dump(model_param, f)
    with open(path.join(new_model_dir, 'meta_param.json'), 'w') as f:
        json.dump(meta_param, f)
    with open(path.join(new_model_dir, 'history.json'), 'w') as f:
        json.dump(history.history, f)








    
