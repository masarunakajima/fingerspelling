from model import *
import tensorflow as tf

class PurePositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, d_model, seq_len):
    super().__init__()
    self.d_model = d_model
    self.pos_encoding = positional_encoding(length=seq_len, depth=d_model)
    # self.linear = tf.keras.layers.Dense(d_model)
    # self.masking = tf.keras.layers.Masking(mask_value=0.0, input_shape=(None, seq_len, d_model))

#   def compute_mask(self, *args, **kwargs):
#     return self.masking.compute_mask(*args, **kwargs)
  
  def call(self, x):    

    x = self.pos_encoding[tf.newaxis, :, :]
    # x = self.masking(x)
    # x = self.linear(x)

    return x

class SelfDecoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, output_seq_len,
               dropout_rate=0.1):
    super(SelfDecoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PurePositionalEmbedding(d_model=d_model,
                                                 seq_len=output_seq_len)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.last_attn_scores = None

  def call(self, x, context):
    # `x` is token-IDs shape (batch, target_seq_len)
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

    x = self.dropout(x)

    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context)

    self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    # The shape of x is (batch_size, target_seq_len, d_model).
    return x
  

class SelfTransformer(tf.keras.Model):
  def __init__(self, *, seq_len, input_dim, num_layers, d_model, num_heads, dff,
               output_seq_len, target_vocab_size, dropout_rate=0.1):
    super().__init__()
    self.encoder = Encoder(seq_len=seq_len, input_dim=input_dim,
                           num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           dropout_rate=dropout_rate)

    self.decoder = SelfDecoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           output_seq_len=output_seq_len,
                           dropout_rate=dropout_rate)
    self.seq_len = seq_len
    self.d_model = d_model
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.dff = dff
    self.output_seq_len = output_seq_len
    self.target_vocab_size = target_vocab_size
    self.input_dim = input_dim


    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    context, x  = inputs
    print(context.shape)
    print(x.shape)

    context = self.encoder(context)  # (batch_size, context_len, d_model)


    x = self.decoder(x, context)  # (batch_size, target_len, d_model)

    # Final linear layer output.
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits
  







# main function
if __name__ == "__main__":

    # Verify GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')


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


    model_dir = "models"


    data_path = 'data/train_landmarks'
    parquet_files = glob.glob(path.join(data_path, '*.parquet'))

    valid_files = [parquet_files[0]]
    train_files = parquet_files[1:]


    label_path = 'train.csv'
    c2p_path = 'character_to_prediction_index.json'

    input_length = 400
    input_dim = len(selected_columns)
    output_seq_len = 32

    tokenizer = get_tokenizer(c2p_path)
    n_files = 10




    d_model = 128*2
    vocab_size = len(tokenizer.word_index)

    num_layers = 4
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    batch_size = 128
    epochs = 5
    runs_per_epoch = 5

    model_param ={
        'seq_len': input_length,
        'input_dim': input_dim,
        'num_layers': num_layers,
        'd_model': d_model,
        'num_heads': num_heads,
        'dff': dff,
        'output_seq_len': output_seq_len, # 'output_seq_len': 'auto
        'target_vocab_size': vocab_size,
        'dropout_rate': dropout_rate
    }

    # with strategy.scope():
    model = SelfTransformer(**model_param)



    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                        epsilon=1e-9)
    
    model.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy])


    num_files = len(train_files)

    valid_dataset = prep_dataset_allstart(valid_files, label_path, input_length, 
                                 output_seq_len, selected_columns, tokenizer)
    valid_dataset = valid_dataset.batch(batch_size)
    valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)
  
    for epoch in range(epochs):
        print("Grand Epoch {}/{}".format(epoch+1, epochs))
        for i in range(num_files//n_files):
            print("dataset {}/{}".format(i+1, num_files//n_files))
            current = i*n_files
            dataset = prep_dataset_allstart(train_files[current:(current+n_files)], label_path, 
                                    input_length, output_seq_len, selected_columns, tokenizer)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            history = model.fit(dataset, validation_data = valid_dataset, epochs=runs_per_epoch)
            tf.keras.backend.clear_session()
        if current+n_files != num_files:
            dataset = prep_dataset_allstart(train_files[(current+n_files):], label_path, 
                                    input_length, output_seq_len, selected_columns, tokenizer)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            history = model.fit(dataset,validation_data = valid_dataset, epochs=runs_per_epoch)

    model_paths = glob.glob(path.join(model_dir, 'selftransformer_*'))
    model_nums = [int(model_path.split('_')[-1]) for model_path in model_paths]
    if len(model_nums) == 0:
        max_num = 0
    else:
        max_num = max(model_nums)
    new_model_dir = path.join(model_dir, 'selftransformer_{}'.format(max_num+1))
    os.makedirs(new_model_dir, exist_ok=True)
    model.save_weights(path.join(new_model_dir, 'weights.h5'))
    with open(path.join(new_model_dir, 'model_param.json'), 'w') as f:
        json.dump(model_param, f)
    with open(path.join(new_model_dir, 'history.json'), 'w') as f:
        json.dump(history.history, f)

