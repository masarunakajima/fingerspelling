import tensorflow as tf
import numpy as np
import pandas as pd
import json
import glob
from os import path
import sys
import os



def get_c2p(c2p_path):
    with open(c2p_path, 'r') as f:
        c2p = json.load(f)
    return c2p

def convert_phrase(phrase, c2p):
    return [c2p[char] for char in phrase]



def adjust_seq_len(mat, seq_len):
    if mat.shape[0] > seq_len:
        mat = mat[:seq_len,:]
    elif mat.shape[0] < seq_len:
        mat = np.pad(mat, ((0, seq_len - mat.shape[0]), (0,0)), 'constant', constant_values = 0)
    return mat


def get_padding_mask(seq_lens, max_len):
    mask_len = np.array(seq_lens) - max_len
    mask_len[mask_len < 0] = 0
    mask = np.zeros((len(seq_lens), max_len))
    for i, l in enumerate(mask_len):
        mask[i, -l:] = 1
    # conver to tensor
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    mask = tf.reshape(mask, (mask.shape[0], 1, 1, mask.shape[1]))
    return mask

def get_features(parquet_paths, selected_columns, input_len):
    df = pd.DataFrame()
    for parquet_path in parquet_paths:
      temp_df = pd.read_parquet(parquet_path, columns = selected_columns)
      df = pd.concat([df, temp_df])   
    df = df.fillna(0)
    grouped_df = df.groupby(df.index).apply(lambda x: x.values)
    grouped_values = grouped_df.apply(lambda x: adjust_seq_len(x, input_len))
    features = np.stack(grouped_values.values).astype(np.float32)
    return features, df.index.unique()


def prep_dataset_conv(parquet_paths, label_path, input_length, output_length, 
                 selected_columns, tokenizer):
    inputs, seq_ids = get_features(parquet_paths, selected_columns, input_length)
    phrase_df = pd.read_csv(label_path)
    phrase_df = phrase_df.set_index('sequence_id')
    phrase_df = phrase_df.loc[seq_ids]
    tokenized_seqs = tokenizer.texts_to_sequences(phrase_df['phrase'].values)
    labels = tf.keras.preprocessing.sequence.pad_sequences(tokenized_seqs, 
                                                               maxlen = output_length, 
                                                               padding = 'post', 
                                                               truncating = 'post')
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    return dataset



        

def prep_dataset_mask(parquet_paths, label_path, input_length, output_length,
                      selected_columns, tokenizer):
    inputs, seq_ids = get_features(parquet_paths, selected_columns, input_length)
    phrase_df = pd.read_csv(label_path)
    phrase_df = phrase_df.set_index('sequence_id')
    phrases = phrase_df.loc[seq_ids]['phrase'].values

    mask_seq = [np.ones(len(phrase)).astype(np.int32) for phrase in phrases]
    labels = tf.keras.preprocessing.sequence.pad_sequences(mask_seq,
                                                           value = 0,
                                                              maxlen = output_length,
                                                              padding = 'post',
                                                              truncating = 'post')
    # print(labels[0])
    
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    return dataset


def get_num_steps(parquet_paths, label_path, batch_size, file_steps):
    label_df = pd.read_csv(label_path)
    file_ids = [int(path.basename(parquet_path).split('.')[0]) for parquet_path in parquet_paths]
    num_steps = 0
    for i in range(0, len(parquet_paths), file_steps):
       # get rows of label_df that match the file_ids
       rows = label_df[label_df['file_id'].isin(file_ids[i:i+file_steps])] 
       num_steps += len(rows) // batch_size
       if len(rows) % batch_size != 0:
         num_steps += 1
    return num_steps



def data_generator(parquet_paths, label_path, input_length, output_length,
                      selected_columns, batch_size, file_steps, c2p):
    n_files = len(parquet_paths)
    # print(label_path)
    label_df = pd.read_csv(label_path)
    label_df = label_df.set_index('sequence_id')
    while True:
      for i in range(0, n_files, file_steps):
        batch_parquet_paths = parquet_paths[i:i + file_steps]
        features, sequence_ids = get_features(batch_parquet_paths, selected_columns, 
                                              input_length)
        # shuffle the features and sequence_ids
        shuffle_idx = np.random.permutation(len(sequence_ids))
        features = features[shuffle_idx]
        sequence_ids = sequence_ids[shuffle_idx]
        phrases = label_df.loc[sequence_ids]['phrase'].values
        phrases = [convert_phrase(phrase, c2p) for phrase in phrases]

        labels = tf.keras.preprocessing.sequence.pad_sequences(phrases,
                                                            value = 0,
                                                            maxlen = output_length,
                                                            padding = 'post',
                                                            truncating = 'post')
        for i in range(0, len(features), batch_size):
          yield features[i:i+batch_size], labels[i:i+batch_size]  


def data_generator_transformer(parquet_paths, label_path, input_length, output_length,
                      selected_columns, batch_size, file_steps, tokenizer):
    n_files = len(parquet_paths)
    # print(label_path)
    label_df = pd.read_csv(label_path)
    label_df = label_df.set_index('sequence_id')
    while True:
      for i in range(0, n_files, file_steps):
        batch_parquet_paths = parquet_paths[i:i + file_steps]
        features, sequence_ids = get_features(batch_parquet_paths, selected_columns, 
                                              input_length)
        # shuffle the features and sequence_ids
        shuffle_idx = np.random.permutation(len(sequence_ids))
        features = features[shuffle_idx]
        sequence_ids = sequence_ids[shuffle_idx]
        phrases = label_df.loc[sequence_ids]['phrase'].values
        tokens = tokenizer.texts_to_sequences(phrases)
        # print(tokens[0])
        enc_input = [[tokenizer.word_index['<SOS>']] + token for token in tokens]
        enc_input = tf.keras.preprocessing.sequence.pad_sequences(enc_input,
                                                            value = 0,
                                                            maxlen = output_length,
                                                            padding = 'post',
                                                            truncating = 'post')
        enc_input = tf.cast(enc_input, tf.int32)
        labels = [token + [tokenizer.word_index['<EOS>']] for token in tokens]
        labels = tf.keras.preprocessing.sequence.pad_sequences(labels,
                                                            value = 0,
                                                            maxlen = output_length,
                                                            padding = 'post',
                                                            truncating = 'post')
        labels = tf.cast(labels, tf.int32)
        # print(enc_input.shape, labels.shape)
        for i in range(0, len(features), batch_size):
          yield (features[i:i+batch_size],enc_input[i:i+batch_size]), labels[i:i+batch_size]  


def mask_data_generator(parquet_paths, label_path, input_length, output_length,
                      selected_columns, batch_size, file_steps):
    n_files = len(parquet_paths)
    # print(label_path)
    label_df = pd.read_csv(label_path)
    label_df = label_df.set_index('sequence_id')
    while True:
      index = 0
      while index < n_files:
        batch_parquet_paths = parquet_paths[index:index + file_steps]
        index += file_steps
        features, sequence_ids = get_features(batch_parquet_paths, selected_columns, 
                                              input_length)
        # shuffle the features and sequence_ids
        shuffle_idx = np.random.permutation(len(sequence_ids))
        features = features[shuffle_idx]
        sequence_ids = sequence_ids[shuffle_idx]
        phrases = label_df.loc[sequence_ids]['phrase'].values
        mask_seq = [np.ones(len(phrase)).astype(np.int32) for phrase in phrases]
        labels = tf.keras.preprocessing.sequence.pad_sequences(mask_seq,
                                                            value = 0,
                                                            maxlen = output_length,
                                                            padding = 'post',
                                                            truncating = 'post')
        for i in range(0, len(features), batch_size):
          yield features[i:i+batch_size], labels[i:i+batch_size]


def prep_dataset(parquet_paths, label_path, input_length, output_length, 
                 selected_columns, tokenizer):
    df = pd.DataFrame()
    for parquet_path in parquet_paths:
      temp_df = pd.read_parquet(parquet_path)
      df = pd.concat([df, temp_df])
    
    

    df = df[selected_columns]
    df = df.dropna(how='all')
    sequence_ids = df.index.unique()
    num_samples = len(sequence_ids)

    df = df.fillna(0)
    grouped_df = df.groupby(df.index).apply(lambda x: x.values)
    grouped_values = grouped_df.apply(lambda x: adjust_seq_len(x, input_length))
    encoder_inputs = np.stack(grouped_values.values).astype(np.float32)
    # encoder_inputs = tf.convert_to_tensor(grouped_values)


    # get the phrase rows for each sequence_ids in order
    phrase_df = pd.read_csv(label_path)
    phrase_df = phrase_df.set_index('sequence_id')
    phrase_df = phrase_df.loc[sequence_ids]

    tokenized_seqs = tokenizer.texts_to_sequences(phrase_df['phrase'].values)

    decoder_input_seqs = [[tokenizer.word_index['<SOS>']] + seq for seq in tokenized_seqs]
    decoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(decoder_input_seqs, 
                                                                   maxlen = output_length, 
                                                                   padding = 'post', 
                                                                   truncating = 'post')
    decoder_inputs = decoder_inputs.astype(np.int32)

    label_seqs = [seq + [tokenizer.word_index['<EOS>']] for seq in tokenized_seqs]
    label_seqs = tf.keras.preprocessing.sequence.pad_sequences(label_seqs, maxlen = output_length, padding = 'post', truncating = 'post')
    label_seqs = label_seqs.astype(np.int32)

    # print(encoder_inputs.shape)
    # print(decoder_inputs.shape)
    # print(label_seqs.shape)


    dataset = tf.data.Dataset.from_tensor_slices(((encoder_inputs, decoder_inputs), label_seqs))
    return dataset


def prep_dataset_nodrop(parquet_paths, label_path, input_length, output_length, 
                 selected_columns, tokenizer):
    df = pd.DataFrame()
    for parquet_path in parquet_paths:
      temp_df = pd.read_parquet(parquet_path, columns = selected_columns)
      df = pd.concat([df, temp_df])   

    df.fillna(0, inplace = True)
    grouped_df = df.groupby(df.index).apply(lambda x: x.values)
    grouped_values = grouped_df.apply(lambda x: adjust_seq_len(x, input_length))
    encoder_inputs = np.stack(grouped_values.values).astype(np.float32)
    # encoder_inputs = tf.convert_to_tensor(grouped_values)


    # get the phrase rows for each sequence_ids in order
    sequence_ids = df.index.unique()
    phrase_df = pd.read_csv(label_path)
    phrase_df = phrase_df.set_index('sequence_id')
    phrase_df = phrase_df.loc[sequence_ids]

    tokenized_seqs = tokenizer.texts_to_sequences(phrase_df['phrase'].values)

    decoder_input_seqs = [[tokenizer.word_index['<SOS>']] + seq for seq in tokenized_seqs]
    decoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(decoder_input_seqs, 
                                                                   maxlen = output_length, 
                                                                   padding = 'post', 
                                                                   truncating = 'post')
    decoder_inputs = decoder_inputs.astype(np.int32)

    label_seqs = [seq + [tokenizer.word_index['<EOS>']] for seq in tokenized_seqs]
    label_seqs = tf.keras.preprocessing.sequence.pad_sequences(label_seqs, maxlen = output_length, padding = 'post', truncating = 'post')
    label_seqs = label_seqs.astype(np.int32)

    # print(encoder_inputs.shape)
    # print(decoder_inputs.shape)
    # print(label_seqs.shape)


    dataset = tf.data.Dataset.from_tensor_slices(((encoder_inputs, decoder_inputs), label_seqs))
    return dataset


def prep_dataset_allstart(parquet_paths, label_path, input_length, output_length, 
                 selected_columns, tokenizer):
    df = pd.DataFrame()
    for parquet_path in parquet_paths:
      temp_df = pd.read_parquet(parquet_path, columns = selected_columns)
      df = pd.concat([df, temp_df])   

    df.fillna(0, inplace = True)
    grouped_df = df.groupby(df.index).apply(lambda x: x.values)
    grouped_values = grouped_df.apply(lambda x: adjust_seq_len(x, input_length))
    encoder_inputs = np.stack(grouped_values.values).astype(np.float32)
    # encoder_inputs = tf.convert_to_tensor(grouped_values)


    # get the phrase rows for each sequence_ids in order
    sequence_ids = df.index.unique()
    phrase_df = pd.read_csv(label_path)
    phrase_df = phrase_df.set_index('sequence_id')
    phrase_df = phrase_df.loc[sequence_ids]

    tokenized_seqs = tokenizer.texts_to_sequences(phrase_df['phrase'].values)

    decoder_input_seqs = [[tokenizer.word_index['<SOS>']]*len(seq)  for seq in tokenized_seqs]
    decoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(decoder_input_seqs, 
                                                                   maxlen = output_length, 
                                                                   padding = 'post', 
                                                                   truncating = 'post')
    decoder_inputs = decoder_inputs.astype(np.int32)

    label_seqs = [seq + [tokenizer.word_index['<EOS>']] for seq in tokenized_seqs]
    label_seqs = tf.keras.preprocessing.sequence.pad_sequences(label_seqs, maxlen = output_length, padding = 'post', truncating = 'post')
    label_seqs = label_seqs.astype(np.int32)

    # print(encoder_inputs.shape)
    # print(decoder_inputs.shape)
    # print(label_seqs.shape)


    dataset = tf.data.Dataset.from_tensor_slices(((encoder_inputs, decoder_inputs), label_seqs))
    return dataset


def prep_dataset_allstart_enc(parquet_paths, label_path, input_length, output_length, 
                 selected_columns, tokenizer):
    df = pd.DataFrame()
    for parquet_path in parquet_paths:
      temp_df = pd.read_parquet(parquet_path, columns = selected_columns)
      df = pd.concat([df, temp_df])   

    df.fillna(0, inplace = True)
    grouped_df = df.groupby(df.index).apply(lambda x: x.values)
    grouped_values = grouped_df.apply(lambda x: adjust_seq_len(x, input_length))
    encoder_inputs = np.stack(grouped_values.values).astype(np.float32)
    # encoder_inputs = tf.convert_to_tensor(grouped_values)


    # get the phrase rows for each sequence_ids in order
    sequence_ids = df.index.unique()
    phrase_df = pd.read_csv(label_path)
    phrase_df = phrase_df.set_index('sequence_id')
    phrase_df = phrase_df.loc[sequence_ids]

    tokenized_seqs = tokenizer.texts_to_sequences(phrase_df['phrase'].values)

    # decoder_input_seqs = [[tokenizer.word_index['<SOS>']]*len(seq)  for seq in tokenized_seqs]
    # decoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(decoder_input_seqs, 
    #                                                                maxlen = output_length, 
    #                                                                padding = 'post', 
    #                                                                truncating = 'post')
    # decoder_inputs = decoder_inputs.astype(np.int32)

    label_seqs = [seq + [tokenizer.word_index['<EOS>']] for seq in tokenized_seqs]
    label_seqs = tf.keras.preprocessing.sequence.pad_sequences(label_seqs, maxlen = output_length, padding = 'post', truncating = 'post')
    label_seqs = label_seqs.astype(np.int32)

    # print(encoder_inputs.shape)
    # # print(decoder_inputs.shape)
    # print(label_seqs.shape)


    dataset = tf.data.Dataset.from_tensor_slices((encoder_inputs, label_seqs))
    return dataset



def get_tokenizer(c2t_path):
    with open (c2t_path, "r") as f:
        character_map = json.load(f)
    t2c = {j:i for i,j in character_map.items()}
    t2c_list = [t2c[i] for i in range(len(t2c))]

    # Create the tokenizer with special tokens
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True,filters='',  lower=False)



    # Fit the tokenizer on the characters
    tokenizer.fit_on_texts(t2c_list)
    tokenizer.word_index['<PAD>'] = 0
    tokenizer.word_index['<EOS>'] = len(tokenizer.word_index)
    tokenizer.word_index['<SOS>'] = len(tokenizer.word_index)

    # Create the reverse lookup from index to character
    tokenizer.index_word = {v:k for k,v in tokenizer.word_index.items()}

    return tokenizer



def positional_encoding(length, depth):
  half_depth = depth//2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  exponents = np.arange(half_depth)[np.newaxis, :]*2/depth   # (1, half_depth)

  denom = 10000**exponents         # (1, depth)
  angles = positions / denom      # (pos, half_depth)

  pos_encoding = np.zeros((length, depth)) # (seq, depth*2)
  pos_encoding[:,::2] = np.sin(angles)
  pos_encoding[:,1::2] = np.cos(angles)


  return tf.cast(pos_encoding, dtype=tf.float32)



class PositionalEmbeddingSeq(tf.keras.layers.Layer):
  def __init__(self, d_model, seq_len, input_dim):
    super().__init__()
    self.d_model = d_model
    self.linear = tf.keras.layers.Dense(d_model) 
    self.pos_encoding = positional_encoding(length=seq_len, depth=d_model)
    self.masking = tf.keras.layers.Masking(mask_value=0.0, input_shape=(None, seq_len, input_dim))

  def compute_mask(self, *args, **kwargs):
    return self.masking.compute_mask(*args, **kwargs)


  def call(self, x):
    x = self.masking(x)
    x = self.linear(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :, :]
    return x




  def call(self, x):
    x = self.masking(x)
    x = self.linear(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :, :]
    return x




class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x
  


class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x  


class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output, attn_scores = self.mha(
        query=x,
        value=x,
        key=x, 
        return_attention_scores=True)
    self.last_attn_scores = attn_scores
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x


class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output, attn_scores = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True, 
        return_attention_scores=True)
    self.last_attn_scores = attn_scores
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
  


class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x



class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model= d_model, dff = dff, dropout_rate=dropout_rate)

  def call(self, x):
    x = self.self_attention(x)
    # print(x.shape)
    x = self.ffn(x)
    return x



class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, seq_len, input_dim, num_layers, d_model, num_heads,
               dff, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbeddingSeq(
        d_model = d_model, seq_len = seq_len, input_dim = input_dim)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.convBlock = conv1DBlock(filters=124, kernel_size=10, strides=1, padding='same', 
                                 activation='relu', use_bias=True, name='conv1d_block',
                                 pool_size=2, pool_strides=1, pool_padding='same', 
                                 dropout_rate=0.1)

  def call(self, x):
    # x = self.convBlock(x)
    # `x` is token-IDs shape: (batch, seq_len, input_dim)
    x = self.pos_embedding(x)  # output Shape `(batch_size, seq_len, d_model)`.

    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.
  


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model=d_model, dff=dff, dropout_rate=dropout_rate)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x


class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
               dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                             d_model=d_model)
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
  
class DecoderSeq(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, seq_len,
               input_dim,
               dropout_rate=0.1):
    super(DecoderSeq, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbeddingSeq(d_model=d_model, 
                                                seq_len = seq_len, 
                                                input_dim = input_dim)
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


class TransformerSeq(tf.keras.Model):
  def __init__(self, *, seq_len, input_dim, num_layers, d_model, num_heads, dff,
               decoder_seq_len, decoder_dim, target_vocab_size, dropout_rate=0.1):
    super().__init__()
    self.encoder = Encoder(seq_len=seq_len, input_dim=input_dim,
                           num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           dropout_rate=dropout_rate)

    self.decoder = DecoderSeq(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           seq_len=decoder_seq_len, input_dim = decoder_dim,
                           dropout_rate=dropout_rate)
    self.seq_len = seq_len
    self.input_dim = input_dim
    self.d_model = d_model
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.dff = dff
    self.decoder_seq_len = decoder_seq_len
    self.decoder_dim = decoder_dim
    self.dropout_rate = dropout_rate


    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    context, x  = inputs  
    # print(context.shape)
    # print(x.shape)  


    context = self.encoder(context)  # (batch_size, context_len, d_model)

    x = self.decoder(x, context)  # (batch_size, target_len, d_model)

    # Final linear layer output.
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    # Return the final output and the attention weights.
    return logits
  






class Transformer(tf.keras.Model):
  def __init__(self, *, seq_len, input_dim, num_layers, d_model, num_heads, dff,
               target_vocab_size, dropout_rate=0.1):
    super().__init__()
    self.encoder = Encoder(seq_len=seq_len, input_dim=input_dim,
                           num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           dropout_rate=dropout_rate)

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate)
    self.seq_len = seq_len
    self.d_model = d_model
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.dff = dff
    self.target_vocab_size = target_vocab_size
    self.input_dim = input_dim


    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    context, x  = inputs  
    # print(context.shape)
    # print(x.shape)  


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
  
  def generate(self, inputs, output_len, start_token):
    batch_size = tf.shape(inputs)[0]
    dec_input = tf.ones((batch_size, 1), dtype=tf.int32) * start_token
    print(inputs.shape)
    print(dec_input.shape)
    context = self.encoder(inputs)
    dec_output = self.decoder(dec_input, context)

    logits = self.final_layer(dec_output)
    for i in range(output_len-1):
      dec_output = self.decoder(dec_input, context)
      logits = self.final_layer(dec_output)
      predicted_tokens = tf.cast(tf.argmax(logits, axis=-1)[:, -1:], tf.int32)
      dec_input = tf.concat([dec_input, predicted_tokens], axis=-1)
    return logits
      

  
def preprocess(inputs, seq_len):

  x = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
  x = tf.image.resize(x, (tf.shape(x)[0], seq_len))
  return x


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  


def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss



def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=-1)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)


def binary_loss(label, pred):
  loss_object = tf.keras.losses.BinaryCrossentropy(
    from_logits=True)
  loss = loss_object(label, pred)

  return loss

def binary_accuracy(label, pred):
  pred = pred > 0.5
  label = tf.cast(label, pred.dtype)
  match = tf.cast(label == pred, tf.float32)
  accuracy = tf.reduce_mean(match)
  return accuracy


class conv1DBlock(tf.keras.layers.Layer):
    def __init__(self,  filters, kernel_size, strides, padding, activation, 
                 use_bias, name, 
                 pool_size=2, pool_strides=2, pool_padding='valid',
                 dropout_rate=0.2):
        super(conv1DBlock, self).__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.pool_padding = pool_padding
        self.dropout_rate = dropout_rate




        self.conv1d = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                             strides=strides, padding=padding, 
                                             activation=activation, use_bias=use_bias, 
                                             name=name)
        # self.ln = tf.keras.layers.LayerNormalization(name=name)
        self.bn = tf.keras.layers.BatchNormalization(name=name)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name=name)
        self.pool = tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=pool_strides, 
                                                 padding=pool_padding,
                                                 name=name)
           
    
    def call(self, inputs, training=False):
        x = self.conv1d(inputs)
        x = self.bn(x, training=training)
        x = self.pool(x)
        return x

class conv1DChain(tf.keras.layers.Layer):
    def __init__(self, chain_num, filters, kernel_size, strides, padding, activation, 
                 use_bias, name,
                pool_size=2, pool_strides=2, pool_padding='valid',
                dropout_rate=0.2):
        super(conv1DChain, self).__init__(name=name)
        self.chain_num = chain_num
        self.conv1d_blocks = []
        for i in range(chain_num):
            self.conv1d_blocks.append(conv1DBlock(filters=filters, kernel_size=kernel_size, 
                                                  strides=strides, padding=padding, 
                                                  activation=activation, use_bias=use_bias, 
                                                  name=f'conv1Dblock_{i}', pool_size=pool_size, 
                                                  pool_strides=pool_strides, 
                                                  pool_padding=pool_padding,
                                                  dropout_rate=dropout_rate))
    def call(self, inputs, training=False):
        x = inputs
        for i in range(self.chain_num):
            x = self.conv1d_blocks[i](x, training=training)
        return x


if __name__ == '__main__':

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
    output_length = 32

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
        'target_vocab_size': vocab_size,
        'dropout_rate': dropout_rate
    }

    # with strategy.scope():
    model = Transformer(**model_param)



    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                        epsilon=1e-9)
    
    model.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy])


    num_files = len(train_files)

    valid_dataset = prep_dataset_nodrop(valid_files, label_path, input_length, 
                                 output_length, selected_columns, tokenizer)
    valid_dataset = valid_dataset.batch(batch_size)
    valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)

    for epoch in range(epochs):
        print("Grand Epoch {}/{}".format(epoch+1, epochs))
        for i in range(num_files//n_files):
            print("dataset {}/{}".format(i+1, num_files//n_files))
            current = i*n_files
            dataset = prep_dataset_nodrop(train_files[current:(current+n_files)], label_path, 
                                    input_length, output_length, selected_columns, tokenizer)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            history = model.fit(dataset, validation_data = valid_dataset, epochs=runs_per_epoch)
            tf.keras.backend.clear_session()
        if current+n_files != num_files:
            dataset = prep_dataset_nodrop(train_files[(current+n_files):], label_path, 
                                    input_length, output_length, selected_columns, tokenizer)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            history = model.fit(dataset,validation_data = valid_dataset, epochs=runs_per_epoch)

    model_paths = glob.glob(path.join(model_dir, '*'))
    model_nums = [int(model_path.split('_')[-1]) for model_path in model_paths]
    if len(model_nums) == 0:
        max_num = 0
    else:
        max_num = max(model_nums)
    new_model_dir = path.join(model_dir, 'transformer_{}'.format(max_num+1))
    os.makedirs(new_model_dir, exist_ok=True)
    model.save_weights(path.join(new_model_dir, 'weights.h5'))
    with open(path.join(new_model_dir, 'model_param.json'), 'w') as f:
        json.dump(model_param, f)
    with open(path.join(new_model_dir, 'history.json'), 'w') as f:
        json.dump(history.history, f)





    # transformer.fit(dataset, epochs=epochs)
