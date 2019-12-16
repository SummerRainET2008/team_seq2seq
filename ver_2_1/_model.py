from ver_2_1 import *
import tensorflow as tf

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz,
               num_layers, rnn_type):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.num_layers = num_layers
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    if rnn_type.lower() == "lstm":
      layer = tf.keras.layers.LSTM
    elif rnn_type.lower() == "gru":
      layer = tf.keras.layers.GRU
    else:
      assert False

    self.bi_layer = tf.keras.layers.Bidirectional(
      layer(
        enc_units, return_sequences=True, return_state=False,
        recurrent_initializer='glorot_uniform'
      ),
      merge_mode='concat'
    )
    self.enc_layers = [
      layer(
        enc_units, return_sequences=True, return_state=False,
        recurrent_initializer='glorot_uniform'
      )
      for _ in range(num_layers - 2)
    ]
    self.top = layer(
      enc_units, return_sequences=True,
      return_state=True,
      recurrent_initializer='glorot_uniform'
    )

  def call(self, x, hidden):
    x = self.embedding(x)
    # x = self.bi_layer(x, initial_state=hidden)
    # x = self.bi_layer(x, init_h, init_c)
    x = self.bi_layer(x)
    for i in range(len(self.enc_layers)):
      x = self.enc_layers[i](x)
    try:
      output, hidden_states = self.top(x)
    except ValueError:
      output, hidden_states, _ = self.top(x)

    # Encoder output shape: (batch size, sequence length, enc_hidden_units)
    return output, hidden_states

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query is hidden state of decoder
    # values is encoder output
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # Doing hidden with time_axis to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1) for applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    # turn score to weights whose sum is 1
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    # values.shape: (batch_size, seq_len, enc_hidden)
    context_vector = attention_weights * values  # (batch_size, seq_len, enc_hidden)
    context_vector = tf.reduce_sum(context_vector, axis=1)  # reduce seq_len dimension

    return context_vector, attention_weights


# Context vector shape: (batch size, enc_hidden_units)
# Attention weights shape: (batch_size, sequence_length, 1)

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, rnn_type):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    if rnn_type.lower() == "lstm":
      layer = tf.keras.layers.LSTM
    elif rnn_type.lower() == "gru":
      layer = tf.keras.layers.GRU
    else:
      assert False

    self.rnn = layer(self.dec_units, return_state=True,
                     return_sequences=True, recurrent_initializer='glorot_uniform')

    self.fc = tf.keras.layers.Dense(vocab_size)
    # use softmax

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    try:
      output, state = self.rnn(x)
    except ValueError:
      output, state, _ = self.rnn(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights

# Decoder output (x) shape: (batch_size, vocab size)

class Model(tf.keras.Model):
  def __init__(self, param):
    super(Model, self).__init__()
    self.param = param
    self.encoder = Encoder(param.vocab_cn_size, param.embedding_dim,
                           param.enc_units, param.batch_size,
                           param.enc_layers, param.rnn_type)
    self.decoder =  Decoder(param.vocab_en_size, param.embedding_dim,
                            param.dec_units, param.batch_size,
                            param.rnn_type)
    self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')


  def call(self, src, trg):
    loss = tf.cast(0, tf.float32)
    enc_hidden_init = self.encoder.initialize_hidden_state()
    enc_output, enc_hidden = self.encoder(src, enc_hidden_init)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims(tf.ones(tf.shape(trg)[0], dtype=tf.int32) * SOS, 1)
    # dec_input shape: [tf.shape(enc_output)[0], 1] means # batch size sentences
    # with only first one word SOS

    # Teacher forcing
    for t in tf.range(1, tf.shape(trg)[1]):
      predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden,
                                                enc_output)
      loss += self.loss_function(trg[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(trg[:, t], 1)

    return loss

  def predict(self, src, max_length_trg): # src.shape: (1, input_length)

    # hidden = [tf.zeros((src.shape[0], self.param.hidden_units))]
    # (vali_batch_sz, hidden_units)
    hidden = tf.zeros((1, self.param.enc_units))  # (1, hidden_units)
    enc_output, enc_hidden = self.encoder(src, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([SOS], 0)
    # [SOS]: shape: [x], dec_input shape: [1, x], x here is 1

    ids = []
    for _ in tf.range(max_length_trg):
      predictions, dec_hidden, attention_weights = self.decoder(
        dec_input, dec_hidden, enc_output
      )
      predicted_id = int(tf.argmax(predictions[0]))
      if predicted_id == EOS:
        break
      ids.append(predicted_id)
      # the predicted ID is fed back into the model
      dec_input = tf.expand_dims([predicted_id], 0)

    return tf.convert_to_tensor(ids, dtype=tf.int32)

  @tf.function(
    input_signature=(
      tf.TensorSpec(shape=[None], dtype=tf.int32),
      tf.TensorSpec(shape=[None, 30000], dtype=tf.float32)
    )
  )
  def loss_function(self, real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = self.loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_) # mean loss of a batch
