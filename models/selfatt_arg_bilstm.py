"""BiLSTM models for argument mining with self-attention mechanism
"""

import numpy
import keras.backend as K
import tensorflow as tf
from keras import optimizers, layers
from keras.models import Model
from keras.initializers import Ones, Zeros

from models.arg_bilstm import ArgBiLSTM


class LayerNormalization(layers.Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape


# It's safe to use a 1-d mask for self-attention
class ScaledDotProductAttention():
    def __init__(self, attn_dropout=0.1):
        self.dropout = layers.Dropout(attn_dropout, name='attention_matrix')

    def __call__(self, q, k, v, mask):   # mask_k or mask_qk
        temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
        # shape=(batch, q.shape[1], k.shape[1])
        attn = layers.Lambda(
            lambda x: K.batch_dot(x[0], x[1], axes=[2,2]) / temper)([q, k])
        if mask is not None:
            mmask = layers.Lambda(lambda x: (-1e+9) * (1.-K.cast(x, 'float32')))(mask)
            attn = layers.Add()([attn, mmask])
        attn = layers.Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = layers.Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class MultiHeadAttention():
    """
    Applies multihead attention to query, key and values when called.

    Attrs:
        d_model: Size of the output of the attention layer.
    """
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, dropout, mode=0):
        self.mode = mode
        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = layers.Dense(n_head*d_k, use_bias=False)
            self.ks_layer = layers.Dense(n_head*d_k, use_bias=False)
            self.vs_layer = layers.Dense(n_head*d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(layers.TimeDistributed(
                    layers.Dense(d_k, use_bias=False)))
                self.ks_layers.append(layers.TimeDistributed(
                    layers.Dense(d_k, use_bias=False)))
                self.vs_layers.append(layers.TimeDistributed(
                    layers.Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention()
        self.w_o = layers.TimeDistributed(layers.Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, s[2]//n_head])
                x = tf.transpose(x, [2, 0, 1, 3])
                # [n_head * batch_size, len_q, d_k]
                x = tf.reshape(x, [-1, s[1], s[2]//n_head])
                return x
            qs = layers.Lambda(reshape1)(qs)
            ks = layers.Lambda(reshape1)(ks)
            vs = layers.Lambda(reshape1)(vs)

            if mask is not None:
                mask = layers.Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)

            def reshape2(x):
                s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]])
                x = tf.transpose(x, [1, 2, 0, 3])
                # [batch_size, len_v, n_head * d_v]
                x = tf.reshape(x, [-1, s[1], n_head*d_v])
                return x
            head = layers.Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = []; attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head); attns.append(attn)
            head = layers.Concatenate()(heads) if n_head > 1 else heads[0]
            attn = layers.Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = layers.Dropout(self.dropout)(outputs)
        return outputs, attn


class SelfAttArgBiLSTM(ArgBiLSTM):
    """Bidirectional LSTM with a self-attention mechanism.

    Additional parameter:
        attention_size: The size of the output of the self-attention layer.
            It must be divisible by n_heads. Default 256.
        n_heads: The number of attention heads. Default 4.

    The self-attention is applied before the BiLSTM layer.
    The code is taken from
    https://github.com/Lsdefine/attention-is-all-you-need-keras
    """
    def __init__(self, params=None):
        super(SelfAttArgBiLSTM, self).__init__(params)
        # d_model in original code
        self.attention_size = self.params.get('attention_size', 256)
        self.n_heads = self.params.get('n_heads', 4)
        # TODO assert they are divisible

    def addPreAttentionLayer(self, merged_input):
        """Add attention mechanisms to the tensor merged_input.

        Args:
            merged_input: 3-dimensional Tensor, where the first
            dimension corresponds to the batch size, the second to the sequence
            timesteps and the last one to the concatenation of features.

        Retruns:
            3-dimensional Tensor of the same dimension as merged_input
        """
        attention_dropout = 0.1 # TODO put this as a parameter
        self_att_layer = MultiHeadAttention(
            self.n_heads, self.attention_size, dropout=attention_dropout)
        output, self_attention = self_att_layer(
            merged_input, merged_input, merged_input)
        import ipdb; ipdb.set_trace()
        # TODO extract self_attention somehow

        # TODO: Residual encoding - decide if to include or not.
        # norm_layer = LayerNormalization()
        # output = norm_layer(layers.Add()([merged_input, output]))

        # TODO Second step of the Transformer encoder, decide if to add
        # pos_ffn_layer = PositionwiseFeedForward(
        #     self.attention_size, self.attention_size*2, dropout=attention_dropout)
        # output = pos_ffn_layer(output)
        return output
