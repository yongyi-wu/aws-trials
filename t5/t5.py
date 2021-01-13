# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
T5 Model

@article{2020t5,
  author  = {Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu},
  title   = {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  journal = {Journal of Machine Learning Research},
  year    = {2020},
  volume  = {21},
  number  = {140},
  pages   = {1-67},
  url     = {http://jmlr.org/papers/v21/20-074.html}
}
"""

__all__ = []

import os
import functools
from typing import Tuple

import mxnet as mx
from mxnet import use_np
from mxnet import np, npx
from mxnet.gluon import HybridBlock, Parameter, nn
from ..attention_cell import (
    gen_self_attn_mask, gen_mem_attn_mask, MultiHeadAttentionCell, RelAttentionScoreCell
)
from ..base import get_model_zoo_home_dir, get_model_zoo_checksum_dir
from ..data import Vocab
from ..data.tokenizers import SentencepieceTokenizer
from ..layers import get_activation
from ..sequence_sampler import BaseStepDecoder
from ..utils.config import CfgNode as CN
from ..utils.misc import load_checksum_stats
from ..utils.registry import Registry


t5_cfg_reg = Registry('t5_cfg')


@t5_cfg_reg.register()
def google_t5_base(): 
    """Configuratino of T5 Base"""
    # model parameters
    cfg = CN()
    cfg.MODEL = CN()
    cfg.MODEL.vocab_size = 32128
    cfg.MODEL.d_model = 512
    cfg.MODEL.d_kv = 64
    cfg.MODEL.d_ff = 2048
    cfg.MODEL.num_layers = 6
    cfg.MODEL.num_heads = 8
    cfg.MODEL.dropout_prob = 0.1
    cfg.MODEL.layer_norm_eps = 1E-6
    cfg.MODEL.activation = 'relu'
    cfg.MODEL.dtype = 'float32'
    cfg.MODEL.layout = 'NT'
    # initializer parameters
    cfg.INITIALIZER = CN()
    # TODO(yongyi-wu)

    cfg.freeze()
    return cfg


PRETRAINED_URL = {
    'google_t5_small': {

    }, 
    'google_t5_base': {

    }, 
    'google_t5_large': {

    }, 
    'google_t5_3B': {

    }, 
    'google_t5_11B': {

    }
}

# FILE_STATS = load_checksum_stats(os.path.join(get_model_zoo_checksum_dir(), 't5.txt'))


@use_np
class T5LayerNorm(HybridBlock): 
    """
    Layer normalization without bias and mean substraction
    """
    def __init__(self, d_model, epsilon, dtype): 
        super().__init__()
        self.gemma = Parameter('layernorm_weight', shape=d_model, init='ones', dtype=dtype)
        self.variance_epsilon = epsilon

    def forward(self, x): 
        var = np.power(x.astype('float32'), 2).mean(-1, keepdims=True)
        x = x * np.reciprocal(np.sqrt(var + self.variance_epsilon))
        if self.gemma.dtype == 'float16': 
            x = x.astype('float16')
        return self.gemma * x


@use_np
class T5FeedForward(HybridBlock): 
    """
    Feed forward network supporting relu and gated-gelu
    """
    def __init__(
        self, 
        d_model, 
        d_ff, 
        dropout_prob, 
        epsilon, 
        activation, 
        weight_initializer, 
        dtype
    ): 
        super().__init__()
        self._activation = activation

        if self._activation == 'relu': 
            self.wi = nn.Dense(
                units=d_ff, 
                in_units=d_model, 
                use_bias=False, 
                weight_initializer=weight_initializer, 
                dtype=dtype    
            )
            self.relu = get_activation('relu')
            self.wo = nn.Dense(
                units=d_model, 
                in_units=d_ff, 
                use_bias=False, 
                weight_initializer=weight_initializer, 
                dtype=dtype
            )
        elif self._activation == 'gated-gelu': 
            self.wi_0 = nn.Dense(
                units=d_ff, 
                in_units=d_model, 
                use_bias=False, 
                weight_initializer=weight_initializer, 
                dtype=dtype
            )
            self.wi_1 = nn.Dense(
                units=d_ff, 
                in_units=d_model, 
                use_bias=False, 
                weight_initializer=weight_initializer, 
                dtype=dtype
            )
            self.wo = nn.Dense(
                units=d_model, 
                in_units=d_ff, 
                use_bias=False, 
                weight_initializer=weight_initializer, 
                dtype=dtype
            )
            self.gelu = get_activation('gelu')
        else: 
            raise ValueError(
                '{} unsupported. Select `relu` or `gated-gelu`'.format(activation)
            )
        self.layer_norm = T5LayerNorm(d_model, layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x): 
        out = self.layer_norm(x)
        if self._activation == 'relu': 
            out = self.wi(out)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.wo(out)
        elif self._activation == 'gated-gelu': 
            out_gelu = self.gelu(self.wi_0(out))
            out_linear = self.wi_1(out)
            out = out_gelu * out_linear
            out = self.dropout(out)
            out = self.wo(out)
        else: 
            raise ValueError(
                '{} unsupported. Select `relu` or `gated-gelu`'.format(activation)
            )
        out = x + self.dropout(out)
        return out


def _assert_decoder(fn): 
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs): 
        assert self._is_decoder, \
            '{}() is available for decoder only'.format(fn.__name__)
        return fn(self *args, **kwargs)
    return wrapper


@use_np
class T5Block(HybridBlock): 
    def __init__(
        self, 
        d_model, 
        d_kv, 
        d_ff, 
        is_decoder, 
        num_heads, 
        dropout_prob, 
        layer_norm_eps, 
        activation, 
        weight_initializer, 
        dtype, 
        layout
    ): 
        super().__init__()
        self._d_model = d_model
        self._d_kv = d_kv
        self._d_ff = d_ff
        self._is_decoder = is_decoder
        self._num_heads = num_heads
        self._inner_dim = self._num_heads * self._d_kv
        self._layout = layout
        self._dtype = dtype
        assert self._layout in ['TN', 'NT'], \
            'Invalid layout: {}. Only "TN" and "NT" are accepted.'.format(layout)

        self.self_attn_layer_norm = T5LayerNorm(
            d_model=self._d_model, 
            epsilon=layer_norm_eps, 
            dtype=self._dtype
        )
        self.self_attn_qkv = nn.Dense(
            units=3 * self._inner_dim, 
            in_units=self._d_model
            use_bias=False, 
            flatten=False, 
            weight_initializer=weight_initializer, 
            dtype=self._dtype
        )
        self.self_attn = MultiHeadAttentionCell(
            query_units=self._inner_dim, 
            num_heads=self._num_heads, 
            attention_dropout=dropout_prob, 
            scaled=False, 
            normalized=False, 
            dtype=self._dtype, 
            layout='NTK' if self._layout == 'NT' else 'TNK', 
            use_einsum=False
        )
        self.self_attn_proj = nn.Dense(
            units=self._d_model, 
            in_units=self._inner_dim, 
            use_bias=False, 
            flatten=False, 
            weight_initializer=weight_initializer, 
            dtype=self._dtype
        )
        if self._is_decoder: 
            self.cross_attn_layer_norm = T5LayerNorm(
                d_model=self._d_model, 
                epsilon=layer_norm_eps, 
                dtype=self._dtype
            )
            self.cross_attn_q = nn.Dense(
                units=self._inner_dim, 
                in_units=self._d_model, 
                use_bias=False, 
                flatten=False, 
                weight_initializer=weight_initializer, 
                dtype=self._dtype
            )
            self.cross_attn_k = nn.Dense(
                units=self._inner_dim, 
                in_units=self._d_model, 
                use_bias=False, 
                flatten=False, 
                weight_initializer=weight_initializer, 
                dtype=self._dtype
            )
            self.cross_attn_v = nn.Dense(
                units=self._inner_dim, 
                in_units=self._d_model, 
                use_bias=False, 
                flatten=False, 
                weight_initializer=weight_initializer, 
                dtype=self._dtype
            )
            self.cross_attn = MultiHeadAttentionCell(
                query_units=self._inner_dim, 
                num_heads=self._num_heads, 
                attention_dropout=dropout_prob, 
                scaled=False, 
                normalized=False, 
                dtype=self._dtype, 
                layout='NTK' if self._layout == 'NT' else 'TNK', 
                use_einsum=False
            )
            self.cross_attn_proj = nn.Dense(
                units=self._d_model, 
                in_units=self._inner_dim, 
                use_bias=False, 
                flatten=False, 
                weight_initializer=weight_initializer, 
                dtype=self._dtype
            )
        self.ffn = T5FeedForward(
            d_model=self._d_model, 
            d_ff=self._d_ff, 
            dropout_prob=dropout_prob, 
            epsilon=layer_norm_eps, 
            activation=activation, 
            weight_initializer=weight_initializer, 
            dtype=self._dtype
        )
        self.dropout = nn.Dropout(dropout_prob)

    @property
    def layout(self): 
        return self._layout

    @_assert_decoder
    @property
    def state_batch_axis(self): 
        if self.layout == 'NT': 
            return 0, 0
        else: 
            return 1, 1

    @_assert_decoder
    def _init_states(self, batch_size, ctx, dtype='float32'): 
        if self.layout == 'NT': 
            shape = (batch_size, 0, self._num_heads, self._d_kv)
        else: 
            shape = ((0, batch_size, self._num_heads, self._d_kv))
        init_key = np.zeros(shape, ctx=ctx, dtype=dtype)
        init_value = np.zeros(shape, ctx=ctx, dtype=dtype)
        return init_key, init_value

    @_assert_decoder
    def incremental_decode(
        self, 
        hidden_states, 
        past_key_value, 
        mem_states, 
        mem_valid_length, 
        mem_attn_mask=None
    ): 
        pass

    def forward(
        self, 
        hidden_states, 
        self_attn_mask, 
        position_embeddings, 
        mem_states=None, 
        mem_attn_mask=None, 
        mem_position_embeddings=None
    ): 
        """
        hidden_states: 
            - layout = 'NT'
                Shape (B, L_seq, d_model)
            - layout = 'TN'
                Shape (L_seq, B, d_model)
        """
        # NT -> NTK: (B, L_seq, inner_dim) -> (B, L_seq, num_heads, n_kv)
        # TN -> TNK: (L_seq, B, inner_dim) -> (L_seq, B, num_heads, n_kv)
        def shape(x):
            return x.reshape(-2, -2, self._num_heads, -1)

        # 1. self-attention
        self_query, self_key, self_value = np.split(
            self.self_attn_qkv(self.self_attn_layer_norm(hidden_states)), 
            indices_or_sections=3, 
            axis=-1
        )
        out, [_, self_attn_weights] = self.self_attn(
            shape(self_query), 
            shape(self_key), 
            shape(self_value), 
            mask=self_attn_mask, 
            edge_scores=position_embeddings
        )
        out = self.dropout(self.self_attn_proj(out))
        out = hidden_states + out

        # 2. cross-attention, if needed
        if self._is_decoder: 
            hidden_states = out
            cross_query, cross_key, cross_value = (
                self.cross_attn_q(self.cross_attn_layer_norm(out)), 
                self.cross_attn_k(mem_states), 
                self.cross_attn_v(mem_states)
            )
            out, [_, cross_attn_weights] = self.cross_attn(
                shape(cross_query), 
                shape(cross_key), 
                shape(cross_value), 
                mask=mem_attn_mask, 
                edge_scores=mem_position_embeddings
            )
            out = self.dropout(self.cross_attn_proj(out))
            out = hidden_states + out

        # 3. feed forward
        out = self.ffn(out)

        return out


@use_np
class T5Stack(HybridBlock): 
    def __init__(
        self, 
        d_model, 
        d_kv, 
        d_ff, 
        num_layers, 
        num_heads, 
        is_decoder, 
        dropout_prob, 
        layer_norm_eps, 
        activation, 
        weight_initializer, 
        dtype, 
        layout
    ): 
        super().__init__()
        self._is_decoder = is_decoder
        self._d_model = d_model
        self._d_kv = d_kv
        self._d_ff = d_ff
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._inner_dim = num_heads * d_kv
        self._dtype = dtype
        self._layout = layout
        assert self._layout in ['TN', 'NT'], \
            'Invalid layout: {}. Only "TN" and "NT" are accepted.'.format(layout)

        self.encoder_position_embedding_layer = RelAttentionScoreCell(
            num_heads=self._num_heads, 
            method='t5', 
            bidirectional=(not self._is_decoder), 
            dtype=self._dtype, 
            layout=self._layout
        )
        if self._is_decoder: 
            self.decoder_position_embedding_layer = RelAttentionScoreCell(
                num_heads=self._num_heads, 
                method='t5', 
                bidirectional=(not self._is_decoder), 
                dtype=self._dtype, 
                layout=self._layout
            )
        self.layers = nn.HybridSequential()
        for _ in range(num_layers): 
            self.layers.add(
                T5Block(
                    d_model=self._d_model, 
                    d_kv=self._d_kv, 
                    d_ff=self.d_ff, 
                    is_decoder=self._is_decoder, 
                    num_heads=self._num_heads, 
                    dropout_prob=dropout_prob, 
                    layer_norm_eps=layer_norm_eps, 
                    activation=activation, 
                    weight_initializer=weight_initializer, 
                    dtype=self._dtype
                    layout=self._layout
                )
            )
        self.final_layer_norm = T5LayerNorm(
            d_model=self._d_model, 
            epsilon=layer_norm_eps, 
            dtype=self._dtype
        )
        self.dropout = nn.Dropout(dropout_prob)

    @property
    def layout(self): 
        return self._layout

    @_assert_decoder
    @property
    def state_batch_axis(self): 
        return list(layer.state_batch_axis for layer in self.layers)

    @_assert_decoder
    def init_states(self, batch_size, ctx, dtype='float32'): 
        return list(layer.init_states(batch_size, ctx, dtype) for layer in self.layers)

    @_assert_decoder
    def incremental_decode(
        hidden_states, 
        past_key_value, 
        mem_states, 
        mem_valid_length
    ): 
        pass

    def _get_relative_position(self, hidden_states, mem_states, past_key_value): 
        # relative_position = mem_i - query_j
        # NT: (B, L_seq, inner_dim); TN: (L_seq, B, inner_dim)
        index = 1 if self.layout == 'NT' else 0
        query_lenth = hidden_states.shape[index]
        if past_key_value is not None: 
            # for incremental decoding only, where past key and value are of shape
            # NT(NTK): (B, L_seq, num_heads, n_kv); TN(TNK): (L_seq, B, num_heads, n_kv)
            query_lenth += past_key_value[0].shape[index]
        key_length = query_lenth if mem_states is None else mem_states.shape[index]
        query_position = np.arange(query_length, dtype=np.int64)[:, None]
        mem_position = np.arange(key_length, dtype=np.int64)[None, :]
        relative_position = mem_position - query_position
        # Shape (query_length, key_length)
        return relative_position

    def forward(
        self, 
        hidden_states, 
        valid_length, 
        mem_states=None, 
        mem_valid_length=None
    ): 
        # 1. relative position embeddings and attention masks
        position_embeddings = self.encoder_position_embedding_layer(
            self._get_relative_position(hidden_states, None, None)
        )
        if self._is_decoder: 
            mem_position_embeddings = self.position_embedding_layer(
                self._get_relative_position(hidden_states, mem_states, None)
            )
        else: 
            mem_position_embeddings = None
        self_attn_mask = gen_self_attn_mask(
            hidden_states, 
            valid_length, 
            dtype=self._dtype, 
            attn_type='causal' if self._is_decoder else 'full', 
            layout=self.layout
        )
        if self._is_decoder: 
            mem_attn_mask = gen_mem_attn_mask(
                mem_states, 
                mem_valid_length, 
                hidden_states, 
                valid_length, 
                dtype=self._dtype, 
                layout=self.layout
            )
        else: 
            mem_attn_mask = None

        # 2. encoder (or decoder) blocks and other layers
        # For the encoder, mem_states, mem_valid_length, mem_attn_mask, 
        # and mem_position_embeddings are None
        hidden_states = self.dropout(data)
        for layer in self.layers: 
            hidden_states = layer(
                hidden_states, 
                self_attn_mask=self_attn_mask, 
                position_embeddings=position_embeddings, 
                mem_states=mem_states, 
                mem_attn_mask=mem_attn_mask, 
                mem_position_embeddings=mem_position_embeddings
            )
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


@use_np
class T5Model(HybridBlock): 
    def __init__(
        vocab_size, 
        d_model, 
        d_kv, 
        d_ff, 
        num_layers, 
        num_heads, 
        dropout_prob, 
        layer_norm_eps, 
        activation, 
        weight_initializer, 
        dtype, 
        layout
    ): 
        assert vocab_size > 0, 'Cannot set "vocab_size" to negative numbers. ' \
            'Are you creating the model with the config from T5Model.get_cfg()? ' \
            'If that is the case, you will need to set the cfg.MODEL.vocab_size ' \
            'manually before passing to T5Model.from_cfg().'
        self._vocab_size = vocab_size
        self._layout = layout
        assert self._layout in ['TN', 'NT'], \
            'Invalid layout: {}. Only "TN" and "NT" are accepted.'.format(layout)

        # input embedding weights are between across encoder and decoder
        self.input_embedding_layer = nn.Embedding(
            input_dim=self._vocab_size, 
            output_dim=self.d_model, 
            dtype=self._dtype
        )
        self.encoder = T5Stack(
            d_model=d_model, 
            d_kv=d_kv, 
            d_ff=d_ff, 
            num_layers=num_layers, 
            num_heads=num_heads, 
            is_decoder=False, 
            dropout_prob=dropout_prob, 
            layer_norm_eps=layer_norm_eps, 
            activation=activation,  
            weight_initializer=weight_initializer, 
            dtype=dtype, 
            layout=layout
        )
        self.decoder = T5Stack(
            d_model=d_model, 
            d_kv=d_kv, 
            d_ff=d_ff, 
            num_layers=num_layers, 
            num_heads=num_heads, 
            is_decoder=True, 
            dropout_prob=dropout_prob, 
            layer_norm_eps=layer_norm_eps, 
            activation=activation,  
            weight_initializer=weight_initializer, 
            dtype=dtype, 
            layout=layout
        )
    
    @property
    def layout(self): 
        return self._layout

    @property
    def vocab_size(self): 
        return self._vocab_size

    def forward(self, src_data, src_valid_length, tgt_data, tgt_valid_length): 
        src_hidden_states = self.input_embedding_layer(src_data)
        enc_out = self.encoder(
            hidden_states=src_hidden_states, 
            valid_length=src_valid_length
        )
        tgt_hidden_states = self.input_embedding_layer(tgt_data)
        dec_out = self.decoder(
            hidden_states=tgt_hidden_states, 
            valid_length=tgt_valid_length, 
            mem_states=enc_out, 
            mem_valid_length=src_valid_length
        )
        return dec_out

    @classmethod
    def get_cfg(cls, key=None): 
        if key is None: 
            return t5_base()
        else: 
            return t5_cfg_reg.create(key)

    @classmethod
    def from_cfg(cls, cfg, dtype=None): 
        pass


class T5NMTInference(HybridBlock, BaseStepDecoder): 
    pass 


def list_pretrained_t5(): 
    return sorted(list(PRETRAINED_URL.keys()))


def _build_t5_tokenizer(vocab_path, do_lower, extra_ids): 
    # manually add additional special tokens corresponding to noise span sentinels
    # with <extra_id_0> be the last token in the new vocabulary
    extra_token = '<extra_id_{}>'
    additional_special_tokens = {
        'extra{}_token'.format(i): extra_token.format(i) for i in range(extra_ids - 1, -1, -1)
    }
    tokenizer = SentencepieceTokenizer(
        model_path=vocab_path,
        lowercase=do_lower, 
        **additional_special_tokens
    )
    # sanity check: every additional token has been inserted with correct order
    inserted_special_tokens = list(extra_token.format(i) for i in range(extra_ids - 1, -1, -1))
    assert list(
        tokenizer._vocab.to_tokens(i) for i in range(len(tokenizer._sp_model), len(tokenizer._vocab))
    ) == inserted_special_tokens, 'Some <extra_id> tokens are not properly inserted'
    return tokenizer


def get_pretrained_t5(model_name: str = 't5-base', 
                      root: str = get_model_zoo_home_dir(), 
                      load_backbone: bool = True, 
                      load_lm: bool = False, 
                      extra_ids: int = 100) \
    -> Tuple[CN, SentencepieceTokenizer, str, str]: 
    """
    TBD
    """
    assert model_name in PRETRAINED_URL, '{} is not found. All available are {}'.format(
        model_name, list_pretrained_t5())
    cfg_path = PRETRAINED_URL[model_name]['cfg']
    if isinstance(cfg_path, CN):
        cfg = cfg_path
    else:
        cfg = None

    vocab_path = PRETRAINED_URL[model_name]['vocab']
    params_path = PRETRAINED_URL[model_name]['params']

    do_lower = True if 'lowercase' in PRETRAINED_URL[model_name]\
                       and PRETRAINED_URL[model_name]['lowercase'] else False
    tokenizer = _build_t5_tokenizer(vocab_path, do_lower, extra_ids)
    if cfg is None: 
        cfg = T5Model.get_cfg().clone_merge(local_paths['cfg'])
    return cfg, tokenizer,
