"""Adapted from https://github.com/google/flax/blob/main/examples/nlp_seq/models.py"""

from typing import TYPE_CHECKING, Any, Optional

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from c_vpr.transformer_utils import (
    CrossTransformerLayer,
    TransformerConfig,
    TransformerLayer,
)

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class CoTTransformerConfig:
    input_transformer_config: TransformerConfig
    cross_transformer_config: TransformerConfig
    cot_seq_length: int
    cot_vocab_size: int
    output_vocab_size: int
    max_num_hops: Optional[int] = None
    use_bias: bool = False
    dtype: Any = jnp.float32


class CoTTransformer(nn.Module):
    config: CoTTransformerConfig

    def setup(self) -> None:
        # Inputs layers.
        self.inputs_tok_embed = nn.Embed(
            num_embeddings=self.config.input_transformer_config.vocab_size,
            features=self.config.input_transformer_config.emb_dim,
            name="inputs_tok_embed",
        )
        self.inputs_pos_embed = nn.Embed(
            num_embeddings=self.config.input_transformer_config.max_len,
            features=self.config.input_transformer_config.emb_dim,
            name="pos_embed",
        )
        if self.config.max_num_hops is not None:
            self.num_hops_embed = nn.Embed(
                num_embeddings=self.config.max_num_hops,
                features=self.config.input_transformer_config.emb_dim,
                name="num_hops_embed",
            )
        self.inputs_dropout = nn.Dropout(rate=self.config.input_transformer_config.dropout_rate)
        self.input_transformer_layers = [
            TransformerLayer(self.config.input_transformer_config)
            for _ in range(self.config.input_transformer_config.num_layers)
        ]

        # CoT layers.
        self.cot_tok_embed = nn.Embed(
            num_embeddings=self.config.cot_vocab_size + 1,  # +1 for the start token embedding
            features=self.config.cross_transformer_config.emb_dim,
            name="cot_tok_embed",
        )
        self.cot_pos_embed = nn.Embed(
            num_embeddings=self.config.cot_seq_length + 1,  # +1 for the start token
            features=self.config.cross_transformer_config.emb_dim,
            name="cot_pos_embed",
        )
        self.cot_dropout = nn.Dropout(rate=self.config.cross_transformer_config.dropout_rate)
        self.cross_transformer_layers = [
            CrossTransformerLayer(self.config.cross_transformer_config)
            for _ in range(self.config.cross_transformer_config.num_layers)
        ]

        # Head layers.
        self.layer_norm = nn.LayerNorm(dtype=self.config.dtype, use_bias=self.config.use_bias)
        self.cot_linear_head = nn.Dense(
            self.config.cot_vocab_size, self.config.use_bias, self.config.dtype
        )
        self.output_linear_head = nn.Dense(
            self.config.output_vocab_size, self.config.use_bias, self.config.dtype
        )

    def encode_inputs(
        self,
        *,
        inputs: chex.Array,
        deterministic: bool,
        num_hops: Optional[chex.Array] = None,
        pad_mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        input_transformer_config = self.config.input_transformer_config

        # Input embedding block.
        assert inputs.ndim == 2  # (B, T)
        tok_embed = self.inputs_tok_embed(inputs.astype("int32"))
        pos_embed = self.inputs_pos_embed(jnp.arange(input_transformer_config.max_len))
        inputs_embeddings = tok_embed + pos_embed
        # Concatenate the number of hops to the input embeddings.
        if num_hops is not None:
            assert num_hops.ndim == 1  # (B,)
            assert self.config.max_num_hops is not None
            num_hops = num_hops.astype("int32") - 1  # num hops is 1-indexed
            num_hops_embedding = self.num_hops_embed(num_hops[:, None])
            inputs_embeddings = jnp.concatenate([inputs_embeddings, num_hops_embedding], axis=-2)
            if pad_mask is not None:
                pad_mask = jnp.concatenate([pad_mask, jnp.ones_like(pad_mask[:, 0:1])], axis=-1)
        inputs_embeddings = self.inputs_dropout(inputs_embeddings, deterministic=deterministic)

        # Input encoder block.
        for _ in range(input_transformer_config.num_repeat_model):
            for layer in self.input_transformer_layers:
                inputs_embeddings = layer(
                    embeddings=inputs_embeddings,
                    deterministic=deterministic,
                    pad_mask=pad_mask,
                )
        return inputs_embeddings

    @nn.compact
    def __call__(
        self,
        *,
        inputs: chex.Array,
        deterministic: bool,
        num_hops: Optional[chex.Array] = None,
        pad_mask: Optional[chex.Array] = None,
        cot_sampling: bool = False,
        cot_key: Optional[chex.PRNGKey] = None,
    ) -> tuple[chex.Array, tuple[chex.Array, chex.Array]]:
        """Applies Transformer model on the inputs.

        Args:
            inputs: input data as tokens.
            deterministic: if false dropout is applied otherwise it is not.
            num_hops: number of hops associated to the inputs.
            pad_mask: mask to apply on the inputs to avoid attending to padding tokens.
            cot_sampling: if true, sample tokens in the forward pass. Otherwise, use argmax.
            cot_key: random key to sample tokens in the forward pass.

        Returns:
            chain of thoughts tokens of shape (B, T_C) representing a sequence of tokens,
            logits of shape (B, T_C, V) representing a sequence of logits.

        Remark:
            could also output a chain of shape (B, T_C, H) representing a sequence of cot
                embeddings.
        """
        config = self.config
        inputs_embeddings = self.encode_inputs(
            inputs=inputs, deterministic=deterministic, num_hops=num_hops, pad_mask=pad_mask
        )

        # CoT autoregressive block.
        cot_tokens = jnp.zeros(
            (inputs.shape[0], config.cot_seq_length + 1),  # +1 for the start token
            dtype=jnp.int32,
        )
        cot_tokens = cot_tokens.at[:, 0].set(config.cot_vocab_size)  # set the start token
        assert cot_key is not None or not cot_sampling
        # TODO: speed up for loop with jax.lax.scan
        cot_logits_list = []
        for i in range(config.cot_seq_length):
            all_cot_logits = self.generate_logits(
                cot_tokens=cot_tokens,
                inputs_embeddings=inputs_embeddings,
                deterministic=deterministic,
                inputs_pad_mask=pad_mask,
            )
            cot_logits = all_cot_logits[:, i, :]
            cot_logits_list.append(cot_logits)
            if cot_sampling:
                sample_key, cot_key = jax.random.split(cot_key)
                new_token = jax.random.categorical(sample_key, cot_logits)
            else:
                new_token = jnp.argmax(cot_logits, axis=-1)
            cot_tokens = cot_tokens.at[:, i + 1].set(new_token)
        output_logits = self.generate_logits(
            cot_tokens=cot_tokens,
            inputs_embeddings=inputs_embeddings,
            deterministic=deterministic,
            inputs_pad_mask=pad_mask,
            final_forward=True,
        )[:, -1, :]
        cot_logits = jnp.stack(cot_logits_list, axis=1)

        cot_tokens = cot_tokens[:, 1:]  # remove the start token
        return output_logits, (cot_tokens, cot_logits)

    def generate_logits(
        self,
        *,
        cot_tokens: chex.Array,
        inputs_embeddings: chex.Array,
        deterministic: bool,
        inputs_pad_mask: Optional[chex.Array] = None,
        final_forward: bool = False,
    ) -> chex.Array:
        """Applies Transformer model on a partially generated chain of thoughts.

        Args:
          cot_tokens: tokens from the chain of thoughts.
          inputs_embeddings: input embeddings from the encoder.
          deterministic: if false dropout is applied otherwise it is not.
          inputs_pad_mask: mask to apply on the inputs embeddings to avoid attending to padding
            tokens.

        Returns:
          logits of shape (B, T_C, V) representing a sequence of logits.
        """
        assert inputs_embeddings.ndim == 3  # (B, T, H)

        # CoT embedding block.
        cot_tok_embed = self.cot_tok_embed(cot_tokens)
        cot_pos_embed = self.cot_pos_embed(jnp.arange(self.config.cot_seq_length + 1))
        cot_embeddings = cot_tok_embed + cot_pos_embed
        cot_embeddings = self.cot_dropout(cot_embeddings, deterministic=deterministic)

        # CoT cross transformer block.
        bs, t, _ = cot_embeddings.shape
        causal_mask = jnp.tril(jnp.ones((bs, 1, t, t), bool))  # TODO: check if this is correct
        for _ in range(self.config.cross_transformer_config.num_repeat_model):
            for layer in self.cross_transformer_layers:
                cot_embeddings = layer(
                    self_embeddings=cot_embeddings,
                    cross_embeddings=inputs_embeddings,
                    deterministic=deterministic,
                    self_pad_mask=causal_mask,
                    cross_pad_mask=inputs_pad_mask,
                )
        cot_embeddings = self.layer_norm(cot_embeddings)
        if final_forward:
            logits = self.output_linear_head(cot_embeddings)
        else:
            logits = self.cot_linear_head(cot_embeddings)
        return logits


if __name__ == "__main__":
    seq_length = 10
    cot_seq_length = 5
    cot_vocab_size = 10
    max_num_hops = 5

    cot_transformer_config = CoTTransformerConfig(
        input_transformer_config=TransformerConfig(
            vocab_size=seq_length,
            output_vocab_size=None,
            emb_dim=384,
            num_heads=6,
            num_layers=1,
            num_repeat_model=1,
            mlp_dim_factor=4,
            max_len=seq_length,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
        ),
        cross_transformer_config=TransformerConfig(
            vocab_size=None,
            output_vocab_size=None,
            emb_dim=384,
            num_heads=6,
            num_layers=1,
            num_repeat_model=1,
            mlp_dim_factor=4,
            max_len=None,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
        ),
        cot_seq_length=cot_seq_length,
        cot_vocab_size=cot_vocab_size,
        output_vocab_size=seq_length,
        max_num_hops=max_num_hops,
    )
    model = CoTTransformer(cot_transformer_config)
    key = jax.random.PRNGKey(0)
    example = jax.random.randint(key, (2, seq_length), minval=0, maxval=seq_length)
    num_hops = jnp.array([1, max_num_hops], int)
    params = model.init(key, inputs=example, deterministic=True, num_hops=num_hops)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print("Number of parameters: {:,}".format(num_params))
    apply_fn = jax.jit(model.apply, static_argnames="deterministic")
    output_logits, (cot_tokens, cot_logits) = apply_fn(
        params,
        inputs=example,
        deterministic=False,
        num_hops=num_hops,
        rngs={"dropout": key},
    )
