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
class CoTModuleConfig:
    input_transformer_config: TransformerConfig
    cot_cross_transformer_config: TransformerConfig
    cot_seq_length: int
    cot_vocab_size: int
    max_num_hops: Optional[int] = None
    use_bias: bool = False
    dtype: Any = jnp.float32


@dataclass
class EncoderConfig:
    cot_transformer_config: Optional[TransformerConfig]
    cot_seq_length: Optional[int]
    cot_vocab_size: Optional[int]
    input_cross_transformer_config: TransformerConfig
    classification_mode: str
    max_num_hops: Optional[int] = None
    use_bias: bool = False
    dtype: Any = jnp.float32


class CoTModule(nn.Module):
    config: CoTModuleConfig

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
            features=self.config.cot_cross_transformer_config.emb_dim,
            name="cot_tok_embed",
        )
        self.cot_pos_embed = nn.Embed(
            num_embeddings=self.config.cot_seq_length + 1,  # +1 for the start token
            features=self.config.cot_cross_transformer_config.emb_dim,
            name="cot_pos_embed",
        )
        self.cot_dropout = nn.Dropout(rate=self.config.cot_cross_transformer_config.dropout_rate)
        self.cot_cross_transformer_layers = [
            CrossTransformerLayer(self.config.cot_cross_transformer_config)
            for _ in range(self.config.cot_cross_transformer_config.num_layers)
        ]

        # Head layers.
        self.layer_norm = nn.LayerNorm(dtype=self.config.dtype, use_bias=self.config.use_bias)
        self.linear_head = nn.Dense(
            self.config.cot_vocab_size, self.config.use_bias, self.config.dtype
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
    ) -> tuple[chex.Array, chex.Array]:
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
        logits_list = []
        # TODO: speed up for loop with jax.lax.scan
        for i in range(config.cot_seq_length):
            all_logits = self.generate_logits(
                cot_tokens=cot_tokens,
                inputs_embeddings=inputs_embeddings,
                deterministic=deterministic,
                inputs_pad_mask=pad_mask,
            )
            logits = all_logits[:, i, :]
            logits_list.append(logits)
            if cot_sampling:
                sample_key, cot_key = jax.random.split(cot_key)
                new_token = jax.random.categorical(sample_key, logits)
            else:
                new_token = jnp.argmax(logits, axis=-1)
            cot_tokens = cot_tokens.at[:, i + 1].set(new_token)
        logits = jnp.stack(logits_list, axis=1)
        cot_tokens = cot_tokens[:, 1:]  # remove the start token
        return cot_tokens, logits

    def generate_logits(
        self,
        *,
        cot_tokens: chex.Array,
        inputs_embeddings: chex.Array,
        deterministic: bool,
        inputs_pad_mask: Optional[chex.Array] = None,
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
        for _ in range(self.config.cot_cross_transformer_config.num_repeat_model):
            for layer in self.cot_cross_transformer_layers:
                cot_embeddings = layer(
                    self_embeddings=cot_embeddings,
                    cross_embeddings=inputs_embeddings,
                    deterministic=deterministic,
                    self_pad_mask=causal_mask,
                    cross_pad_mask=inputs_pad_mask,
                )
        cot_embeddings = self.layer_norm(cot_embeddings)
        logits = self.linear_head(cot_embeddings)
        return logits


class Encoder(nn.Module):
    config: EncoderConfig

    def setup(self) -> None:
        if self.config.cot_transformer_config is not None:
            self.cot_transformer_layers = [
                TransformerLayer(self.config.cot_transformer_config)
                for _ in range(self.config.cot_transformer_config.num_layers)
            ]
        if self.config.max_num_hops is not None:
            self.num_hops_embed = nn.Embed(
                num_embeddings=self.config.max_num_hops,
                features=self.config.input_cross_transformer_config.emb_dim,
                name="num_hops_embed",
            )
        self.input_cross_transformer_layers = [
            CrossTransformerLayer(self.config.input_cross_transformer_config)
            for _ in range(self.config.input_cross_transformer_config.num_layers)
        ]

    @nn.compact
    def __call__(  # noqa: CCR001
        self,
        *,
        inputs: chex.Array,
        cot_tokens: Optional[chex.Array],
        deterministic: bool,
        num_hops: Optional[chex.Array] = None,
        inputs_pad_mask: Optional[chex.Array] = None,
        cot_pad_mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        """Applies Transformer model on the inputs.

        Args:
            inputs: input data as tokens.
            cot_tokens: chain of thought tokens from the cot module.
            deterministic: if false dropout is applied otherwise it is not.
            num_hops: number of hops associated to the inputs.
            inputs_pad_mask: mask to apply on the inputs to avoid attending to padding tokens.
            cot_pad_mask: mask to apply on the cot to avoid attending to padding tokens.

        Returns:
            output of shape (B, V) representing a sequence of logits.
        """
        # CoT embedding block.
        cot_transformer_config = self.config.cot_transformer_config
        if cot_tokens is not None:
            assert cot_tokens.ndim == 2  # (B, T_C)
            assert cot_transformer_config is not None
            assert self.config.cot_seq_length is not None
            assert self.config.cot_vocab_size is not None
            cot_tok_embed = nn.Embed(
                num_embeddings=self.config.cot_vocab_size
                + 1,  # +1 for the start token embedding used for padding
                features=cot_transformer_config.emb_dim,
                name="cot_tok_embed",
            )(cot_tokens.astype("int32"))
            cot_pos_embed = nn.Embed(
                num_embeddings=self.config.cot_seq_length,
                features=cot_transformer_config.emb_dim,
                name="cot_pos_embed",
            )(jnp.arange(self.config.cot_seq_length))
            cot_embeddings = cot_tok_embed + cot_pos_embed
            cot_embeddings = nn.Dropout(rate=cot_transformer_config.dropout_rate)(
                cot_embeddings, deterministic=deterministic
            )

            # CoT encoder block.
            for _ in range(cot_transformer_config.num_repeat_model):
                for layer in self.cot_transformer_layers:
                    cot_embeddings = layer(
                        embeddings=cot_embeddings,
                        deterministic=deterministic,
                        pad_mask=cot_pad_mask,
                    )
        else:
            cot_embeddings = None

        # Input embedding block.
        input_cross_transformer_config = self.config.input_cross_transformer_config
        assert inputs.ndim == 2  # (B, T)
        tok_embed = nn.Embed(
            num_embeddings=input_cross_transformer_config.vocab_size,
            features=input_cross_transformer_config.emb_dim,
            name="tok_embed",
        )(inputs.astype("int32"))
        pos_embed = nn.Embed(
            num_embeddings=input_cross_transformer_config.max_len,
            features=input_cross_transformer_config.emb_dim,
            name="pos_embed",
        )(jnp.arange(input_cross_transformer_config.max_len))
        x = tok_embed + pos_embed
        # Concatenate the number of hops to the input embeddings.
        if num_hops is not None:
            assert num_hops.ndim == 1  # (B,)
            assert self.config.max_num_hops is not None
            num_hops = num_hops.astype("int32") - 1  # num hops is 1-indexed
            num_hops_embedding = self.num_hops_embed(num_hops[:, None])
            x = jnp.concatenate([x, num_hops_embedding], axis=-2)
            if inputs_pad_mask is not None:
                inputs_pad_mask = jnp.concatenate(
                    [inputs_pad_mask, jnp.ones_like(inputs_pad_mask[:, 0:1])], axis=-1
                )
        # Concatenate the cls token to the input embeddings.
        if self.config.classification_mode == "cls_token":
            cls_token = nn.Embed(
                num_embeddings=1,
                features=input_cross_transformer_config.emb_dim,
                name="cls_token",
            )(jnp.zeros((inputs.shape[0], 1), int))
            x = jnp.concatenate([cls_token, x], axis=-2)
            if inputs_pad_mask is not None:
                inputs_pad_mask = jnp.concatenate(
                    [jnp.ones_like(inputs_pad_mask[:, 0:1]), inputs_pad_mask], axis=-1
                )

        x = nn.Dropout(rate=input_cross_transformer_config.dropout_rate)(
            x, deterministic=deterministic
        )

        # Input cross transformer block.
        for _ in range(input_cross_transformer_config.num_repeat_model):
            for layer in self.input_cross_transformer_layers:
                x = layer(
                    self_embeddings=x,
                    cross_embeddings=cot_embeddings,
                    deterministic=deterministic,
                    self_pad_mask=inputs_pad_mask,
                    cross_pad_mask=cot_pad_mask,
                )

        if self.config.classification_mode == "cls_token":
            x = x[:, 0, :]
        else:
            assert self.config.classification_mode == "mean_embedding"
            x = nn.LayerNorm(dtype=self.config.dtype, use_bias=self.config.use_bias)(x)
            x = x.mean(axis=1)
        logits = nn.Dense(
            input_cross_transformer_config.output_vocab_size,
            self.config.use_bias,
            self.config.dtype,
        )(x)

        return logits


class AugmentedTransformer(nn.Module):
    """Transformer Model which produces intermediate CoT tokens."""

    cot_module_config: Optional[CoTModuleConfig]
    encoder_config: TransformerConfig

    def setup(self) -> None:
        self.cot_module = CoTModule(self.cot_module_config) if self.cot_module_config else None
        self.encoder = Encoder(self.encoder_config)

    def cot_module_generate_cot_logits(
        self,
        *,
        inputs: chex.Array,
        cot_tokens: chex.Array,
        deterministic: bool,
        num_hops: Optional[chex.Array] = None,
        inputs_pad_mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        assert self.cot_module is not None
        inputs_embeddings = self.cot_module.encode_inputs(
            inputs=inputs, deterministic=deterministic, num_hops=num_hops, pad_mask=inputs_pad_mask
        )
        cot_logits = self.cot_module.generate_logits(
            cot_tokens=cot_tokens,
            inputs_embeddings=inputs_embeddings,
            deterministic=deterministic,
            inputs_pad_mask=inputs_pad_mask,
        )
        # TODO: check if this is correct???
        cot_logits = cot_logits[:, :-1, :]  # do not need to train on the last token
        return cot_logits

    def cot_module_call(
        self,
        *,
        inputs: chex.Array,
        deterministic: bool,
        num_hops: Optional[chex.Array] = None,
        pad_mask: Optional[chex.Array] = None,
        cot_sampling: bool = False,
        cot_key: Optional[chex.PRNGKey] = None,
    ) -> tuple[chex.Array, chex.Array]:
        assert self.cot_module is not None
        return self.cot_module(  # type: ignore
            inputs=inputs,
            deterministic=deterministic,
            num_hops=num_hops,
            pad_mask=pad_mask,
            cot_sampling=cot_sampling,
            cot_key=cot_key,
        )

    def encoder_call(
        self,
        *,
        inputs: chex.Array,
        cot_tokens: Optional[chex.Array],
        deterministic: bool,
        num_hops: Optional[chex.Array] = None,
        inputs_pad_mask: Optional[chex.Array] = None,
        cot_pad_mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        return self.encoder(
            inputs=inputs,
            cot_tokens=cot_tokens,
            deterministic=deterministic,
            num_hops=num_hops,
            inputs_pad_mask=inputs_pad_mask,
            cot_pad_mask=cot_pad_mask,
        )

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
    ) -> tuple[chex.Array, tuple[Optional[chex.Array], Optional[chex.Array]]]:
        """Applies AugmentedTransformer model on the inputs.

        Args:
            inputs: input data
            deterministic: if false dropout is applied otherwise it is not.
            num_hops: number of hops associated to the inputs.
            pad_mask: mask to apply on the inputs to avoid attending to padding tokens.
            cot_sampling: if true, sample tokens in the CoT Module. Otherwise, use argmax.
            cot_key: random key to sample tokens in the CoT Module.

        Returns:
            output of an augmented transformer.
        """
        # CoT Module block.
        if self.cot_module is not None:
            cot_tokens, cot_logits = self.cot_module(
                inputs=inputs,
                deterministic=deterministic,
                num_hops=num_hops,
                pad_mask=pad_mask,
                cot_sampling=cot_sampling,
                cot_key=cot_key,
            )
        else:
            cot_tokens, cot_logits = None, None

        # Encoder block.
        logits = self.encoder(
            inputs=inputs,
            cot_tokens=cot_tokens,
            deterministic=deterministic,
            num_hops=num_hops,
            inputs_pad_mask=pad_mask,
            cot_pad_mask=None,
        )

        return logits, (cot_tokens, cot_logits)


if __name__ == "__main__":
    seq_length = 10
    cot_seq_length = 5
    cot_vocab_size = 5
    max_num_hops = 5

    cot_module_config = CoTModuleConfig(
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
        cot_cross_transformer_config=TransformerConfig(
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
        max_num_hops=max_num_hops,
    )
    # cot_module_config = None
    encoder_config = EncoderConfig(
        cot_transformer_config=TransformerConfig(
            vocab_size=cot_vocab_size,
            output_vocab_size=None,
            emb_dim=384,
            num_heads=6,
            num_layers=1,
            num_repeat_model=1,
            mlp_dim_factor=4,
            max_len=cot_seq_length,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
        ),
        cot_seq_length=cot_seq_length,
        cot_vocab_size=cot_vocab_size,
        input_cross_transformer_config=TransformerConfig(
            vocab_size=seq_length,
            output_vocab_size=seq_length,
            emb_dim=384,
            num_heads=6,
            num_layers=1,
            num_repeat_model=1,
            mlp_dim_factor=4,
            max_len=seq_length,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
        ),
        classification_mode="mean_embedding",
        max_num_hops=max_num_hops,
    )
    model = AugmentedTransformer(
        cot_module_config,
        encoder_config,
    )
    key = jax.random.PRNGKey(0)
    example = jax.random.randint(key, (2, seq_length), minval=0, maxval=seq_length)
    num_hops = jnp.array([1, max_num_hops], int)
    params = model.init(key, inputs=example, deterministic=True, num_hops=num_hops)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print("Number of parameters: {:,}".format(num_params))
    apply_fn = jax.jit(model.apply, static_argnames="deterministic")
    output = apply_fn(
        params,
        inputs=example,
        deterministic=False,
        num_hops=num_hops,
        rngs={"dropout": key},
    )
