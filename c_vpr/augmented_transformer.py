"""Adapted from https://github.com/google/flax/blob/main/examples/nlp_seq/models.py"""

from typing import TYPE_CHECKING, Any, Optional

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from c_vpr.transformer import MlpBlock, TransformerConfig, TransformerLayer

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class CoTModuleConfig:
    cross_transformer_config: TransformerConfig
    cot_seq_length: int
    cot_vocab_size: int
    use_bias: bool = False
    dtype: Any = jnp.float32


class Encoder(nn.Module):
    config: TransformerConfig

    def setup(self) -> None:
        self.transformer_layers = [
            TransformerLayer(self.config) for _ in range(self.config.num_layers)
        ]

    @nn.compact
    def __call__(
        self,
        *,
        inputs: chex.Array,
        deterministic: bool,
        pad_mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        """Applies Transformer model on the inputs.

        Args:
          inputs: input data
          deterministic: if false dropout is applied otherwise it is not.
          pad_mask: mask to apply on the inputs to avoid attending to padding tokens.

        Returns:
          output of shape (B, T, H) representing a sequence of embeddings.

        """
        assert inputs.ndim == 2  # (batch, len)

        config = self.config

        x = inputs.astype("int32")
        x = nn.Embed(
            num_embeddings=config.vocab_size,
            features=config.emb_dim,
            name="tok_embed",
        )(x)
        pos_embed = nn.Embed(
            num_embeddings=config.max_len,
            features=config.emb_dim,
            name="pos_embed",
        )(jnp.arange(config.max_len))
        x = x + pos_embed
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)

        for _ in range(config.num_repeat_model):
            for layer in self.transformer_layers:
                x = layer(x, deterministic=deterministic, pad_mask=pad_mask)

        return x


class CrossTransformerLayer(nn.Module):
    """Transformer layer that does the following: self-attention, cross-attention, and feed-forward.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerConfig

    @nn.compact
    def __call__(
        self,
        self_embeddings: chex.Array,
        cross_embeddings: Optional[chex.Array],
        deterministic: bool,
        self_pad_mask: Optional[chex.Array] = None,
        cross_pad_mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        """Applies TransformerLayer module.

        Args:
          self_embeddings: input data.
          cross_embeddings: input data to do cross-attention on.
          deterministic: if false dropout is applied otherwise it is not.
          self_pad_mask: mask to apply on the self inputs to avoid attending to padding tokens.
          cross_pad_mask: mask to apply on the cross inputs to avoid attending to padding tokens.

        Returns:
          output after cross transformer layer.
        """
        config = self.config
        assert self_embeddings.ndim == 3
        if cross_embeddings is not None:
            assert cross_embeddings.ndim == 3

        # Self-attention block.
        x = nn.LayerNorm(dtype=config.dtype, use_bias=config.use_bias)(self_embeddings)
        x = nn.MultiHeadDotProductAttention(
            num_heads=config.num_heads,
            dtype=config.dtype,
            dropout_rate=config.attention_dropout_rate,
            use_bias=config.use_bias,
        )(
            inputs_q=x,
            inputs_kv=x,
            mask=self_pad_mask,
            deterministic=deterministic,
        )
        residuals = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)
        self_embeddings = self_embeddings + residuals

        # Cross-attention block.
        if cross_embeddings is not None:
            inputs_q = nn.LayerNorm(dtype=config.dtype, use_bias=config.use_bias)(self_embeddings)
            inputs_kv = nn.LayerNorm(dtype=config.dtype, use_bias=config.use_bias)(cross_embeddings)
            x = nn.MultiHeadDotProductAttention(
                num_heads=config.num_heads,
                dtype=config.dtype,
                dropout_rate=config.attention_dropout_rate,
                use_bias=config.use_bias,
            )(
                inputs_q=inputs_q,
                inputs_kv=inputs_kv,
                mask=cross_pad_mask,
                deterministic=deterministic,
            )
            residuals = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)
            self_embeddings = self_embeddings + residuals

        # MLP block.
        x = nn.LayerNorm(dtype=config.dtype, use_bias=config.use_bias)(self_embeddings)
        residuals = MlpBlock(config=config)(x, deterministic=deterministic)
        return self_embeddings + residuals


class CoTModule(nn.Module):
    config: CoTModuleConfig

    def setup(self) -> None:
        self.cross_transformer_config = self.config.cross_transformer_config
        self.cot_tok_embed = nn.Embed(
            num_embeddings=self.config.cot_vocab_size + 1,  # +1 for the start token embedding
            features=self.cross_transformer_config.emb_dim,
            name="cot_tok_embed",
        )
        self.cot_pos_embed = nn.Embed(
            num_embeddings=self.config.cot_seq_length + 1,  # +1 for the start token
            features=self.cross_transformer_config.emb_dim,
            name="cot_pos_embed",
        )
        self.tok_dropout = nn.Dropout(rate=self.cross_transformer_config.dropout_rate)
        self.cross_transformer_layers = [
            CrossTransformerLayer(self.cross_transformer_config)
            for _ in range(self.cross_transformer_config.num_layers)
        ]
        self.linear_head = nn.Dense(
            self.config.cot_vocab_size, self.config.use_bias, self.config.dtype
        )

    def __call__(
        self,
        *,
        encoder_embeddings: chex.Array,
        deterministic: bool,
        cot_key: Optional[chex.PRNGKey] = None,
        pad_mask: Optional[chex.Array] = None,
    ) -> tuple[chex.Array, chex.Array]:
        """Applies Transformer model on the inputs.

        Args:
          encoder_embeddings: input embeddings from the encoder.
          deterministic: if false dropout is applied otherwise it is not.
          cot_key: random key to sample tokens during forward pass.
          pad_mask: mask to apply on the encoder embeddings to avoid attending to padding tokens.

        Returns:
          chain of thoughts tokens of shape (B, T_C) representing a sequence of tokens,
          logits of shape (B, T_C, V) representing a sequence of logits.

        Remark:
          could also output a chain of shape (B, T_C, H) representing a sequence of cot embeddings.
        """
        config = self.config
        cot_tokens = jnp.zeros(
            (
                encoder_embeddings.shape[0],
                config.cot_seq_length + 1,
            ),  # +1 for the start token
            dtype=jnp.int32,
        )
        cot_tokens = cot_tokens.at[:, 0].set(config.cot_vocab_size)  # set the start token

        logits_list = []
        for i in range(self.config.cot_seq_length):
            all_logits = self.generate_logits(
                cot_tokens=cot_tokens,
                encoder_embeddings=encoder_embeddings,
                deterministic=deterministic,
                pad_mask=pad_mask,
            )
            logits = all_logits[:, i, :]
            logits_list.append(logits)
            if deterministic:
                new_token = jnp.argmax(logits, axis=-1)
            else:
                sample_key, cot_key = jax.random.split(cot_key)
                new_token = jax.random.categorical(sample_key, logits)
            cot_tokens = cot_tokens.at[:, i + 1].set(new_token)
        logits = jnp.stack(logits_list, axis=1)
        cot_tokens = cot_tokens[:, 1:]  # remove the start token
        return cot_tokens, logits

    def generate_logits(
        self,
        *,
        cot_tokens: chex.Array,
        encoder_embeddings: chex.Array,
        deterministic: bool,
        pad_mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        """Applies Transformer model on a partially generated chain of thoughts.

        Args:
          cot_tokens: input tokens from the chain of thoughts.
          encoder_embeddings: input embeddings from the encoder.
          deterministic: if false dropout is applied otherwise it is not.
          pad_mask: mask to apply on the encoder embeddings to avoid attending to padding tokens.

        Returns:
          logits of shape (B, T_C, V) representing a sequence of logits.
        """
        assert encoder_embeddings.ndim == 3  # (B, T, H)

        tok_embed = self.cot_tok_embed(cot_tokens)
        pos_embed = self.cot_pos_embed(jnp.arange(self.config.cot_seq_length + 1))
        x = tok_embed + pos_embed
        x = self.tok_dropout(x, deterministic=deterministic)
        bs, t, _ = x.shape
        causal_mask = jnp.tril(jnp.ones((bs, 1, t, t), bool))  # TODO: check if this is correct

        for _ in range(self.cross_transformer_config.num_repeat_model):
            for layer in self.cross_transformer_layers:
                x = layer(
                    self_embeddings=x,
                    cross_embeddings=encoder_embeddings,
                    deterministic=deterministic,
                    self_pad_mask=causal_mask,
                    cross_pad_mask=pad_mask,
                )
        logits = self.linear_head(x)
        return logits


class Decoder(nn.Module):
    config: TransformerConfig
    cot_module_config: CoTModuleConfig

    def setup(self) -> None:
        self.cross_transformer_layers = [
            CrossTransformerLayer(self.config) for _ in range(self.config.num_layers)
        ]

    @nn.compact
    def __call__(
        self,
        *,
        encoder_embeddings: chex.Array,
        cot_tokens: Optional[chex.Array],
        deterministic: bool,
        encoder_pad_mask: Optional[chex.Array] = None,
        cot_pad_mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        """Applies Transformer model on the inputs.

        Args:
          encoder_embeddings: input embeddings from the encoder.
          cot_tokens: chain of thought tokens from the cot module.
          deterministic: if false dropout is applied otherwise it is not.
          encoder_pad_mask: mask to apply on the encoder_embeddings inputs to avoid attending to
            padding embeddings.
          cot_pad_mask: mask to apply on the cot inputs to avoid attending to padding tokens.

        Returns:
          output of shape (B, V) representing a sequence of logits.
        """
        assert encoder_embeddings.ndim == 3  # (B, T, H)

        if cot_tokens is not None:
            assert cot_tokens.ndim == 2  # (B, T_C)
            assert self.cot_module_config is not None
            tok_embed = nn.Embed(
                num_embeddings=self.cot_module_config.cot_vocab_size,
                features=self.config.emb_dim,
                name="cot_tok_embed",
            )(cot_tokens)
            pos_embed = nn.Embed(
                num_embeddings=self.cot_module_config.cot_seq_length,
                features=self.config.emb_dim,
                name="cot_pos_embed",
            )(jnp.arange(self.cot_module_config.cot_seq_length))
            cot_embeddings = tok_embed + pos_embed
            cot_embeddings = nn.Dropout(rate=self.config.dropout_rate)(
                cot_embeddings, deterministic=deterministic
            )
        else:
            cot_embeddings = None
        x = encoder_embeddings

        for _ in range(self.config.num_repeat_model):
            for layer in self.cross_transformer_layers:
                x = layer(
                    self_embeddings=x,
                    cross_embeddings=cot_embeddings,
                    deterministic=deterministic,
                    self_pad_mask=encoder_pad_mask,
                    cross_pad_mask=cot_pad_mask,
                )
        x = nn.LayerNorm(dtype=self.config.dtype, use_bias=self.config.use_bias)(x)
        x = x.mean(axis=1)
        logits = nn.Dense(self.config.output_vocab_size, self.config.use_bias, self.config.dtype)(x)

        return logits


class AugmentedTransformer(nn.Module):
    """Transformer Model which produces intermediate CoT tokens."""

    encoder_config: TransformerConfig
    cot_module_config: Optional[CoTModuleConfig]
    decoder_config: TransformerConfig

    def setup(self) -> None:
        self.encoder = Encoder(self.encoder_config)
        self.cot_module = CoTModule(self.cot_module_config) if self.cot_module_config else None
        self.decoder = Decoder(self.decoder_config, self.cot_module_config)

    def encode(
        self,
        *,
        inputs: chex.Array,
        deterministic: bool,
        pad_mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        return self.encoder(inputs=inputs, deterministic=deterministic, pad_mask=pad_mask)

    def generate_cot_logits_from_encoder_embeddings(
        self,
        *,
        cot_tokens: chex.Array,
        encoder_embeddings: chex.Array,
        deterministic: bool,
        pad_mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        assert isinstance(self.cot_module, CoTModule)
        return self.cot_module.generate_logits(
            cot_tokens=cot_tokens,
            encoder_embeddings=encoder_embeddings,
            deterministic=deterministic,
            pad_mask=pad_mask,
        )

    def cot_module_call(
        self,
        *,
        encoder_embeddings: chex.Array,
        deterministic: bool,
        cot_key: Optional[chex.PRNGKey] = None,
        pad_mask: Optional[chex.Array] = None,
    ) -> tuple[chex.Array, chex.Array]:
        assert self.cot_module is not None
        return self.cot_module(
            encoder_embeddings=encoder_embeddings,
            deterministic=deterministic,
            cot_key=cot_key,
            pad_mask=pad_mask,
        )

    def decode(
        self,
        *,
        encoder_embeddings: chex.Array,
        cot_tokens: Optional[chex.Array],
        deterministic: bool,
        encoder_pad_mask: Optional[chex.Array] = None,
        cot_pad_mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        return self.decoder(
            encoder_embeddings=encoder_embeddings,
            cot_tokens=cot_tokens,
            deterministic=deterministic,
            encoder_pad_mask=encoder_pad_mask,
            cot_pad_mask=cot_pad_mask,
        )

    @nn.compact
    def __call__(
        self,
        *,
        inputs: chex.Array,
        deterministic: bool,
        cot_key: Optional[chex.PRNGKey] = None,
        pad_mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        """Applies AugmentedTransformer model on the inputs.

        Args:
          inputs: input data
          deterministic: if false dropout is applied otherwise it is not.
          cot_key: random key to sample tokens in the CoT Module.
          pad_mask: mask to apply on the inputs to avoid attending to padding tokens.

        Returns:
          output of an augmented transformer.
        """
        # Encoder block.
        x = self.encoder(inputs=inputs, deterministic=deterministic, pad_mask=pad_mask)

        # CoT Module block.
        if self.cot_module is not None:
            cot_tokens, _ = self.cot_module(
                encoder_embeddings=x,
                deterministic=deterministic,
                cot_key=cot_key,
                pad_mask=pad_mask,
            )
        else:
            cot_tokens = None

        # Decoder block.
        logits = self.decoder(
            encoder_embeddings=x,
            cot_tokens=cot_tokens,
            deterministic=deterministic,
            encoder_pad_mask=pad_mask,
            cot_pad_mask=None,
        )
        return logits


if __name__ == "__main__":
    seq_length = 10
    encoder_config = TransformerConfig(
        vocab_size=seq_length,
        output_vocab_size=None,
        emb_dim=384,
        num_heads=6,
        num_layers=1,
        num_repeat_model=0,
        mlp_dim_factor=4,
        max_len=seq_length,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
    )
    cot_seq_length = 5
    cot_module_config = CoTModuleConfig(
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
        cot_vocab_size=seq_length,
    )
    # cot_module_config = None
    decoder_config = TransformerConfig(
        vocab_size=seq_length,
        output_vocab_size=seq_length,
        emb_dim=384,
        num_heads=6,
        num_layers=2,
        num_repeat_model=1,
        mlp_dim_factor=4,
        max_len=seq_length,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
    )
    model = AugmentedTransformer(
        encoder_config,
        cot_module_config,
        decoder_config,
    )
    key = jax.random.PRNGKey(0)
    example = jax.random.randint(key, (1, seq_length), minval=0, maxval=seq_length)
    params = model.init(key, inputs=example, deterministic=True)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print("Number of parameters: {:,}".format(num_params))
    apply_fn = jax.jit(model.apply, static_argnames="deterministic")
    cot_key = jax.random.PRNGKey(1)
    output = apply_fn(
        params,
        inputs=example,
        deterministic=False,
        cot_key=cot_key,
        rngs={"dropout": key},
    )
