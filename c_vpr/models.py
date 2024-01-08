"""Adapted from https://github.com/google/flax/blob/main/examples/nlp_seq/models.py"""

from typing import Any, Callable, Optional

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import struct


@struct.dataclass
class TransformerConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

    vocab_size: int
    output_vocab_size: int
    emb_dim: int
    num_heads: int
    num_layers: int
    num_repeat_model: int
    mlp_dim: int
    max_len: int
    dropout_rate: float
    attention_dropout_rate: float
    use_bias: bool
    activation: str = "silu"
    learn_posemb: bool = False
    dtype: Any = jnp.float32


class AddPositionEmbs(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs: chex.Array) -> chex.Array:
        """Applies AddPositionEmbs module.

        By default this layer uses a fixed sinusoidal embedding table if
        learn_posemb is False.

        Args:
          inputs: input data.

        Returns:
          output: `(bs, timesteps, in_dim)`
        """
        config = self.config
        # inputs.shape is (batch_size, seq_len, emb_dim)
        assert inputs.ndim == 3, (
            "Number of dimensions should be 3, but it is: %d" % inputs.ndim
        )
        length = inputs.shape[1]
        pos_emb_shape = (1, config.max_len, inputs.shape[-1])
        if config.learn_posemb:
            pos_embedding = self.param(
                "pos_embedding",
                AddPositionEmbs.sinusoidal_init(max_len=config.max_len),
                pos_emb_shape,
            )
        else:
            # Use a fixed (non-learned) sinusoidal position embedding.
            pos_embedding = AddPositionEmbs.sinusoidal_init(max_len=config.max_len)(
                None, pos_emb_shape, None
            )
        pe = pos_embedding[:, :length, :]
        return inputs + pe

    @classmethod
    def sinusoidal_init(cls, max_len: int) -> Callable:
        """1D Sinusoidal Position Embedding Initializer.

        Args:
            max_len: maximum possible length for the input

        Returns:
            output: init function returning `(1, max_len, d_feature)`
        """

        def init(
            key: chex.PRNGKey, shape: tuple, dtype: np.dtype = np.float32
        ) -> chex.Array:
            """Sinusoidal init."""
            del key, dtype
            d_feature = shape[-1]
            pe = np.zeros((max_len, d_feature), dtype=np.float32)
            position = np.arange(0, max_len)[:, np.newaxis]
            div_term = np.exp(
                np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature)
            )
            pe[:, 0::2] = np.sin(position * div_term)
            pe[:, 1::2] = np.cos(position * div_term)
            pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
            return jnp.array(pe)

        return init


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
      out_dim: optionally specify out dimension.
    """

    config: TransformerConfig
    out_dim: Optional[int] = None

    def setup(self) -> None:
        if self.config.activation == "relu":
            self.activation = nn.relu
        elif self.config.activation == "silu":
            self.activation = nn.silu
        else:
            raise ValueError

    @nn.compact
    def __call__(self, inputs: chex.Array, deterministic: bool = True) -> chex.Array:
        """Applies Transformer MlpBlock module."""
        config = self.config
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(config.mlp_dim, config.use_bias, config.dtype)(inputs)
        x = self.activation(x)
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(actual_out_dim, config.use_bias, config.dtype)(x)
        output = nn.Dropout(rate=config.dropout_rate)(
            output, deterministic=deterministic
        )
        return output


class TransformerLayer(nn.Module):
    """Transformer encoder layer.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs: chex.Array, deterministic: bool):
        """Applies TransformerLayer module.

        Args:
          inputs: input data.
          deterministic: if false dropout is applied otherwise it is not.

        Returns:
          output after transformer encoder layer.
        """
        config = self.config

        # Attention block.
        assert inputs.ndim == 3
        x = nn.LayerNorm(dtype=config.dtype, use_bias=config.use_bias)(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=config.num_heads,
            dtype=config.dtype,
            dropout_rate=config.attention_dropout_rate,
            deterministic=deterministic,
            use_bias=config.use_bias,
        )(x, x)

        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm(dtype=config.dtype, use_bias=config.use_bias)(x)
        y = MlpBlock(config=config)(y, deterministic=deterministic)
        return x + y


class Transformer(nn.Module):
    """Transformer Model for sequence tagging."""

    config: TransformerConfig

    def setup(self):
        self.transformer_layers = [
            TransformerLayer(self.config) for _ in range(self.config.num_layers)
        ]

    @nn.compact
    def __call__(self, *, inputs, deterministic: bool):
        """Applies Transformer model on the inputs.

        Args:
          inputs: input data
          train: if it is training.

        Returns:
          output of a transformer encoder.

        """
        assert inputs.ndim == 2  # (batch, len)

        config = self.config

        x = inputs.astype("int32")
        x = nn.Embed(
            num_embeddings=config.vocab_size, features=config.emb_dim, name="tok_embed"
        )(x)
        # x = AddPositionEmbs(config)(x)
        assert config.learn_posemb
        pos_embed = nn.Embed(
            num_embeddings=config.max_len, features=config.emb_dim, name="pos_embed"
        )(jnp.arange(config.max_len))
        x = x + pos_embed
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)

        for _ in range(config.num_repeat_model):
            for layer in self.transformer_layers:
                x = layer(x, deterministic=deterministic)

        x = nn.LayerNorm(dtype=config.dtype, use_bias=config.use_bias)(x)
        x = x.mean(axis=1)
        logits = nn.Dense(config.output_vocab_size, config.use_bias, config.dtype)(x)
        return logits


if __name__ == "__main__":
    seq_length = 10
    config = TransformerConfig(
        vocab_size=seq_length,
        output_vocab_size=seq_length,
        emb_dim=384,
        num_heads=6,
        num_layers=6,
        num_repeat_model=1,
        mlp_dim=1536,
        max_len=seq_length,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        use_bias=False,
        activation="silu",
        learn_posemb=True,
    )
    model = Transformer(config)
    key = jax.random.PRNGKey(0)
    example = jax.random.randint(key, (1, seq_length), minval=0, maxval=seq_length)
    params = model.init(key, inputs=example, deterministic=True)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print("Number of parameters: {:,}".format(num_params))
    apply_fn = jax.jit(model.apply, static_argnames="deterministic")
    output = apply_fn(
        params, inputs=example, deterministic=False, rngs={"dropout": key}
    )
