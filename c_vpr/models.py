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
    mlp_dim: int
    max_len: int
    dropout_rate: float
    attention_dropout_rate: float
    posemb_init: Optional[Callable] = None  # None: fixed, otherwise learned embeddings.
    dtype: Any = jnp.float32


def sinusoidal_init(max_len: int) -> Callable:
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
        div_term = np.exp(np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
        return jnp.array(pe)

    return init


class AddPositionEmbs(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs: chex.Array) -> chex.Array:
        """Applies AddPositionEmbs module.

        By default this layer uses a fixed sinusoidal embedding table. If a
        learned position embedding is desired, pass an initializer to
        posemb_init in the configuration.

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
        if config.posemb_init is None:
            # Use a fixed (non-learned) sinusoidal position embedding.
            pos_embedding = sinusoidal_init(max_len=config.max_len)(
                None, pos_emb_shape, None
            )
        else:
            pos_embedding = self.param(
                "pos_embedding", config.posemb_init, pos_emb_shape
            )
        pe = pos_embedding[:, :length, :]
        return inputs + pe


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
      out_dim: optionally specify out dimension.
    """

    config: TransformerConfig
    out_dim: Optional[int] = None

    @nn.compact
    def __call__(self, inputs: chex.Array, deterministic: bool = True) -> chex.Array:
        """Applies Transformer MlpBlock module."""
        config = self.config
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(config.mlp_dim, dtype=config.dtype)(inputs)
        x = nn.relu(x)
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(actual_out_dim, dtype=config.dtype)(x)
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
        x = nn.LayerNorm(dtype=config.dtype)(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=config.num_heads,
            dtype=config.dtype,
            dropout_rate=config.attention_dropout_rate,
            deterministic=deterministic,
        )(x, x)

        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm(dtype=config.dtype)(x)
        y = MlpBlock(config=config)(y, deterministic=deterministic)
        return x + y


class Transformer(nn.Module):
    """Transformer Model for sequence tagging."""

    config: TransformerConfig

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
            num_embeddings=config.vocab_size, features=config.emb_dim, name="embed"
        )(x)
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)
        x = AddPositionEmbs(config)(x)

        for _ in range(config.num_layers):
            x = TransformerLayer(config)(x, deterministic=deterministic)

        x = nn.LayerNorm(dtype=config.dtype)(x)
        x = x.mean(axis=1)
        logits = nn.Dense(config.output_vocab_size)(x)
        return logits


if __name__ == "__main__":
    seq_length = 10
    config = TransformerConfig(
        vocab_size=seq_length,
        output_vocab_size=seq_length,
        emb_dim=384,
        num_heads=6,
        num_layers=6,
        mlp_dim=1536,
        max_len=seq_length,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
    )
    model = Transformer(config)
    key = jax.random.PRNGKey(0)
    example = jax.random.randint(key, (1, seq_length), minval=0, maxval=seq_length)
    params = model.init(key, inputs=example, deterministic=True)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print("Number of parameters: {:,}".format(num_params))
    model.apply(params, inputs=example, deterministic=False, rngs={"dropout": key})
