"""Adapted from https://github.com/google/flax/blob/main/examples/nlp_seq/models.py"""

from typing import TYPE_CHECKING, Any, Optional

import chex
import jax.numpy as jnp
from flax import linen as nn

if TYPE_CHECKING:
    from dataclasses import dataclass, field
else:
    from flax.struct import dataclass, field


@dataclass
class TransformerConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

    # TODO: do ablation of what parameters actually matter (e.g. activation,
    # use_bias or dropout_rate might not matter)
    vocab_size: Optional[int]
    output_vocab_size: Optional[int]
    num_repeat_model: int
    num_layers: int
    num_heads: int
    emb_dim_per_head: int
    mlp_dim_factor: float
    max_len: Optional[int]
    dropout_rate: float
    attention_dropout_rate: float
    use_bias: bool = False
    activation: str = "silu"
    dtype: Any = jnp.float32
    emb_dim: int = field(default=None)

    def __post_init__(self):
        object.__setattr__(self, "emb_dim", self.num_heads * self.emb_dim_per_head)


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
        x = nn.Dense(int(config.mlp_dim_factor * config.emb_dim), config.use_bias, config.dtype)(
            inputs
        )
        x = self.activation(x)
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(actual_out_dim, config.use_bias, config.dtype)(x)
        output = nn.Dropout(rate=config.dropout_rate)(output, deterministic=deterministic)
        return output


class TransformerLayer(nn.Module):
    # TODO: remove and use the CrossTransformerLayer instead.
    """Transformer encoder layer.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerConfig

    @nn.compact
    def __call__(
        self,
        embeddings: chex.Array,
        deterministic: bool,
        pad_mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        """Applies TransformerLayer module.

        Args:
          embeddings: input embeddings.
          deterministic: if false dropout is applied otherwise it is not.

        Returns:
          output after transformer encoder layer.
        """
        config = self.config

        # Attention block.
        assert embeddings.ndim == 3
        x = nn.LayerNorm(dtype=config.dtype, use_bias=config.use_bias)(embeddings)
        x = nn.MultiHeadDotProductAttention(
            num_heads=config.num_heads,
            dtype=config.dtype,
            dropout_rate=config.attention_dropout_rate,
            use_bias=config.use_bias,
        )(inputs_q=x, inputs_kv=x, mask=pad_mask, deterministic=deterministic)
        residuals = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)
        embeddings += residuals

        # MLP block.
        x = nn.LayerNorm(dtype=config.dtype, use_bias=config.use_bias)(embeddings)
        residuals = MlpBlock(config=config)(x, deterministic=deterministic)
        embeddings += residuals
        return embeddings


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
          cross_embeddings (optional): input data to do cross-attention on.
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
        self_embeddings += residuals

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
            self_embeddings += residuals

        # MLP block.
        x = nn.LayerNorm(dtype=config.dtype, use_bias=config.use_bias)(self_embeddings)
        residuals = MlpBlock(config=config)(x, deterministic=deterministic)
        self_embeddings += residuals

        return self_embeddings
