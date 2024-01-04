from typing import Callable

from trax.models.transformer import _EncoderBlock
from trax import layers as tl


def TransformerEncoder(
    vocab_size: int,
    n_classes: int = 10,
    d_model: int = 512,
    d_ff: int = 2048,
    n_layers: int = 6,
    n_heads: int = 8,
    dropout: float = 0.1,
    max_len: int = 2048,
    mode: str = "train",
    ff_activation: Callable = tl.Relu,
):
    """ADAPTED FROM trax.models.TransformerEncoder.
    Returns a Transformer encoder model.

    The input to the model is a tensor of tokens.

    Args:
      vocab_size: int: vocab size
      n_classes: how many classes on output
      d_model: int:  depth of embedding
      d_ff: int: depth of feed-forward layer
      n_layers: int: number of encoder/decoder layers
      n_heads: int: number of attention heads
      dropout: float: dropout rate (how much to drop out)
      max_len: int: maximum symbol length for positional encoding
      mode: str: 'train' or 'eval'
      ff_activation: the non-linearity in feed-forward layer

    Returns:
      A Transformer model as a layer that maps from a tensor of tokens to
      activations over a set of output classes.
    """
    positional_encoder = [
        tl.Embedding(d_model, vocab_size),
        tl.Dropout(rate=dropout, name="emb_dropout", mode=mode),
        tl.PositionalEncoding(max_len=max_len),
    ]

    encoder_blocks = [
        _EncoderBlock(d_model, d_ff, n_heads, dropout, i, mode, ff_activation)
        for i in range(n_layers)
    ]

    return tl.Serial(
        tl.Branch(positional_encoder, tl.PaddingMask()),
        encoder_blocks,
        tl.Select([0], n_in=2),
        tl.LayerNorm(),
        tl.Mean(axis=1),
        tl.Dense(n_classes),
        tl.LogSoftmax(),
    )


if __name__ == "__main__":
    seq_length = 10
    model = TransformerEncoder(
        vocab_size=seq_length,
        n_classes=seq_length,
        d_model=384,
        d_ff=1536,
        n_layers=6,
        n_heads=6,
        dropout=0.0,
        max_len=seq_length,
        mode="train",
    )
    import jax.numpy as jnp

    model(jnp.array([1, 3, 2, 1, 4, 5, 1, 9, 4, 6], int))
