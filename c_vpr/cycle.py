from typing import Union

import chex
import jax
import jax.numpy as jnp

from c_vpr.env import Env


class Cycle(Env):
    """Permutation without fixed point (aka derangement) and only one cycle."""

    def __init__(self, input_length: int = 100) -> None:
        """Args:
        input_length: int, length of the input sequence.
        """
        self.input_length = input_length

    def sample(
        self,
        key: chex.PRNGKey,
        return_cot: bool = False,
        return_target: bool = False,
    ) -> tuple[chex.Array, ...]:
        hops_key, sample_key = jax.random.split(key)
        num_hops = jax.random.randint(
            hops_key, shape=(), minval=1, maxval=self.input_length
        )
        args = self.sample_n_hops(sample_key, num_hops, return_cot, return_target)
        if isinstance(args, tuple):
            return (num_hops, *args)
        else:
            return num_hops, args

    def sample_n_hops(
        self,
        key: chex.PRNGKey,
        num_hops: int,
        return_cot: bool = False,
        return_target: bool = False,
    ) -> Union[chex.Array, tuple[chex.Array, ...]]:
        """Uniformly samples a cycle with `num_hops` in it."""
        permutation = jax.random.permutation(key, jnp.arange(self.input_length))
        zero_index = permutation.argmin()
        target = permutation[(zero_index + 1 + num_hops) % self.input_length]
        sequence = jnp.empty_like(permutation)
        sequence = sequence.at[permutation].set(jnp.roll(permutation, -1))
        if return_cot:
            cot = jnp.roll(permutation, -zero_index - 1)[:num_hops]
            if return_target:
                return sequence, cot, target
            else:
                return sequence, cot
        else:
            if return_target:
                return sequence, target
            else:
                return sequence
