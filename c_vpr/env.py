from typing import Union

import chex
import jax
import jax.numpy as jnp


class C_VPR:  # noqa: N801
    """Conditional Pointer Value Retrieval from adaptivity and modularity for efficient generalization
    over task complexity [Abnar et al., 2023].
    """

    def __init__(self, input_length: int = 100) -> None:  # noqa: N803
        """Args:
        input_length: int, length of the input sequence.
        """
        self.input_length = input_length

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        example = jax.random.randint(
            key, shape=(self.input_length,), minval=0, maxval=self.input_length
        )
        return example

    def sample_n_hops(
        self,
        key: chex.PRNGKey,
        num_hops: int,
        return_cot: bool = False,
        return_target: bool = False,
    ) -> Union[chex.Array, tuple[chex.Array, ...]]:
        """Uniformly samples a sequence with `num_hops` in it."""
        pointers_key, example_key, last_index_key = jax.random.split(key, 3)
        pointers = jax.random.choice(
            pointers_key, jnp.arange(1, self.input_length), (num_hops,), replace=False
        )
        pointers = jnp.sort(pointers)
        example = jax.random.randint(
            example_key,
            shape=(self.input_length,),
            minval=0,
            maxval=self.input_length,
        )

        # Write the pointers in the sequence.
        indices = jnp.concatenate([jnp.zeros((1,), int), pointers[:-1]])
        example = example.at[indices].set(pointers)

        # Resample the last index to make sure it is less or equal than the highest pointer.
        last_index = pointers[-1]
        target = jax.random.randint(
            last_index_key, shape=(), minval=0, maxval=last_index + 1
        )
        example = example.at[last_index].set(target)

        if return_cot:
            cot = pointers
            if return_target:
                return example, cot, target
            else:
                return example, cot
        else:
            if return_target:
                return example, target
            else:
                return example

    def get_num_hops(self, example: chex.Array) -> chex.Array:
        num_hops, pointer = jnp.asarray(0, int), 0
        new_pointer = example[pointer]

        def body_fn(carry: tuple) -> tuple:
            num_hops, pointer, new_pointer = carry
            num_hops += 1
            pointer = new_pointer
            new_pointer = example[pointer]
            return num_hops, pointer, new_pointer

        # Follow the increasing sequence of pointers while the new pointer is greater than the
        # current one.
        num_hops, *_ = jax.lax.while_loop(
            cond_fun=lambda c: c[2] > c[1],
            body_fun=body_fn,
            init_val=(num_hops, pointer, new_pointer),
        )
        return num_hops
