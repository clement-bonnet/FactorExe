import chex
import jax
import jax.numpy as jnp


class C_VPR:  # noqa: N801
    """Conditional Pointer Value Retrieval from adaptivity and modularity for efficient generalization
    over task complexity [Abnar et al., 2023].
    """

    def __init__(self, K: int = 100, L: int = 100) -> None:  # noqa: N803
        """Args:
        K: int, max input value, the input being in the range [0, K-1].
        L: int, length of the input sequence.
        """
        self.max_input_value = K
        self.input_length = L

    def sample(self, key: chex.PRNGKey):
        example = jax.random.randint(
            key, shape=(self.input_length,), minval=0, maxval=self.max_input_value
        )
        return example

    def get_num_hops(self, example: chex.Array) -> int:
        num_hops, pointer = jnp.asarray(0, int), 0
        new_pointer = example[pointer]

        def body_fn(carry: tuple) -> tuple:
            num_hops, pointer, new_pointer = carry
            num_hops += 1
            pointer = new_pointer
            new_pointer = example[pointer]
            return num_hops, pointer, new_pointer

        num_hops, *_ = jax.lax.while_loop(
            cond_fun=lambda c: c[2] > c[1],
            body_fun=body_fn,
            init_val=(num_hops, pointer, new_pointer),
        )
        # while new_pointer > pointer:
        #     num_hops += 1
        #     pointer = new_pointer
        #     new_pointer = example[pointer]
        return num_hops
