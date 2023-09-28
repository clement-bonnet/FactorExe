import jax
import jax.numpy as jnp


def init_network(key, input_dim=1):
    params = {
        "w1": jax.random.normal(key, (input_dim,)),
        "b1": jnp.zeros(1),
    }
    return params


def forward(params, x):
    pred = jnp.dot(x, params["w1"]) + params["b1"]
    return pred.squeeze(axis=-1)
