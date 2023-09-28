import jax.numpy as jnp
import jax

jax.config.update("jax_platform_name", "cpu")
import matplotlib.pyplot as plt

from network import forward, init_network


key = jax.random.PRNGKey(0)
batch_size = 10
lr = 3e-2
num_epochs = 300


@jax.jit
def update(params, x, y):
    """Compute the gradient for a batch and update the parameters."""

    def loss_fn(params, x, y):
        y_pred = jax.vmap(forward, (None, 0))(params, x)
        return jnp.mean((y - y_pred) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    params = jax.tree_util.tree_map(lambda x, g: x - lr * g, params, grads)
    metrics = {
        "loss": loss,
        **{k + "_grad": v for k, v in grads.items()},
        **params,
    }
    return params, metrics


def epoch_fn(params, key):
    x = jax.random.uniform(key, (batch_size,), minval=-1.0, maxval=1.0)
    y = x
    params, metrics = update(params, x, y)
    return params, metrics


params = init_network(key)
keys = jax.random.split(key, num_epochs)
params, metrics = jax.lax.scan(epoch_fn, params, keys)


n_plots = len(metrics)
fig, axes = plt.subplots(1, n_plots, figsize=(3 * n_plots, 4))
downsampling = 5
for idx, (list_name, values) in enumerate(metrics.items()):
    to_plot = values[::downsampling]
    if to_plot.ndim > 1:
        to_plot = to_plot[:, 0]
    axes[idx].plot(to_plot)
    axes[idx].set_title(list_name)
    axes[idx].grid(True)

plt.tight_layout()
plt.show()
