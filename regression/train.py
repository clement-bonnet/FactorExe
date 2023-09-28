import jax.numpy as jnp
import jax

jax.config.update("jax_platform_name", "cpu")
import matplotlib.pyplot as plt

from network import LinearNetwork, FactorNetwork
import utils


def run(with_factor: bool, num_runs: int = 1, seed: int = 0):
    batch_size = 10
    lr = 2e-2
    factor_loss_coeff = 1.0
    num_epochs = 50

    net = FactorNetwork() if with_factor else LinearNetwork()

    def one_run(key):
        @jax.jit
        def update(params, x, y, key):
            """Compute the gradient for a batch and update the parameters."""

            def loss_fn(params, x, y, key):
                y_pred, factor_log_prob = jax.vmap(net.forward, (None, 0, None))(
                    params, x, key
                )
                losses = (y - y_pred) ** 2
                mse_loss = jnp.mean(losses)
                loss = mse_loss
                if factor_log_prob is not None:
                    rl_loss = jnp.mean(jax.lax.stop_gradient(losses) * factor_log_prob)
                    loss += factor_loss_coeff * rl_loss
                return loss, mse_loss

            grads, mse_loss = jax.grad(loss_fn, has_aux=True)(params, x, y, key)
            params = jax.tree_util.tree_map(lambda x, g: x - lr * g, params, grads)
            metrics = {
                "mse_loss": mse_loss,
                **utils.flatten({k + "_grad": v for k, v in grads.items()}),
                **utils.flatten(params),
            }
            return params, metrics

        def epoch_fn(params, key):
            x_key, update_key = jax.random.split(key)
            x = jax.random.uniform(x_key, (batch_size,), minval=-1.0, maxval=1.0)
            y = jnp.abs(x)
            params, metrics = update(params, x, y, update_key)
            return params, metrics

        params = net.init_network(key)
        keys = jax.random.split(key, num_epochs)
        params, metrics = jax.lax.scan(epoch_fn, params, keys)

        return params, metrics

    keys = jax.random.split(jax.random.PRNGKey(seed), num_runs)
    # TODO: use vmap
    params, metrics = jax.lax.map(one_run, keys)
    return params, metrics


if __name__ == "__main__":
    _, metrics1 = run(with_factor=True, num_runs=10)
    _, metrics2 = run(with_factor=False, num_runs=10)
    metrics2["w1_0"] = metrics2.pop("w1")
    metrics2["w1_grad_0"] = metrics2.pop("w1_grad")

    downsampling = 1
    color1, color2 = plt.rcParams["axes.prop_cycle"].by_key()["color"][:2]
    n_plots = len(metrics1)
    rows, cols = 3, (n_plots - 1) // 3 + 1
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))

    for idx, (metric_name, values1) in enumerate(metrics1.items()):
        axes[idx // cols, idx % cols].plot(
            values1[:, ::downsampling].mean(axis=0), c=color1, label="Factor"
        )
        for value in values1[:, ::downsampling]:
            axes[idx // cols, idx % cols].plot(value, c=color1, alpha=0.2)

        if not metrics2.get(metric_name) is None:
            values2 = metrics2[metric_name]
            axes[idx // cols, idx % cols].plot(
                values2[:, ::downsampling].mean(axis=0), c=color2, label="Linear"
            )
            for value in values2[:, ::downsampling]:
                axes[idx // cols, idx % cols].plot(value, c=color2, alpha=0.2)

        axes[idx // cols, idx % cols].set_title(metric_name)
        axes[idx // cols, idx % cols].grid(True)
    # for idx, (metric_name, values) in enumerate(metrics2.items()):
    #     axes[idx // cols, idx % cols].plot(
    #         values[:, ::downsampling].mean(axis=0), c=color2, label="Linear"
    #     )
    #     for value in values[:, ::downsampling]:
    #         axes[idx // cols, idx % cols].plot(value, c=color2, alpha=0.2)
    #     axes[idx // cols, idx % cols].set_title(metric_name)
    #     axes[idx // cols, idx % cols].grid(True)
    axes[0, 0].legend()
    plt.tight_layout()
    plt.show()
