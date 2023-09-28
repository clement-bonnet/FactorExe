import abc

import jax
import jax.numpy as jnp


class Network(abc.ABC):
    @abc.abstractmethod
    def init_network(self, key):
        pass

    @abc.abstractmethod
    def forward(self, params, x, key):
        pass


class LinearNetwork(Network):
    def __init__(self, input_dim: int = 1) -> None:
        self.input_dim = input_dim

    def init_network(self, key):
        params = {
            "w1": jax.random.normal(key, (self.input_dim,)),
            "b1": jnp.zeros(1),
        }
        return params

    def forward(self, params, x, key):
        del key
        pred = jnp.dot(x, params["w1"]) + params["b1"]
        return pred.squeeze(axis=-1), None


class FactorNetwork(Network):
    def __init__(
        self,
        input_dim: int = 1,
        factor_num_classes: int = 2,
        factor_seq_length: int = 1,
    ) -> None:
        self.input_dim = input_dim
        self.factor_num_classes = factor_num_classes
        assert factor_seq_length == 1
        self.factor_seq_length = factor_seq_length

    def init_network(self, key):
        w1_key, w_factor_1_key = jax.random.split(key)
        params = {
            "w1": jax.random.normal(
                w1_key,
                (self.input_dim + self.factor_seq_length,),
            ),
            "b1": jnp.zeros(self.factor_seq_length),
            "w_factor_1": jax.random.normal(
                w_factor_1_key, (self.input_dim, self.factor_num_classes)
            ),
            "b_factor_1": jnp.zeros(self.factor_num_classes),
        }
        return params

    def forward(self, params, x, key):
        factor, factor_log_prob = self.compute_factor(params, x, key)
        x = jnp.concatenate([x[None], factor])
        pred = jnp.dot(x, params["w1"]) + params["b1"]
        return pred.squeeze(axis=-1), factor_log_prob

    def compute_factor(self, params, x, key):
        logits = jnp.dot(x, params["w_factor_1"]) + params["b_factor_1"]
        factor = jax.random.categorical(key, logits)
        log_prob = jnp.take(jax.nn.log_softmax(logits), factor)
        return factor, log_prob
