import functools
import logging
import os
from typing import Optional

import chex
import jax
import jax.numpy as jnp
import optax
from flax.serialization import from_bytes, msgpack_serialize, to_state_dict
from flax.training.train_state import TrainState
from tqdm.auto import trange

import wandb
from c_vpr.env import C_VPR
from c_vpr.models import Transformer, TransformerConfig

logging.getLogger().setLevel(logging.INFO)


class Trainer:
    def __init__(
        self, c_vpr: C_VPR, num_hops: int, seq_length: int, batch_size: int
    ) -> None:
        self.c_vpr = c_vpr
        self.num_hops = num_hops
        self.seq_length = seq_length
        self.batch_size = batch_size

    def init_train_state(
        self,
        model: Transformer,
        key: chex.PRNGKey,
        learning_rate: float,
        verbose: bool = True,
    ) -> TrainState:
        inputs = jnp.zeros((1, self.seq_length), int)
        params = model.init(key, inputs=inputs, deterministic=True)["params"]
        if verbose:
            num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
            logging.info("Number of parameters: {:,}".format(num_params))
        optimizer = optax.adamw(learning_rate)
        apply_fn = jax.jit(model.apply, static_argnames="deterministic")
        return TrainState.create(apply_fn=apply_fn, tx=optimizer, params=params)

    def train_step(
        self, state: TrainState, key: chex.PRNGKey
    ) -> tuple[TrainState, dict]:
        example_key, dropout_key = jax.random.split(key)
        example_keys = jax.random.split(example_key, self.batch_size)
        return_target = True
        examples, labels = jax.vmap(self.c_vpr.sample_n_hops, in_axes=(0, None, None))(
            example_keys, self.num_hops, return_target
        )

        def loss_fn(params: dict, key: chex.PRNGKey) -> tuple[TrainState, chex.Array]:
            logits = state.apply_fn(
                {"params": params},
                inputs=examples,
                deterministic=False,
                rngs={"dropout": key},
            )
            loss = self.cross_entropy_loss(logits=logits, labels=labels)
            return loss, logits

        grads, logits = jax.grad(loss_fn, has_aux=True)(state.params, dropout_key)
        state = state.apply_gradients(grads=grads)
        metrics = self.compute_metrics(logits=logits, labels=labels)
        return state, metrics

    def train_epoch(
        self, state: TrainState, key: chex.PRNGKey, num_steps: int
    ) -> tuple[TrainState, dict]:
        keys = jax.random.split(key, num_steps)
        state, metrics = jax.lax.scan(self.train_step, state, keys)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return state, metrics

    def train(
        self,
        state: TrainState,
        key: chex.PRNGKey,
        num_iterations: int,
        log_every: int,
    ) -> TrainState:
        jit_train_epoch = jax.jit(
            functools.partial(self.train_epoch, num_steps=log_every)
        )
        num_epochs = num_iterations // log_every
        for epoch in trange(1, num_epochs + 1):
            key, epoch_key = jax.random.split(key)
            state, metrics = jit_train_epoch(state, epoch_key)
            wandb.log(metrics, step=epoch * log_every)
            if metrics["accuracy"] > 0.95:
                break
        return state

    def cross_entropy_loss(self, logits: chex.Array, labels: chex.Array) -> float:
        one_hot_encoded_labels = jax.nn.one_hot(labels, num_classes=self.seq_length)
        return optax.softmax_cross_entropy(
            logits=logits, labels=one_hot_encoded_labels
        ).mean()

    def compute_metrics(self, logits: chex.Array, labels: chex.Array) -> dict:
        loss = self.cross_entropy_loss(logits=logits, labels=labels)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
        }
        return metrics

    def save_checkpoint(
        self, ckpt_path: str, state: TrainState, iteration: int
    ) -> None:
        with open(ckpt_path, "wb") as outfile:
            outfile.write(msgpack_serialize(to_state_dict(state)))
        artifact = wandb.Artifact(f"{wandb.run.name}-checkpoint", type="model")
        artifact.add_file(ckpt_path)
        wandb.log_artifact(artifact, aliases=["latest", f"iteration_{iteration}"])

    def load_checkpoint(self, ckpt_file: str, state: TrainState) -> TrainState:
        artifact = wandb.use_artifact(f"{wandb.run.name}-checkpoint:latest")
        artifact_dir = artifact.download()
        ckpt_path = os.path.join(artifact_dir, ckpt_file)
        with open(ckpt_path, "rb") as data_file:
            byte_data = data_file.read()
        return from_bytes(state, byte_data)


def run_exp(
    num_hops: int = 5,
    seq_length: int = 10,
    num_layers: int = 2,
    num_repeat_model: int = 1,
    run_name: Optional[str] = None,
):
    config = TransformerConfig(
        vocab_size=seq_length,
        output_vocab_size=seq_length,
        emb_dim=384,
        num_heads=6,
        num_layers=num_layers,
        num_repeat_model=num_repeat_model,
        mlp_dim=1536,
        max_len=seq_length,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
    )
    model = Transformer(config)
    c_vpr = C_VPR(seq_length)

    wandb.init(
        project="FactorExe",
        config=config.__dict__,
        name=run_name,
    )
    learning_rate = 5e-4
    num_iterations = 200_000
    batch_size = 256
    log_every = 100
    wandb.config.num_hops = num_hops
    wandb.config.learning_rate = learning_rate
    wandb.config.num_iterations = num_iterations
    wandb.config.batch_size = batch_size

    trainer = Trainer(c_vpr, num_hops, seq_length, batch_size)
    key = jax.random.PRNGKey(0)
    state = trainer.init_train_state(model, key, learning_rate)
    trainer.train(state, key, num_iterations, log_every)
    wandb.finish()


if __name__ == "__main__":
    run_exp(
        num_hops=1,
        seq_length=100,
        num_layers=6,
        run_name="num_hops: 1, seq_length: 100, num_layers: 6",
    )
    run_exp(
        num_hops=2,
        seq_length=100,
        num_layers=6,
        run_name="num_hops: 2, seq_length: 100, num_layers: 6",
    )
    run_exp(
        num_hops=3,
        seq_length=100,
        num_layers=6,
        run_name="num_hops: 3, seq_length: 100, num_layers: 6",
    )
    run_exp(
        num_hops=4,
        seq_length=100,
        num_layers=6,
        run_name="num_hops: 4, seq_length: 100, num_layers: 6",
    )
    run_exp(
        num_hops=5,
        seq_length=100,
        num_layers=6,
        run_name="num_hops: 5, seq_length: 100, num_layers: 6",
    )
    run_exp(
        num_hops=6,
        seq_length=100,
        num_layers=6,
        run_name="num_hops: 6, seq_length: 100, num_layers: 6",
    )
    run_exp(
        num_hops=10,
        seq_length=100,
        num_layers=6,
        run_name="num_hops: 10, seq_length: 100, num_layers: 6",
    )
    run_exp(
        num_hops=20,
        seq_length=100,
        num_layers=6,
        run_name="num_hops: 20, seq_length: 100, num_layers: 6",
    )
