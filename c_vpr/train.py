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
from c_vpr.augmented_transformer import AugmentedTransformer, CoTModuleConfig
from c_vpr.cycle import Cycle
from c_vpr.env import C_VPR, Env
from c_vpr.transformer import Transformer, TransformerConfig

logging.getLogger().setLevel(logging.INFO)


class Trainer:
    def __init__(
        self,
        env: Env,
        train_num_hops: int | list[int],
        eval_num_hops: int | list[int] | None,
        seq_length: int,
        batch_size: int,
        eval_size: int,
        augmented_transformer: bool = True,
    ) -> None:
        self.env = env
        self.train_num_hops = (
            [train_num_hops] if isinstance(train_num_hops, int) else train_num_hops
        )
        self.eval_num_hops = [eval_num_hops] if isinstance(eval_num_hops, int) else eval_num_hops
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.eval_size = eval_size
        self.augmented_transformer = augmented_transformer

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
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate))
        apply_fn = jax.jit(model.apply, static_argnames="deterministic")
        return TrainState.create(apply_fn=apply_fn, tx=optimizer, params=params)

    def train_step(self, state: TrainState, key: chex.PRNGKey) -> tuple[TrainState, dict]:
        num_hops_key, sample_key, cot_key, dropout_key = jax.random.split(key, 4)

        def sample_n_hops(key: chex.PRNGKey, num_hops_index: int) -> tuple[chex.Array, ...]:
            if len(self.train_num_hops) == 1:
                del num_hops_index
                return self.env.sample_n_hops(key, self.train_num_hops[0], return_target=True)
            else:
                return jax.lax.switch(  # type: ignore
                    num_hops_index,
                    [
                        functools.partial(
                            self.env.sample_n_hops,
                            num_hops=num_hops,
                            return_target=True,
                        )
                        for num_hops in self.train_num_hops
                    ],
                    key,
                )

        num_hops_indices = jax.random.choice(
            num_hops_key,
            jnp.arange(len(self.train_num_hops)),
            (self.batch_size,),
            replace=True,
        )
        sample_keys = jax.random.split(sample_key, self.batch_size)
        examples, labels = jax.vmap(sample_n_hops)(sample_keys, num_hops_indices)

        def loss_fn(params: dict, dropout_key: chex.PRNGKey) -> tuple[TrainState, chex.Array]:
            input_kwargs = dict(  # noqa: C408
                variables={"params": params},
                inputs=examples,
                deterministic=False,
                rngs={"dropout": dropout_key},
            )
            if self.augmented_transformer:
                input_kwargs.update(cot_key=cot_key)
            logits = state.apply_fn(**input_kwargs)
            loss = self.cross_entropy_loss(logits=logits, labels=labels)
            return loss, logits

        grads, logits = jax.grad(loss_fn, has_aux=True)(state.params, dropout_key)
        state = state.apply_gradients(grads=grads)
        metrics = self.compute_metrics(logits, labels)
        grad_norm = jnp.sqrt(sum([jnp.sum(x**2) for x in jax.tree_util.tree_leaves(grads)]))
        metrics.update(grad_norm=grad_norm)
        return state, metrics

    def train_epoch(
        self, state: TrainState, key: chex.PRNGKey, num_steps: int
    ) -> tuple[TrainState, dict]:
        keys = jax.random.split(key, num_steps)
        state, metrics = jax.lax.scan(self.train_step, state, keys)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return state, metrics

    def eval(self, state: TrainState, key: chex.PRNGKey) -> dict[str, chex.Array]:
        metrics: dict[str, chex.Array] = {}
        if self.eval_num_hops is None:
            return metrics
        sample_keys = jax.random.split(key, len(self.eval_num_hops))
        for num_hops, sample_key in zip(self.eval_num_hops, sample_keys):
            keys = jax.random.split(sample_key, self.eval_size)
            examples, labels = jax.vmap(
                functools.partial(self.env.sample_n_hops, num_hops=num_hops, return_target=True)
            )(keys)
            logits = state.apply_fn(
                {"params": state.params},
                inputs=examples,
                deterministic=True,
            )
            metrics.update(
                {
                    f"eval/num_hops:{num_hops}/{k}": v
                    for k, v in self.compute_metrics(logits, labels).items()
                }
            )

        return metrics

    def train(
        self,
        state: TrainState,
        key: chex.PRNGKey,
        num_iterations: int,
        log_every: int,
    ) -> TrainState:
        jit_train_epoch = jax.jit(functools.partial(self.train_epoch, num_steps=log_every))
        jit_eval = jax.jit(self.eval)
        num_epochs = num_iterations // log_every
        for epoch in trange(1, num_epochs + 1):
            key, epoch_key, eval_key = jax.random.split(key, 3)
            state, metrics = jit_train_epoch(state, epoch_key)
            metrics.update(jit_eval(state, eval_key))
            wandb.log(metrics, step=epoch * log_every)
        return state

    def cross_entropy_loss(self, logits: chex.Array, labels: chex.Array) -> chex.Array:
        one_hot_encoded_labels = jax.nn.one_hot(labels, num_classes=self.seq_length)
        return optax.softmax_cross_entropy(logits=logits, labels=one_hot_encoded_labels).mean()

    def compute_metrics(self, logits: chex.Array, labels: chex.Array) -> dict[str, chex.Array]:
        loss = self.cross_entropy_loss(logits=logits, labels=labels)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
        }
        return metrics

    def save_checkpoint(self, ckpt_path: str, state: TrainState, iteration: int) -> None:
        with open(ckpt_path, "wb") as outfile:
            outfile.write(msgpack_serialize(to_state_dict(state)))
        run_name = wandb.run.name.replace(",", "").replace(":", "").replace(" ", "")
        artifact = wandb.Artifact(f"{run_name}--checkpoint", type="model")
        artifact.add_file(ckpt_path)
        wandb.log_artifact(artifact, aliases=["latest", f"iteration_{iteration}"])

    def load_checkpoint(
        self, ckpt_file: str, state: TrainState, run_name: Optional[str] = None
    ) -> TrainState:
        run_name = run_name or wandb.run.name
        run_name = run_name.replace(",", "").replace(":", "").replace(" ", "")
        artifact = wandb.use_artifact(f"{run_name}--checkpoint:latest")
        artifact_dir = artifact.download()
        ckpt_path = os.path.join(artifact_dir, ckpt_file)
        with open(ckpt_path, "rb") as data_file:
            byte_data = data_file.read()
        return from_bytes(state, byte_data)


def run_transformer_exp(
    env_name: str = "C_VPR",
    train_num_hops: int | list[int] = 3,
    eval_num_hops: int | list[int] | None = None,
    seq_length: int = 10,
    num_layers: int = 2,
    num_repeat_model: int = 1,
    emb_dim: int = 384,
    num_heads: int = 6,
    mlp_dim_factror: float = 4,
    dropout_rate: float = 0.0,
    learning_rate: float = 5e-4,
    num_iterations: int = 100_000,
    batch_size: int = 512,
    eval_size: int = 500,
    log_every: int = 100,
    run_name: Optional[str] = None,
) -> None:
    config = TransformerConfig(
        vocab_size=seq_length,
        output_vocab_size=seq_length,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_repeat_model=num_repeat_model,
        mlp_dim_factror=mlp_dim_factror,
        max_len=seq_length,
        dropout_rate=dropout_rate,
        attention_dropout_rate=dropout_rate,
    )
    model = Transformer(config)
    if env_name == "C_VPR":
        env = C_VPR(seq_length)
    elif env_name == "Cycle":
        env = Cycle(seq_length)  # type: ignore
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    wandb.init(
        project="FactorExe",
        config=config.__dict__,
        name=run_name,
    )
    wandb.config.train_num_hops = train_num_hops
    wandb.config.eval_num_hops = eval_num_hops
    wandb.config.learning_rate = learning_rate
    wandb.config.num_iterations = num_iterations
    wandb.config.batch_size = batch_size
    wandb.config.eval_size = eval_size

    trainer = Trainer(
        env,
        train_num_hops,
        eval_num_hops,
        seq_length,
        batch_size,
        eval_size,
        augmented_transformer=False,
    )
    key = jax.random.PRNGKey(0)
    state = trainer.init_train_state(model, key, learning_rate)
    state = trainer.train(state, key, num_iterations, log_every)
    trainer.save_checkpoint("checkpoint.msgpack", state, iteration=num_iterations)
    wandb.finish()


def run_augmented_transformer_exp(
    env_name: str = "C_VPR",
    train_num_hops: int | list[int] = 3,
    eval_num_hops: int | list[int] | None = None,
    seq_length: int = 10,
    encoder_num_heads: int = 6,
    encoder_num_layers: int = 2,
    encoder_num_repeat_model: int = 1,
    cot_module: bool = False,
    decoder_num_heads: int = 6,
    decoder_num_layers: int = 2,
    decoder_num_repeat_model: int = 1,
    emb_dim: int = 384,
    mlp_dim_factror: float = 4,
    all_dropouts_rate: float = 0.0,
    learning_rate: float = 5e-4,
    num_iterations: int = 100_000,
    batch_size: int = 512,
    eval_size: int = 500,
    log_every: int = 100,
    run_name: Optional[str] = None,
) -> None:
    encoder_config = TransformerConfig(
        vocab_size=seq_length,
        output_vocab_size=None,
        emb_dim=emb_dim,
        num_heads=encoder_num_heads,
        num_layers=encoder_num_layers,
        num_repeat_model=encoder_num_repeat_model,
        mlp_dim_factror=mlp_dim_factror,
        max_len=seq_length,
        dropout_rate=all_dropouts_rate,
        attention_dropout_rate=all_dropouts_rate,
    )
    if cot_module:
        cot_seq_length = 5
        cot_num_heads = 6
        cot_num_layers = 1
        cot_num_repeat_model = 1
        cot_module_config = CoTModuleConfig(
            cross_transformer_config=TransformerConfig(
                vocab_size=None,
                output_vocab_size=None,
                emb_dim=emb_dim,
                num_heads=cot_num_heads,
                num_layers=cot_num_layers,
                num_repeat_model=cot_num_repeat_model,
                mlp_dim_factror=mlp_dim_factror,
                max_len=None,
                dropout_rate=all_dropouts_rate,
                attention_dropout_rate=all_dropouts_rate,
            ),
            cot_seq_length=cot_seq_length,
            cot_vocab_size=seq_length,
        )
    else:
        cot_module_config = None
    decoder_config = TransformerConfig(
        vocab_size=None,
        output_vocab_size=seq_length,
        emb_dim=emb_dim,
        num_heads=decoder_num_heads,
        num_layers=decoder_num_layers,
        num_repeat_model=decoder_num_repeat_model,
        mlp_dim_factror=mlp_dim_factror,
        max_len=None,
        dropout_rate=all_dropouts_rate,
        attention_dropout_rate=all_dropouts_rate,
    )
    model = AugmentedTransformer(
        encoder_config,
        cot_module_config,
        decoder_config,
    )
    if env_name == "C_VPR":
        env = C_VPR(seq_length)
    elif env_name == "Cycle":
        env = Cycle(seq_length)  # type: ignore
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    config = {}
    for c in [encoder_config, cot_module_config, decoder_config]:
        if c is not None:
            config.update(c.__dict__)
    wandb.init(
        project="FactorExe",
        config=config,
        name=run_name,
    )
    wandb.config.train_num_hops = train_num_hops
    wandb.config.eval_num_hops = eval_num_hops
    wandb.config.learning_rate = learning_rate
    wandb.config.num_iterations = num_iterations
    wandb.config.batch_size = batch_size
    wandb.config.eval_size = eval_size

    trainer = Trainer(
        env,
        train_num_hops,
        eval_num_hops,
        seq_length,
        batch_size,
        eval_size,
        augmented_transformer=True,
    )
    key = jax.random.PRNGKey(0)
    state = trainer.init_train_state(model, key, learning_rate)
    state = trainer.train(state, key, num_iterations, log_every)
    trainer.save_checkpoint("checkpoint.msgpack", state, iteration=num_iterations)
    wandb.finish()


if __name__ == "__main__":
    # Selected C_VPR difficulties: [5-150, 10-300, 20-600]
    # Selected Cycle difficulties: [3-50]
    run_augmented_transformer_exp(
        env_name="Cycle",
        train_num_hops=1,
        eval_num_hops=[1, 2, 3, 4],
        seq_length=50,
        batch_size=256,
        log_every=100,
        encoder_num_repeat_model=0,
        cot_module=False,
        decoder_num_layers=1,
        num_iterations=10_000,
        run_name="Cycle 1-50, AT1",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        train_num_hops=2,
        eval_num_hops=[1, 2, 3, 4],
        seq_length=50,
        batch_size=256,
        log_every=100,
        encoder_num_repeat_model=0,
        cot_module=False,
        decoder_num_layers=1,
        num_iterations=10_000,
        run_name="Cycle 2-50, AT1",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        train_num_hops=2,
        eval_num_hops=[1, 2, 3, 4],
        seq_length=50,
        batch_size=256,
        log_every=100,
        encoder_num_repeat_model=0,
        cot_module=False,
        decoder_num_layers=2,
        num_iterations=10_000,
        run_name="Cycle 2-50, AT2",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        train_num_hops=2,
        eval_num_hops=[1, 2, 3, 4],
        seq_length=50,
        batch_size=256,
        log_every=100,
        encoder_num_repeat_model=0,
        cot_module=False,
        decoder_num_layers=3,
        num_iterations=10_000,
        run_name="Cycle 2-50, AT3",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        train_num_hops=2,
        eval_num_hops=[1, 2, 3, 4],
        seq_length=50,
        batch_size=256,
        log_every=100,
        encoder_num_repeat_model=0,
        cot_module=False,
        decoder_num_layers=1,
        decoder_num_repeat_model=2,
        num_iterations=10_000,
        run_name="Cycle 2-50, AT1-2",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        train_num_hops=3,
        eval_num_hops=[1, 2, 3, 4],
        seq_length=50,
        batch_size=256,
        log_every=100,
        encoder_num_repeat_model=0,
        cot_module=False,
        decoder_num_layers=1,
        num_iterations=10_000,
        run_name="Cycle 3-50, AT1",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        train_num_hops=3,
        eval_num_hops=[1, 2, 3, 4],
        seq_length=50,
        batch_size=256,
        log_every=100,
        encoder_num_repeat_model=0,
        cot_module=False,
        decoder_num_layers=2,
        num_iterations=10_000,
        run_name="Cycle 3-50, AT2",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        train_num_hops=3,
        eval_num_hops=[1, 2, 3, 4],
        seq_length=50,
        batch_size=256,
        log_every=100,
        encoder_num_repeat_model=0,
        cot_module=False,
        decoder_num_layers=1,
        decoder_num_repeat_model=2,
        num_iterations=10_000,
        run_name="Cycle 3-50, AT1-2",
    )
