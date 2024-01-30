import functools
import logging
import os
from enum import Enum
from typing import Optional

import chex
import jax
import jax.numpy as jnp
import optax
from flax.serialization import from_bytes, msgpack_serialize, to_state_dict
from flax.training.train_state import TrainState
from tqdm.auto import trange

import wandb
from c_vpr.augmented_transformer import (
    AugmentedTransformer,
    CoTModuleConfig,
    EncoderConfig,
)
from c_vpr.cycle import Cycle
from c_vpr.env import C_VPR, Env
from c_vpr.transformer_utils import TransformerConfig

logging.getLogger().setLevel(logging.INFO)


class MODE(Enum):
    SUPERVISED = 1
    COT = 2
    RL = 3


class Trainer:
    def __init__(
        self,
        model: AugmentedTransformer,
        env: Env,
        mode: MODE,
        train_num_hops: int | list[int],
        eval_num_hops: int | list[int] | None,
        seq_length: int,
        batch_size: int,
        eval_size: int,
        cot_start_token: Optional[int] = None,
        cot_loss_weight_mixing: float = 1.0,
        rl_loss_weight_mixing: float = 1.0,
        rl_baseline_batch_size: Optional[int] = None,
        decode_from_sampled_cot_tokens: bool = False,
    ) -> None:
        self.model = model
        self.env = env
        if mode not in MODE:
            raise ValueError(f"Unknown mode: {mode}")
        if mode in [MODE.COT, MODE.RL] and cot_start_token is None:
            raise ValueError("COT and RL modes require cot_start_token to be set")
        if mode == MODE.SUPERVISED and model.cot_module_config is not None:
            logging.warn(
                "Got mode: SUPERVISED and cot_module: True. However, the CoTModule is not trained"
                "in this mode. This means the tokens the encoder attends to are generated randomly"
                "which may hinder the performance of the model. Turn cot_module to False or use"
                "COT or RL mode to train the CoTModule."
            )
        self.mode = mode
        self.cot_start_token = cot_start_token
        self.train_num_hops = (
            [train_num_hops] if isinstance(train_num_hops, int) else train_num_hops
        )
        self.eval_num_hops = [eval_num_hops] if isinstance(eval_num_hops, int) else eval_num_hops
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.eval_size = eval_size
        self.cot_loss_weight_mixing = cot_loss_weight_mixing
        self.rl_loss_weight_mixing = rl_loss_weight_mixing
        self.rl_baseline_batch_size = rl_baseline_batch_size
        self.decode_from_sampled_cot_tokens = decode_from_sampled_cot_tokens

    def init_train_state(
        self,
        key: chex.PRNGKey,
        learning_rate: float,
        verbose: bool = True,
    ) -> TrainState:
        inputs = jnp.zeros((1, self.seq_length), int)
        params = self.model.init(key, inputs=inputs, deterministic=True)["params"]
        if verbose:
            num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
            logging.info("Number of parameters: {:,}".format(num_params))
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate))
        apply_fn = jax.jit(
            self.model.apply, static_argnames=["deterministic", "cot_sampling", "method"]
        )
        return TrainState.create(apply_fn=apply_fn, tx=optimizer, params=params)

    def _sample_n_hops(
        self, key: chex.PRNGKey, num_hops_index: int, return_cot: bool = False
    ) -> tuple[chex.Array, ...]:
        if len(self.train_num_hops) == 1:
            del num_hops_index
            return self.env.sample_n_hops(
                key, self.train_num_hops[0], return_cot=return_cot, return_target=True
            )
        else:
            if return_cot:
                raise NotImplementedError(
                    "Sampling with return_cot=True is not supported when using multiple "
                    "values for train_num_hops. TODO: implement cot padding."
                )
            return jax.lax.switch(  # type: ignore
                num_hops_index,
                [
                    functools.partial(
                        self.env.sample_n_hops,
                        num_hops=num_hops,
                        return_cot=return_cot,
                        return_target=True,
                    )
                    for num_hops in self.train_num_hops
                ],
                key,
            )

    def train_step_supervised(
        self, state: TrainState, key: chex.PRNGKey
    ) -> tuple[TrainState, dict]:
        num_hops_key, sample_key, cot_key, dropout_key = jax.random.split(key, 4)

        num_hops_indices = jax.random.choice(
            num_hops_key,
            jnp.arange(len(self.train_num_hops)),
            (self.batch_size,),
            replace=True,
        )
        sample_keys = jax.random.split(sample_key, self.batch_size)
        inputs, labels = jax.vmap(self._sample_n_hops)(sample_keys, num_hops_indices)

        def loss_fn(params: dict, dropout_key: chex.PRNGKey) -> tuple[TrainState, chex.Array]:
            logits, _ = state.apply_fn(
                variables={"params": params},
                inputs=inputs,
                deterministic=False,
                rngs={"dropout": dropout_key},
            )
            loss = cross_entropy_loss(logits, labels)
            return loss, logits

        grads, logits = jax.grad(loss_fn, has_aux=True)(state.params, dropout_key)
        state = state.apply_gradients(grads=grads)
        metrics = self.compute_metrics(logits, labels)
        grad_norm = jnp.sqrt(sum([jnp.sum(x**2) for x in jax.tree_util.tree_leaves(grads)]))
        metrics.update(grad_norm=grad_norm)
        return state, metrics

    def train_step_cot(self, state: TrainState, key: chex.PRNGKey) -> tuple[TrainState, dict]:
        num_hops_key, sample_key, cot_key, dropout_key = jax.random.split(key, 4)
        drop_key1, drop_key2, drop_key3 = jax.random.split(dropout_key, 3)

        num_hops_indices = jax.random.choice(
            num_hops_key,
            jnp.arange(len(self.train_num_hops)),
            (self.batch_size,),
            replace=True,
        )
        sample_keys = jax.random.split(sample_key, self.batch_size)
        inputs, cots, labels = jax.vmap(functools.partial(self._sample_n_hops, return_cot=True))(
            sample_keys, num_hops_indices
        )

        def loss_fn(params: dict) -> tuple[TrainState, chex.Array]:
            # CoT Module forward pass.
            cot_tokens = jnp.concatenate(
                [jnp.full((cots.shape[0], 1), self.cot_start_token), cots], axis=1
            )
            cot_logits = state.apply_fn(
                variables={"params": params},
                cot_tokens=cot_tokens,
                inputs=inputs,
                deterministic=False,
                inputs_pad_mask=None,
                rngs={"dropout": drop_key1},
                method=self.model.cot_module_generate_cot_logits,
            )
            cot_loss = cross_entropy_loss(logits=cot_logits, labels=cots)
            if self.decode_from_sampled_cot_tokens:
                cot_tokens, _ = state.apply_fn(
                    variables={"params": params},
                    inputs=inputs,
                    deterministic=False,
                    pad_mask=None,
                    cot_sampling=True,
                    cot_key=cot_key,
                    rngs={"dropout": drop_key2},
                    method=self.model.cot_module_call,
                )
            else:
                cot_tokens = cots

            # Encoder forward pass.
            logits = state.apply_fn(
                variables={"params": params},
                inputs=inputs,
                cot_tokens=cot_tokens,
                deterministic=False,
                inputs_pad_mask=None,
                cot_pad_mask=None,
                rngs={"dropout": drop_key3},
                method=self.model.encoder_call,
            )
            supervised_loss = cross_entropy_loss(logits, labels)

            loss = supervised_loss + self.cot_loss_weight_mixing * cot_loss
            return loss, (logits, cot_loss)

        grads, (logits, cot_loss) = jax.grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = self.compute_metrics(logits, labels)
        grad_norm = jnp.sqrt(sum([jnp.sum(x**2) for x in jax.tree_util.tree_leaves(grads)]))
        metrics.update(grad_norm=grad_norm, cot_loss=cot_loss)
        return state, metrics

    def train_step_rl(self, state: TrainState, key: chex.PRNGKey) -> tuple[TrainState, dict]:
        num_hops_key, sample_key, cot_key, dropout_key = jax.random.split(key, 4)

        num_hops_indices = jax.random.choice(
            num_hops_key,
            jnp.arange(len(self.train_num_hops)),
            (self.batch_size,),
            replace=True,
        )
        sample_keys = jax.random.split(sample_key, self.batch_size)
        inputs, labels = jax.vmap(self._sample_n_hops)(sample_keys, num_hops_indices)

        def loss_fn(params: dict) -> tuple[TrainState, chex.Array]:
            # TODO: vmap the forward below to compute a baseline for each sample.
            logits, (cot_tokens, cot_logits) = state.apply_fn(
                variables={"params": params},
                inputs=inputs,
                deterministic=False,
                cot_key=cot_key,
                cot_sampling=True,
                rngs={"dropout": dropout_key},
            )
            supervised_loss = cross_entropy_loss(logits, labels)
            rewards = jax.lax.stop_gradient(-cross_entropy_loss(logits, labels, mean=False))

            if self.rl_baseline_batch_size is not None:

                def baseline_apply(
                    inputs: chex.Array, cot_key: chex.PRNGKey
                ) -> tuple[chex.Array, tuple[chex.Array, chex.Array]]:
                    return state.apply_fn(  # type: ignore
                        variables={"params": params},
                        inputs=inputs,
                        deterministic=False,
                        cot_key=cot_key,
                        cot_sampling=True,
                        rngs={"dropout": dropout_key},
                    )

                b_cot_keys = jax.random.split(cot_key, self.rl_baseline_batch_size)
                b_logits, _ = jax.vmap(baseline_apply)(
                    inputs[None].repeat(self.rl_baseline_batch_size, axis=0), b_cot_keys
                )
                b_rewards = jax.lax.stop_gradient(
                    -cross_entropy_loss(
                        b_logits,
                        labels[None].repeat(self.rl_baseline_batch_size, axis=0),
                        mean=False,
                    )
                )
                # Subtract the mean of the baseline rewards.
                rewards -= jnp.mean(b_rewards, axis=0)

            cot_all_log_prob = jax.nn.log_softmax(cot_logits, axis=-1)
            cot_log_prob = jnp.take_along_axis(
                cot_all_log_prob, cot_tokens[..., None], axis=-1
            ).squeeze(-1)
            rl_loss = jnp.mean(-rewards[..., None] * cot_log_prob)

            loss = supervised_loss + self.rl_loss_weight_mixing * rl_loss
            return loss, (logits, rl_loss)

        grads, (logits, rl_loss) = jax.grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = self.compute_metrics(logits, labels)
        grad_norm = jnp.sqrt(sum([jnp.sum(x**2) for x in jax.tree_util.tree_leaves(grads)]))
        metrics.update(grad_norm=grad_norm, rl_loss=rl_loss)
        return state, metrics

    def train_epoch(
        self, state: TrainState, key: chex.PRNGKey, num_steps: int
    ) -> tuple[TrainState, dict]:
        keys = jax.random.split(key, num_steps)
        if self.mode == MODE.SUPERVISED:
            state, metrics = jax.lax.scan(self.train_step_supervised, state, keys)
        elif self.mode == MODE.COT:
            state, metrics = jax.lax.scan(self.train_step_cot, state, keys)
        elif self.mode == MODE.RL:
            state, metrics = jax.lax.scan(self.train_step_rl, state, keys)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return state, metrics

    def eval(
        self, state: TrainState, key: chex.PRNGKey, cot_sampling: bool = False
    ) -> dict[str, chex.Array]:
        """Note that when using chain of thought, if cot_sampling is False, we evaluate the
        model using the argmax of the cot logits, which is equivalent to sampling using a
        temperature of 0."""
        metrics: dict[str, chex.Array] = {}
        if self.eval_num_hops is None:
            return metrics
        sample_keys = jax.random.split(key, len(self.eval_num_hops))
        for num_hops, sample_key in zip(self.eval_num_hops, sample_keys):
            keys = jax.random.split(sample_key, self.eval_size)
            inputs, labels = jax.vmap(
                functools.partial(self.env.sample_n_hops, num_hops=num_hops, return_target=True)
            )(keys)
            cot_key, key = jax.random.split(key)
            logits, _ = state.apply_fn(
                variables={"params": state.params},
                inputs=inputs,
                deterministic=True,
                cot_key=cot_key,
                cot_sampling=cot_sampling,
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

    def compute_metrics(self, logits: chex.Array, labels: chex.Array) -> dict[str, chex.Array]:
        loss = cross_entropy_loss(logits=logits, labels=labels)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
        }
        return metrics

    def save_checkpoint(self, ckpt_path: str, state: TrainState, iteration: int) -> None:
        with open(ckpt_path, "wb") as outfile:
            outfile.write(msgpack_serialize(to_state_dict(state)))
        run_name = (
            wandb.run.name.replace(",", ".")
            .replace(":", "")
            .replace(" ", "")
            .replace("(", "_")
            .replace(")", "_")
        )
        artifact = wandb.Artifact(f"{run_name}--checkpoint", type="model")
        artifact.add_file(ckpt_path)
        wandb.log_artifact(artifact, aliases=["latest", f"iteration_{iteration}"])

    def load_checkpoint(
        self,
        ckpt_file: str,
        state: TrainState,
        run_name: Optional[str] = None,
        version: str = "latest",
    ) -> TrainState:
        run_name = run_name or wandb.run.name
        run_name = (
            run_name.replace(",", ".")
            .replace(":", "")
            .replace(" ", "")
            .replace("(", "_")
            .replace(")", "_")
        )
        artifact = wandb.use_artifact(f"{run_name}--checkpoint:{version}")
        artifact_dir = artifact.download()
        ckpt_path = os.path.join(artifact_dir, ckpt_file)
        with open(ckpt_path, "rb") as data_file:
            byte_data = data_file.read()
        return from_bytes(state, byte_data)


def cross_entropy_loss(logits: chex.Array, labels: chex.Array, mean: bool = True) -> chex.Array:
    num_classes = logits.shape[-1]
    one_hot_encoded_labels = jax.nn.one_hot(labels, num_classes=num_classes)
    cross_entropies = optax.softmax_cross_entropy(logits=logits, labels=one_hot_encoded_labels)
    if mean:
        return cross_entropies.mean()
    else:
        return cross_entropies


def run_augmented_transformer_exp(  # noqa: CCR001
    env_name: str = "Cycle",
    mode: MODE = MODE.SUPERVISED,
    train_num_hops: int | list[int] = 3,
    eval_num_hops: int | list[int] | None = None,
    seq_length: int = 10,
    cot_module: bool = False,
    cot_seq_length: int = 3,
    cot_vocab_size: int = 3,
    cot_module_input_encoder_num_repeat: int = 1,
    cot_module_input_encoder_num_layers: int = 1,
    cot_module_cross_transformer_num_repeat: int = 1,
    cot_module_cross_transformer_num_layers: int = 1,
    encoder_cot_encoder_num_repeat: int = 1,
    encoder_cot_encoder_num_layers: int = 1,
    encoder_cross_transformer_num_repeat: int = 1,
    encoder_cross_transformer_num_layers: int = 1,
    emb_dim: int = 384,
    num_heads: int = 6,
    mlp_dim_factor: float = 4,
    all_dropouts_rate: float = 0.0,
    cot_loss_weight_mixing: float = 1.0,
    rl_loss_weight_mixing: float = 1.0,
    rl_baseline_batch_size: Optional[int] = None,
    decode_from_sampled_cot_tokens: bool = False,
    learning_rate: float = 3e-4,
    num_iterations: int = 100_000,
    batch_size: int = 512,
    eval_size: int = 500,
    log_every: int = 100,
    seed: int = 0,
    run_name: Optional[str] = None,
) -> None:
    if cot_module:
        if mode in [MODE.COT, MODE.RL] and isinstance(train_num_hops, int):
            if cot_seq_length != train_num_hops:
                raise ValueError(
                    f"cot_seq_length ({cot_seq_length}) is different from train_num_hops "
                    f"({train_num_hops}), which means that the chain of thought sequence and the "
                    "chain of thoughts labels have different lengths. This is not supported yet."
                )
            if cot_vocab_size != seq_length:
                raise ValueError(
                    f"cot_vocab_size ({cot_vocab_size}) is different from seq_length "
                    f"({seq_length}), which is not supported yet."
                )
        cot_module_config = CoTModuleConfig(
            input_transformer_config=TransformerConfig(
                vocab_size=seq_length,
                output_vocab_size=None,
                num_repeat_model=cot_module_input_encoder_num_repeat,
                num_layers=cot_module_input_encoder_num_layers,
                num_heads=num_heads,
                emb_dim=emb_dim,
                mlp_dim_factor=mlp_dim_factor,
                max_len=seq_length,
                dropout_rate=all_dropouts_rate,
                attention_dropout_rate=all_dropouts_rate,
            ),
            cot_cross_transformer_config=TransformerConfig(
                vocab_size=None,
                output_vocab_size=None,
                num_repeat_model=cot_module_cross_transformer_num_repeat,
                num_layers=cot_module_cross_transformer_num_layers,
                num_heads=num_heads,
                emb_dim=emb_dim,
                mlp_dim_factor=mlp_dim_factor,
                max_len=None,
                dropout_rate=all_dropouts_rate,
                attention_dropout_rate=all_dropouts_rate,
            ),
            cot_seq_length=cot_seq_length,
            cot_vocab_size=cot_vocab_size,
        )
    else:
        if mode != MODE.SUPERVISED:
            raise ValueError(
                f"Only SUPERVISED mode supports cot_module to be False. Got mode: {mode} "
                f"and cot_module: {cot_module}."
            )
        cot_module_config = None

    encoder_config = EncoderConfig(
        cot_transformer_config=TransformerConfig(
            vocab_size=cot_vocab_size,
            output_vocab_size=None,
            num_repeat_model=encoder_cot_encoder_num_repeat,
            num_layers=encoder_cot_encoder_num_layers,
            num_heads=num_heads,
            emb_dim=emb_dim,
            mlp_dim_factor=mlp_dim_factor,
            max_len=cot_seq_length,
            dropout_rate=all_dropouts_rate,
            attention_dropout_rate=all_dropouts_rate,
        ),
        cot_seq_length=cot_seq_length,
        cot_vocab_size=cot_vocab_size,
        input_cross_transformer_config=TransformerConfig(
            vocab_size=seq_length,
            output_vocab_size=seq_length,
            num_repeat_model=encoder_cross_transformer_num_repeat,
            num_layers=encoder_cross_transformer_num_layers,
            num_heads=num_heads,
            emb_dim=emb_dim,
            mlp_dim_factor=mlp_dim_factor,
            max_len=seq_length,
            dropout_rate=all_dropouts_rate,
            attention_dropout_rate=all_dropouts_rate,
        ),
    )
    model = AugmentedTransformer(
        cot_module_config,
        encoder_config,
    )

    if env_name == "C_VPR":
        env = C_VPR(seq_length)
    elif env_name == "Cycle":
        env = Cycle(seq_length)  # type: ignore
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    config = {}
    for c in [cot_module_config, encoder_config]:
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
    wandb.config.cot_module = cot_module
    wandb.config.seed = seed
    wandb.config.mode = mode.name
    wandb.config.cot_loss_weight_mixing = cot_loss_weight_mixing
    wandb.config.rl_loss_weight_mixing = rl_loss_weight_mixing
    wandb.config.decode_from_sampled_cot_tokens = decode_from_sampled_cot_tokens

    trainer = Trainer(
        model,
        env,
        mode,
        train_num_hops,
        eval_num_hops,
        seq_length,
        batch_size,
        eval_size,
        cot_start_token=cot_module_config.cot_vocab_size if cot_module_config else None,
        cot_loss_weight_mixing=cot_loss_weight_mixing,
        rl_loss_weight_mixing=rl_loss_weight_mixing,
        rl_baseline_batch_size=rl_baseline_batch_size,
        decode_from_sampled_cot_tokens=decode_from_sampled_cot_tokens,
    )
    key = jax.random.PRNGKey(seed)
    state = trainer.init_train_state(key, learning_rate)
    state = trainer.train(state, key, num_iterations, log_every)
    trainer.save_checkpoint("checkpoint.msgpack", state, iteration=num_iterations)
    wandb.finish()


if __name__ == "__main__":
    # Selected C_VPR difficulties: [5-150, 10-300, 20-600]
    # Selected Cycle difficulties: []
    run_augmented_transformer_exp(
        env_name="Cycle",
        mode=MODE.RL,
        train_num_hops=3,
        seq_length=40,
        cot_module=True,
        cot_seq_length=3,
        cot_vocab_size=40,
        batch_size=256,
        log_every=500,
        num_iterations=500_000,
        rl_baseline_batch_size=8,
        run_name="Cycle 3-40 RL_mode AT1 baseline_8",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        mode=MODE.RL,
        train_num_hops=3,
        seq_length=40,
        cot_module=True,
        cot_seq_length=3,
        cot_vocab_size=40,
        batch_size=256,
        log_every=500,
        num_iterations=500_000,
        rl_baseline_batch_size=2,
        run_name="Cycle 3-40 RL_mode AT1 baseline_2",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        mode=MODE.RL,
        train_num_hops=3,
        seq_length=40,
        cot_module=True,
        cot_seq_length=3,
        cot_vocab_size=40,
        batch_size=256,
        log_every=500,
        num_iterations=500_000,
        rl_baseline_batch_size=None,
        run_name="Cycle 3-40 RL_mode AT1 no_baseline",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        mode=MODE.RL,
        train_num_hops=3,
        seq_length=40,
        cot_module=True,
        cot_seq_length=3,
        cot_vocab_size=40,
        batch_size=256,
        log_every=500,
        num_iterations=500_000,
        rl_baseline_batch_size=32,
        run_name="Cycle 3-40 RL_mode AT1 baseline_32",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        mode=MODE.RL,
        train_num_hops=3,
        seq_length=40,
        cot_module=True,
        cot_seq_length=3,
        cot_vocab_size=40,
        batch_size=256,
        log_every=500,
        num_iterations=500_000,
        rl_baseline_batch_size=4,
        run_name="Cycle 3-40 RL_mode AT1 baseline_4",
    )

    import sys

    sys.exit()
    # run_augmented_transformer_exp(
    #     env_name="Cycle",
    #     mode=MODE.SUPERVISED,
    #     train_num_hops=3,
    #     seq_length=40,
    #     cot_module=False,
    #     cot_seq_length=3,
    #     cot_vocab_size=40,
    #     batch_size=256,
    #     log_every=500,
    #     num_iterations=100_000,
    #     run_name=f"Cycle 3-40 SUPERVISED_mode T1 seed_{seed}",
    # )
    # run_augmented_transformer_exp(
    #     env_name="Cycle",
    #     mode=MODE.SUPERVISED,
    #     train_num_hops=3,
    #     seq_length=40,
    #     encoder_cross_transformer_num_layers=2,
    #     cot_module=False,
    #     cot_seq_length=3,
    #     cot_vocab_size=40,
    #     batch_size=256,
    #     log_every=500,
    #     num_iterations=100_000,
    #     run_name=f"Cycle 3-40 SUPERVISED_mode T2 seed_{seed}",
    # )
    # run_augmented_transformer_exp(
    #     env_name="Cycle",
    #     mode=MODE.SUPERVISED,
    #     train_num_hops=3,
    #     seq_length=40,
    #     encoder_cross_transformer_num_layers=3,
    #     cot_module=False,
    #     cot_seq_length=3,
    #     cot_vocab_size=40,
    #     batch_size=256,
    #     log_every=500,
    #     num_iterations=100_000,
    #     run_name=f"Cycle 3-40 SUPERVISED_mode T3 seed_{seed}",
    # )
    for seed in range(3):
        run_augmented_transformer_exp(
            env_name="Cycle",
            mode=MODE.SUPERVISED,
            train_num_hops=4,
            seq_length=40,
            encoder_cross_transformer_num_layers=1,
            cot_module=False,
            cot_seq_length=4,
            cot_vocab_size=40,
            batch_size=256,
            log_every=500,
            num_iterations=1_000_000,
            seed=seed,
            run_name=f"Cycle 4-40 SUPERVISED_mode T1 seed_{seed}",
        )
        run_augmented_transformer_exp(
            env_name="Cycle",
            mode=MODE.RL,
            train_num_hops=4,
            seq_length=40,
            cot_module=True,
            cot_seq_length=4,
            cot_vocab_size=40,
            batch_size=256,
            log_every=500,
            num_iterations=1_000_000,
            seed=seed,
            run_name=f"Cycle 4-40 RL_mode AT1 seed_{seed}",
        )
        run_augmented_transformer_exp(
            env_name="Cycle",
            mode=MODE.COT,
            train_num_hops=4,
            seq_length=40,
            cot_module=True,
            cot_seq_length=4,
            cot_vocab_size=40,
            batch_size=256,
            log_every=500,
            num_iterations=1_000_000,
            seed=seed,
            run_name=f"Cycle 4-40 COT_mode AT1 seed_{seed}",
        )
        run_augmented_transformer_exp(
            env_name="Cycle",
            mode=MODE.SUPERVISED,
            train_num_hops=4,
            seq_length=40,
            encoder_cross_transformer_num_layers=2,
            cot_module=False,
            cot_seq_length=4,
            cot_vocab_size=40,
            batch_size=256,
            log_every=500,
            num_iterations=1_000_000,
            seed=seed,
            run_name=f"Cycle 4-40 SUPERVISED_mode T2 seed_{seed}",
        )
        run_augmented_transformer_exp(
            env_name="Cycle",
            mode=MODE.RL,
            train_num_hops=4,
            seq_length=40,
            encoder_cross_transformer_num_layers=2,
            cot_module=True,
            cot_seq_length=4,
            cot_vocab_size=40,
            batch_size=256,
            log_every=500,
            num_iterations=1_000_000,
            seed=seed,
            run_name=f"Cycle 4-40 RL_mode AT2 seed_{seed}",
        )
        run_augmented_transformer_exp(
            env_name="Cycle",
            mode=MODE.SUPERVISED,
            train_num_hops=4,
            seq_length=40,
            encoder_cross_transformer_num_layers=3,
            cot_module=False,
            cot_seq_length=4,
            cot_vocab_size=40,
            batch_size=256,
            log_every=500,
            num_iterations=1_000_000,
            seed=seed,
            run_name=f"Cycle 4-40 SUPERVISED_mode T3 seed_{seed}",
        )

    # run_augmented_transformer_exp(
    #     env_name="Cycle",
    #     mode=MODE.COT,
    #     train_num_hops=3,
    #     seq_length=40,
    #     encoder_cross_transformer_num_layers=2,
    #     cot_module=True,
    #     cot_seq_length=3,
    #     cot_vocab_size=40,
    #     batch_size=256,
    #     log_every=500,
    #     num_iterations=100_000,
    #     run_name=f"Cycle 3-40 COT_mode AT2 seed_{seed}",
    # )
    # run_augmented_transformer_exp(
    #     env_name="Cycle",
    #     mode=MODE.COT,
    #     train_num_hops=3,
    #     seq_length=40,
    #     encoder_cross_transformer_num_layers=3,
    #     cot_module=True,
    #     cot_seq_length=3,
    #     cot_vocab_size=40,
    #     batch_size=256,
    #     log_every=500,
    #     num_iterations=100_000,
    #     run_name=f"Cycle 3-40 COT_mode AT3 seed_{seed}",
    # )
