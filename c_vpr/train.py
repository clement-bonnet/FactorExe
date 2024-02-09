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
from PIL import Image, ImageDraw
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
        use_poppy: bool = False,
        poppy_size: Optional[int] = None,
        poppy_train_encoder_on_best_cot: bool = False,
        poppy_train_cot_module_using_poppy: bool = False,
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
        self.use_poppy = use_poppy
        self.poppy_size = poppy_size
        self.poppy_train_encoder_on_best_cot = poppy_train_encoder_on_best_cot
        self.poppy_train_cot_module_using_poppy = poppy_train_cot_module_using_poppy
        self.decode_from_sampled_cot_tokens = decode_from_sampled_cot_tokens

    def init_train_state(
        self,
        key: chex.PRNGKey,
        learning_rate: float,
        verbose: bool = True,
    ) -> TrainState:
        inputs = jnp.zeros((1, self.seq_length), int)
        params = self.model.init(key, inputs=inputs, deterministic=True, num_hops=jnp.array([1]))[
            "params"
        ]
        if verbose:
            num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
            logging.info("Number of parameters: {:,}".format(num_params))
        warmup_steps = 99
        linear_warmup_scheduler = optax.warmup_exponential_decay_schedule(
            init_value=learning_rate / (warmup_steps + 1),
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            transition_steps=1,
            end_value=learning_rate,
            decay_rate=1.0,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adamw(linear_warmup_scheduler)
        )
        apply_fn = jax.jit(
            self.model.apply, static_argnames=["deterministic", "cot_sampling", "method"]
        )
        return TrainState.create(apply_fn=apply_fn, tx=optimizer, params=params)

    def _sample_n_hops_for_train(
        self, key: chex.PRNGKey, num_hops: Optional[int] = None, return_cot: bool = False
    ) -> tuple[chex.Array, ...]:
        if len(self.train_num_hops) == 1:
            return self.env.sample_n_hops(
                key, self.train_num_hops[0], return_cot=return_cot, return_target=True
            )
        else:
            assert num_hops is not None
            if return_cot:
                sequence, cot, target = jax.lax.switch(
                    jnp.argmax(num_hops == jnp.asarray(self.train_num_hops)),
                    [
                        functools.partial(
                            self.env.sample_n_hops,
                            num_hops=n_hops,
                            return_cot=return_cot,
                            return_target=True,
                            cot_pading_length=max(self.train_num_hops) - n_hops,
                            cot_pading_value=self.cot_start_token,
                        )
                        for n_hops in self.train_num_hops
                    ],
                    key,
                )
                return sequence, cot, target
            else:
                sequence, target = self.env.sample_n_hops(
                    key=key,
                    num_hops=num_hops,
                    return_cot=return_cot,
                    return_target=True,
                )
                return sequence, target

    def train_step_supervised(
        self, state: TrainState, key: chex.PRNGKey
    ) -> tuple[TrainState, dict]:
        num_hops_key, sample_key, dropout_key = jax.random.split(key, 3)

        num_hops = jax.random.choice(
            num_hops_key,
            jnp.asarray(self.train_num_hops),
            (self.batch_size,),
            replace=True,
        )
        sample_keys = jax.random.split(sample_key, self.batch_size)
        inputs, labels = jax.vmap(self._sample_n_hops_for_train)(sample_keys, num_hops)

        def loss_fn(params: dict, dropout_key: chex.PRNGKey) -> tuple[TrainState, chex.Array]:
            logits, _ = state.apply_fn(
                variables={"params": params},
                inputs=inputs,
                num_hops=num_hops,
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

        num_hops = jax.random.choice(
            num_hops_key,
            jnp.asarray(self.train_num_hops),
            (self.batch_size,),
            replace=True,
        )
        sample_keys = jax.random.split(sample_key, self.batch_size)
        inputs, cots, labels = jax.vmap(
            functools.partial(self._sample_n_hops_for_train, return_cot=True)
        )(sample_keys, num_hops)

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
                num_hops=num_hops,
                inputs_pad_mask=None,
                rngs={"dropout": drop_key1},
                method=self.model.cot_module_generate_cot_logits,
            )
            cot_entropy = -jnp.mean(
                jnp.sum(jax.nn.log_softmax(cot_logits) * jax.nn.softmax(cot_logits), -1)
            )
            cot_loss = cross_entropy_loss(logits=cot_logits, labels=cots)
            if self.decode_from_sampled_cot_tokens:
                cot_tokens, _ = state.apply_fn(
                    variables={"params": params},
                    inputs=inputs,
                    deterministic=False,
                    num_hops=num_hops,
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
                num_hops=num_hops,
                inputs_pad_mask=None,
                cot_pad_mask=None,
                rngs={"dropout": drop_key3},
                method=self.model.encoder_call,
            )
            supervised_loss = cross_entropy_loss(logits, labels)

            loss = supervised_loss + self.cot_loss_weight_mixing * cot_loss
            return loss, (logits, cot_loss, cot_entropy)

        grads, (logits, cot_loss, cot_entropy) = jax.grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = self.compute_metrics(logits, labels)
        grad_norm = jnp.sqrt(sum([jnp.sum(x**2) for x in jax.tree_util.tree_leaves(grads)]))
        metrics.update(grad_norm=grad_norm, cot_loss=cot_loss, cot_entropy=cot_entropy)
        return state, metrics

    def train_step_rl(self, state: TrainState, key: chex.PRNGKey) -> tuple[TrainState, dict]:
        num_hops_key, sample_key, cot_key, dropout_key = jax.random.split(key, 4)

        num_hops = jax.random.choice(
            num_hops_key,
            jnp.asarray(self.train_num_hops),
            (self.batch_size,),
            replace=True,
        )
        sample_keys = jax.random.split(sample_key, self.batch_size)
        inputs, labels = jax.vmap(self._sample_n_hops_for_train)(sample_keys, num_hops)

        def poppy_loss_fn(params: dict) -> tuple[TrainState, chex.Array]:
            assert self.poppy_size is not None

            def apply_to_vmap(
                inputs: chex.Array, cot_key: chex.PRNGKey
            ) -> tuple[chex.Array, tuple[chex.Array, chex.Array]]:
                return state.apply_fn(  # type: ignore
                    variables={"params": params},
                    inputs=inputs,
                    deterministic=False,
                    num_hops=num_hops,
                    cot_key=cot_key,
                    cot_sampling=True,
                    rngs={"dropout": dropout_key},
                )

            cot_keys = jax.random.split(cot_key, self.poppy_size)
            logits, (cot_tokens, cot_logits) = jax.vmap(apply_to_vmap)(
                inputs[None].repeat(self.poppy_size, axis=0), cot_keys
            )
            cot_entropy = -jnp.mean(
                jnp.sum(jax.nn.log_softmax(cot_logits) * jax.nn.softmax(cot_logits), -1)
            )
            rewards = jax.lax.stop_gradient(
                -cross_entropy_loss(
                    logits,
                    labels[None].repeat(self.poppy_size, axis=0),
                    mean=False,
                )
            )

            if self.poppy_train_encoder_on_best_cot:
                # TODO: Check if this is correct
                best_logits = jnp.take_along_axis(
                    logits, jnp.argmax(rewards, axis=0)[None, :, None], axis=0
                ).squeeze(0)
                supervised_loss = cross_entropy_loss(best_logits, labels)
            else:
                supervised_loss = cross_entropy_loss(
                    logits, labels[None].repeat(self.poppy_size, axis=0)
                )

            if self.poppy_train_cot_module_using_poppy:
                second_best_rewards, best_rewards = jnp.sort(rewards, axis=0)[-2:]
                best_logits = jnp.take_along_axis(
                    logits, jnp.argmax(rewards, axis=0)[None, :, None], axis=0
                ).squeeze(0)
                best_cot_tokens = jnp.take_along_axis(
                    cot_tokens, jnp.argmax(rewards, axis=0)[None, :, None], axis=0
                ).squeeze(0)
                best_cot_all_log_prob = jax.nn.log_softmax(best_logits, axis=-1)
                # TODO: check if this is correct
                best_cot_log_prob = jnp.take_along_axis(
                    best_cot_all_log_prob, best_cot_tokens, axis=-1
                )
                rl_loss = jnp.mean(
                    -(best_rewards - second_best_rewards)[:, None] * best_cot_log_prob
                )
            else:
                cot_all_log_prob = jax.nn.log_softmax(cot_logits, axis=-1)
                cot_log_prob = jnp.take_along_axis(
                    cot_all_log_prob, cot_tokens[..., None], axis=-1
                ).squeeze(-1)
                rl_loss = jnp.mean(-rewards[..., None] * cot_log_prob)

            loss = supervised_loss + self.rl_loss_weight_mixing * rl_loss
            return loss, (logits, rl_loss, cot_entropy)

        def loss_fn(params: dict) -> tuple[TrainState, chex.Array]:
            logits, (cot_tokens, cot_logits) = state.apply_fn(
                variables={"params": params},
                inputs=inputs,
                deterministic=False,
                num_hops=num_hops,
                cot_key=cot_key,
                cot_sampling=True,
                rngs={"dropout": dropout_key},
            )
            cot_entropy = -jnp.mean(
                jnp.sum(jax.nn.log_softmax(cot_logits) * jax.nn.softmax(cot_logits), -1)
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
                        num_hops=num_hops,
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
            return loss, (logits, rl_loss, cot_entropy)

        if self.use_poppy:
            grads, (logits, rl_loss, cot_entropy) = jax.grad(poppy_loss_fn, has_aux=True)(
                state.params
            )
        else:
            grads, (logits, rl_loss, cot_entropy) = jax.grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = self.compute_metrics(logits, labels)
        grad_norm = jnp.sqrt(sum([jnp.sum(x**2) for x in jax.tree_util.tree_leaves(grads)]))
        metrics.update(grad_norm=grad_norm, rl_loss=rl_loss, cot_entropy=cot_entropy)
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
                num_hops=jnp.full((self.eval_size,), num_hops),
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

    def generate_cot_samples(
        self, state: TrainState, key: chex.PRNGKey, num_samples: int
    ) -> dict[str, chex.Array]:
        metrics: dict[str, chex.Array] = {}
        if self.eval_num_hops is None:
            return metrics
        sample_keys = jax.random.split(key, len(self.eval_num_hops))
        for num_hops, sample_key in zip(self.eval_num_hops, sample_keys):
            keys = jax.random.split(sample_key, num_samples)
            inputs, labels = jax.vmap(
                functools.partial(self.env.sample_n_hops, num_hops=num_hops, return_target=True)
            )(keys)
            _, (cot_tokens, _) = state.apply_fn(
                variables={"params": state.params},
                inputs=inputs,
                deterministic=True,
                num_hops=jnp.full((num_samples,), num_hops),
                cot_key=sample_key,
                cot_sampling=True,
            )
            metrics.update({f"test/cot_tokens/num_hops:{num_hops}": (inputs, labels, cot_tokens)})

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
        jit_generate_cot_samples = jax.jit(
            functools.partial(self.generate_cot_samples, num_samples=5)
        )
        num_epochs = num_iterations // log_every
        for epoch in trange(1, num_epochs + 1):
            key, epoch_key, eval_key = jax.random.split(key, 3)
            state, metrics = jit_train_epoch(state, epoch_key)
            metrics.update(jit_eval(state, eval_key))
            if self.mode == MODE.RL:
                test_metrics = jit_generate_cot_samples(state, eval_key)
                for k, (inputs, labels, cot_tokens) in test_metrics.items():
                    img = Image.new("RGB", (500, 350), color=(0, 0, 0))
                    draw = ImageDraw.Draw(img)
                    draw.text(
                        (10, 10),
                        f"    inputs\n{inputs}\n\n"
                        f"    labels\n{labels}\n\n"
                        f"    cot_tokens\n{cot_tokens}",
                        fill=(255, 255, 255),
                    )
                    metrics.update({k: wandb.Image(img)})
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
            .replace("[", "_")
            .replace("]", "_")
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
            .replace("[", "_")
            .replace("]", "_")
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
    seq_length: int = 40,
    cot_module: bool = False,
    cot_seq_length: int = 3,
    cot_vocab_size: int = 3,
    cot_module_input_encoder_num_repeat: int = 0,
    cot_module_input_encoder_num_layers: int = 0,
    cot_module_cross_transformer_num_repeat: int = 1,
    cot_module_cross_transformer_num_layers: int = 1,
    encoder_cot_encoder_num_repeat: int = 0,
    encoder_cot_encoder_num_layers: int = 0,
    encoder_cross_transformer_num_repeat: int = 1,
    encoder_cross_transformer_num_layers: int = 2,
    emb_dim: int = 384,
    num_heads: int = 6,
    mlp_dim_factor: float = 4,
    all_dropouts_rate: float = 0.0,
    cot_loss_weight_mixing: float = 1.0,
    rl_loss_weight_mixing: float = 1.0,
    rl_baseline_batch_size: Optional[int] = None,
    use_poppy: bool = False,
    poppy_size: Optional[int] = None,
    poppy_train_encoder_on_best_cot: bool = False,
    poppy_train_cot_module_using_poppy: bool = False,
    decode_from_sampled_cot_tokens: bool = True,
    classification_mode: str = "cls_token",
    learning_rate: float = 3e-4,
    num_iterations: int = 100_000,
    batch_size: int = 256,
    eval_size: int = 500,
    log_every: int = 100,
    seed: int = 0,
    run_name: Optional[str] = None,
) -> None:
    if isinstance(train_num_hops, list):
        max_num_hops = max(train_num_hops)
        if isinstance(eval_num_hops, list):
            max_num_hops = max(max_num_hops, max(eval_num_hops))
        elif isinstance(eval_num_hops, int):
            max_num_hops = max(max_num_hops, eval_num_hops)
    elif isinstance(train_num_hops, int):
        max_num_hops = train_num_hops
        if isinstance(eval_num_hops, list):
            max_num_hops = max(max_num_hops, max(eval_num_hops))
        elif isinstance(eval_num_hops, int):
            max_num_hops = max(max_num_hops, eval_num_hops)
    else:
        raise ValueError(f"Unknown type for train_num_hops: {type(train_num_hops)}")

    assert classification_mode in ["cls_token", "mean_embedding"]
    if cot_module:
        if mode in [MODE.COT, MODE.RL]:
            if cot_seq_length < max_num_hops + 1:
                raise ValueError(
                    f"cot_seq_length ({cot_seq_length}) is smaller than max_num_hops + 1"
                    f"({max_num_hops} + 1), which means that the chain of thought sequence is "
                    "too small compared to the number of hops. This is not encouraged and "
                    "raises an error for now."
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
            max_num_hops=max_num_hops,
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
        classification_mode=classification_mode,
        max_num_hops=max_num_hops,
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
        use_poppy=use_poppy,
        poppy_size=poppy_size,
        poppy_train_encoder_on_best_cot=poppy_train_encoder_on_best_cot,
        poppy_train_cot_module_using_poppy=poppy_train_cot_module_using_poppy,
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
    # run_augmented_transformer_exp(
    #     env_name="Cycle",
    #     mode=MODE.RL,
    #     train_num_hops=2,
    #     eval_num_hops=2,
    #     seq_length=40,
    #     cot_module=True,
    #     encoder_cross_transformer_num_layers=1,
    #     cot_seq_length=3,
    #     cot_vocab_size=40,
    #     log_every=50,
    #     num_iterations=50_000,
    #     use_poppy=True,
    #     poppy_size=10,
    #     poppy_train_encoder_on_best_cot=True,
    #     poppy_train_cot_module_using_poppy=False,
    #     run_name="Cycle 2-40 RL T1 poppy_10 poppy_encoder",
    # )
    run_augmented_transformer_exp(
        env_name="Cycle",
        mode=MODE.RL,
        train_num_hops=2,
        eval_num_hops=2,
        seq_length=40,
        cot_module=True,
        encoder_cross_transformer_num_layers=1,
        cot_seq_length=3,
        cot_vocab_size=40,
        log_every=50,
        num_iterations=50_000,
        use_poppy=True,
        poppy_size=100,
        batch_size=32,
        poppy_train_encoder_on_best_cot=False,
        poppy_train_cot_module_using_poppy=True,
        run_name="Cycle 2-40 RL T1 poppy_100 poppy_cot_module bs_32",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        mode=MODE.RL,
        train_num_hops=2,
        eval_num_hops=2,
        seq_length=40,
        cot_module=True,
        encoder_cross_transformer_num_layers=1,
        cot_seq_length=3,
        cot_vocab_size=40,
        log_every=50,
        num_iterations=50_000,
        use_poppy=True,
        poppy_size=10,
        poppy_train_encoder_on_best_cot=False,
        poppy_train_cot_module_using_poppy=True,
        run_name="Cycle 2-40 RL T1 poppy_10 poppy_cot_module",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        mode=MODE.RL,
        train_num_hops=2,
        eval_num_hops=2,
        seq_length=40,
        cot_module=True,
        encoder_cross_transformer_num_layers=1,
        cot_seq_length=3,
        cot_vocab_size=40,
        log_every=50,
        num_iterations=50_000,
        use_poppy=True,
        poppy_size=100,
        batch_size=32,
        poppy_train_encoder_on_best_cot=True,
        poppy_train_cot_module_using_poppy=True,
        run_name="Cycle 2-40 RL T1 poppy_100 poppy_both bs_32",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        mode=MODE.RL,
        train_num_hops=2,
        eval_num_hops=2,
        seq_length=40,
        cot_module=True,
        encoder_cross_transformer_num_layers=1,
        cot_seq_length=3,
        cot_vocab_size=40,
        log_every=50,
        num_iterations=50_000,
        use_poppy=True,
        poppy_size=10,
        poppy_train_encoder_on_best_cot=True,
        poppy_train_cot_module_using_poppy=True,
        run_name="Cycle 2-40 RL T1 poppy_10 poppy_both",
    )
    # run_augmented_transformer_exp(
    #     env_name="Cycle",
    #     mode=MODE.RL,
    #     train_num_hops=2,
    #     eval_num_hops=2,
    #     seq_length=40,
    #     cot_module=True,
    #     encoder_cross_transformer_num_layers=1,
    #     cot_seq_length=3,
    #     cot_vocab_size=40,
    #     log_every=50,
    #     num_iterations=50_000,
    #     use_poppy=True,
    #     poppy_size=10,
    #     poppy_train_encoder_on_best_cot=False,
    #     poppy_train_cot_module_using_poppy=False,
    #     run_name="Cycle 2-40 RL T1 poppy_10 poppy_neither",
    # )
    # run_augmented_transformer_exp(
    #     env_name="Cycle",
    #     mode=MODE.RL,
    #     train_num_hops=2,
    #     eval_num_hops=2,
    #     seq_length=40,
    #     cot_module=True,
    #     encoder_cross_transformer_num_layers=1,
    #     cot_seq_length=3,
    #     cot_vocab_size=40,
    #     log_every=50,
    #     num_iterations=50_000,
    #     use_poppy=False,
    #     run_name="Cycle 2-40 RL T1",
    # )

    # for seed in range(3):
    #     run_augmented_transformer_exp(
    #         env_name="Cycle",
    #         mode=MODE.RL,
    #         train_num_hops=1,
    #         eval_num_hops=1,
    #         seq_length=40,
    #         cot_module=True,
    #         encoder_cross_transformer_num_layers=1,
    #         cot_seq_length=2,
    #         cot_vocab_size=40,
    #         log_every=500,
    #         num_iterations=500_000,
    #         seed=seed,
    #         run_name=f"Cycle 1-40 RL T1 long seed_{seed} with_cot_logs",
    #     )
    # run_augmented_transformer_exp(
    #     env_name="Cycle",
    #     mode=MODE.SUPERVISED,
    #     train_num_hops=1,
    #     eval_num_hops=1,
    #     seq_length=40,
    #     cot_module=False,
    #     encoder_cross_transformer_num_layers=1,
    #     cot_seq_length=2,
    #     cot_vocab_size=40,
    #     log_every=500,
    #     num_iterations=500_000,
    #     seed=seed,
    #     run_name=f"Cycle 1-40 SUPERVISED T1 long seed_{seed}",
    # )
    import sys

    sys.exit()

    run_augmented_transformer_exp(
        env_name="Cycle",
        mode=MODE.SUPERVISED,
        train_num_hops=[1, 2, 3, 4, 5],
        eval_num_hops=[1, 2, 3, 4, 5],
        seq_length=40,
        cot_module=False,
        encoder_cross_transformer_num_layers=1,
        cot_seq_length=6,
        cot_vocab_size=40,
        log_every=100,
        num_iterations=20_000,
        classification_mode="mean_embedding",
        run_name="Cycle [1,2,3,4,5]-40 SUPERVISED T1 mean_embedding",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        mode=MODE.SUPERVISED,
        train_num_hops=[1, 2, 3, 4, 5],
        eval_num_hops=[1, 2, 3, 4, 5],
        seq_length=40,
        cot_module=False,
        encoder_cross_transformer_num_layers=2,
        cot_seq_length=6,
        cot_vocab_size=40,
        log_every=100,
        num_iterations=20_000,
        classification_mode="mean_embedding",
        run_name="Cycle [1,2,3,4,5]-40 SUPERVISED T2 mean_embedding",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        mode=MODE.SUPERVISED,
        train_num_hops=[1, 2, 3, 4, 5],
        eval_num_hops=[1, 2, 3, 4, 5],
        seq_length=40,
        cot_module=False,
        encoder_cross_transformer_num_layers=3,
        cot_seq_length=6,
        cot_vocab_size=40,
        log_every=100,
        num_iterations=20_000,
        classification_mode="mean_embedding",
        run_name="Cycle [1,2,3,4,5]-40 SUPERVISED T3 mean_embedding",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        mode=MODE.COT,
        train_num_hops=[1, 2, 3, 4, 5],
        eval_num_hops=[1, 2, 3, 4, 5],
        seq_length=40,
        cot_module=True,
        encoder_cross_transformer_num_layers=1,
        cot_seq_length=6,
        cot_vocab_size=40,
        log_every=100,
        num_iterations=20_000,
        classification_mode="mean_embedding",
        run_name="Cycle [1,2,3,4,5]-40 COT T1 mean_embedding",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        mode=MODE.COT,
        train_num_hops=[1, 2, 3, 4, 5],
        eval_num_hops=[1, 2, 3, 4, 5],
        seq_length=40,
        cot_module=True,
        encoder_cross_transformer_num_layers=2,
        cot_seq_length=6,
        cot_vocab_size=40,
        log_every=100,
        num_iterations=20_000,
        classification_mode="mean_embedding",
        run_name="Cycle [1,2,3,4,5]-40 COT T2 mean_embedding",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        mode=MODE.RL,
        train_num_hops=[1, 2, 3, 4, 5],
        eval_num_hops=[1, 2, 3, 4, 5],
        seq_length=40,
        cot_module=True,
        encoder_cross_transformer_num_layers=1,
        cot_seq_length=6,
        cot_vocab_size=40,
        log_every=100,
        num_iterations=20_000,
        classification_mode="mean_embedding",
        run_name="Cycle [1,2,3,4,5]-40 RL T1 mean_embedding",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        mode=MODE.RL,
        train_num_hops=[1, 2, 3, 4, 5],
        eval_num_hops=[1, 2, 3, 4, 5],
        seq_length=40,
        cot_module=True,
        encoder_cross_transformer_num_layers=2,
        cot_seq_length=6,
        cot_vocab_size=40,
        log_every=100,
        num_iterations=20_000,
        classification_mode="mean_embedding",
        run_name="Cycle [1,2,3,4,5]-40 RL T2 mean_embedding",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        mode=MODE.RL,
        train_num_hops=[1, 2, 3, 4, 5],
        eval_num_hops=[1, 2, 3, 4, 5],
        seq_length=40,
        cot_module=True,
        encoder_cross_transformer_num_layers=3,
        cot_seq_length=6,
        cot_vocab_size=40,
        log_every=100,
        num_iterations=20_000,
        classification_mode="mean_embedding",
        run_name="Cycle [1,2,3,4,5]-40 RL T3 mean_embedding",
    )

    run_augmented_transformer_exp(
        env_name="Cycle",
        mode=MODE.RL,
        train_num_hops=[1, 2, 3, 4, 5],
        eval_num_hops=[1, 2, 3, 4, 5],
        seq_length=40,
        cot_module=True,
        encoder_cross_transformer_num_layers=1,
        cot_seq_length=6,
        cot_vocab_size=40,
        log_every=500,
        num_iterations=500_000,
        run_name="Cycle [1,2,3,4,5]-40 RL T1 long",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        mode=MODE.SUPERVISED,
        train_num_hops=[1, 2, 3, 4, 5],
        eval_num_hops=[1, 2, 3, 4, 5],
        seq_length=40,
        cot_module=False,
        encoder_cross_transformer_num_layers=1,
        cot_seq_length=6,
        cot_vocab_size=40,
        log_every=500,
        num_iterations=500_000,
        run_name="Cycle [1,2,3,4,5]-40 SUPERVISED T1 long",
    )
    run_augmented_transformer_exp(
        env_name="Cycle",
        mode=MODE.SUPERVISED,
        train_num_hops=[1, 2, 3, 4, 5],
        eval_num_hops=[1, 2, 3, 4, 5],
        seq_length=40,
        cot_module=True,
        encoder_cross_transformer_num_layers=1,
        cot_seq_length=6,
        cot_vocab_size=40,
        log_every=500,
        num_iterations=500_000,
        run_name="Cycle [1,2,3,4,5]-40 SUPERVISED T1 cot_module long",
    )
