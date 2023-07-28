# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import optax
from jumanji.env import Environment
from jumanji.environments import BinPack
from jumanji.environments.packing.bin_pack import generator as bin_pack_generator
from jumanji.training.agents.base import Agent
from jumanji.training.agents.random import RandomAgent
from jumanji.training.loggers import Logger, NoOpLogger, TensorboardLogger
from jumanji.training.networks.protocols import RandomPolicy
from jumanji.training.types import ActingState, TrainingState
from omegaconf import DictConfig

from agents.factor_exe import FactorExeAgent
from agents.pd import PDAgent
from evaluator import Evaluator
from loggers import NeptuneLogger, TerminalLogger
from networks import (
    make_actor_factor_exe_networks_bin_pack,
    make_actor_networks_bin_pack,
    make_random_policy_bin_pack,
)
from networks.actor import ActorNetworks
from networks.actor_factor_exe import ActorFactorExeNetworks
from wrapper import BinPackSolutionWrapper, VmapAutoResetWrapperBinPackSolution


def setup_logger(cfg: DictConfig) -> Logger:
    logger: Logger
    # Log only once if there are multiple hosts on the pod.
    if jax.process_index() != 0:
        return NoOpLogger()
    if cfg.logger.type == "tensorboard":
        logger = TensorboardLogger(
            name=cfg.logger.name, save_checkpoint=cfg.logger.save_checkpoint
        )
    elif cfg.logger.type == "neptune":
        logger = NeptuneLogger(
            name=cfg.logger.name,
            project="InstaDeep/metal",
            cfg=cfg,
            save_checkpoint=cfg.logger.save_checkpoint,
        )
    elif cfg.logger.type == "terminal":
        logger = TerminalLogger(
            name=cfg.logger.name, save_checkpoint=cfg.logger.save_checkpoint
        )
    else:
        raise ValueError(
            f"logger expected in ['neptune', 'tensorboard', 'terminal'], got {cfg.logger}."
        )
    return logger


def _make_raw_env(cfg: DictConfig) -> Environment:
    assert cfg.env.name == "bin_pack"
    generator = bin_pack_generator.RandomGenerator(
        max_num_items=cfg.env.kwargs.max_num_items,
        max_num_ems=cfg.env.kwargs.max_num_ems,
        split_num_same_items=cfg.env.kwargs.split_num_same_items,
    )
    env = BinPackSolutionWrapper(
        generator=generator, obs_num_ems=cfg.env.kwargs.obs_num_ems
    )
    return env


def setup_env(cfg: DictConfig) -> Environment:
    env = _make_raw_env(cfg)
    env = VmapAutoResetWrapperBinPackSolution(env)
    return env


def setup_agent(cfg: DictConfig, env: Environment) -> Agent:
    agent: Agent
    if cfg.agent == "random":
        random_policy = _setup_random_policy(cfg, env)
        agent = RandomAgent(
            env=env,
            n_steps=cfg.env.training.n_steps,
            total_batch_size=cfg.env.training.total_batch_size,
            random_policy=random_policy,
        )
    elif cfg.agent == "pd":
        actor_networks = _setup_actor_neworks(cfg, env)
        optimizer = optax.adam(cfg.env.agent.learning_rate)
        agent = PDAgent(
            env=env,
            n_steps=cfg.env.training.n_steps,
            total_batch_size=cfg.env.training.total_batch_size,
            actor_networks=actor_networks,
            optimizer=optimizer,
        )
    elif cfg.agent == "factor_exe":
        actor_factor_exe_networks = _setup_actor_factor_exe_neworks(cfg, env)
        every_k_schedule = cfg.env.training.every_k_schedule
        optimizer = optax.MultiSteps(
            optax.adam(cfg.env.agent.learning_rate), every_k_schedule=every_k_schedule
        )
        # Divide the total batch size by the number of multi-steps.
        assert cfg.env.training.total_batch_size % every_k_schedule == 0
        total_batch_size = cfg.env.training.total_batch_size // every_k_schedule
        agent = FactorExeAgent(
            env=env,
            n_steps=cfg.env.training.n_steps,
            total_batch_size=total_batch_size,
            actor_factor_exe_networks=actor_factor_exe_networks,
            optimizer=optimizer,
            factor_iterations=cfg.env.network.factor_iterations,
            reinforce_loss_coeff=cfg.env.agent.reinforce_loss_coeff,
            reinforce_estimators=cfg.env.agent.reinforce_estimators,
            use_one_kl_loss_estimator=cfg.env.agent.use_one_kl_loss_estimator,
            factors_entropy_coeff=cfg.env.agent.factors_entropy_coeff,
        )
    else:
        raise ValueError(
            f"Expected agent name to be in ['random', 'pd', 'factor_exe'], got {cfg.agent}."
        )
    return agent


def _setup_random_policy(cfg: DictConfig, env: Environment) -> RandomPolicy:
    assert cfg.agent == "random"
    if cfg.env.name == "bin_pack":
        assert isinstance(env.unwrapped, BinPack)
        random_policy = make_random_policy_bin_pack(bin_pack=env.unwrapped)
    else:
        raise ValueError(f"Environment name not found. Got {cfg.env.name}.")
    return random_policy


def _setup_actor_neworks(cfg: DictConfig, env: Environment) -> ActorNetworks:
    assert cfg.agent == "pd"
    if cfg.env.name == "bin_pack":
        assert isinstance(env.unwrapped, BinPack)
        actor_networks = make_actor_networks_bin_pack(
            bin_pack=env.unwrapped,
            num_transformer_layers=cfg.env.network.num_transformer_layers,
            transformer_num_heads=cfg.env.network.transformer_num_heads,
            transformer_key_size=cfg.env.network.transformer_key_size,
            transformer_mlp_units=cfg.env.network.transformer_mlp_units,
        )
    else:
        raise ValueError(f"Environment name not found. Got {cfg.env.name}.")
    return actor_networks


def _setup_actor_factor_exe_neworks(
    cfg: DictConfig, env: Environment
) -> ActorFactorExeNetworks:
    assert cfg.agent == "factor_exe"
    if cfg.env.name == "bin_pack":
        assert isinstance(env.unwrapped, BinPack)
        actor_networks = make_actor_factor_exe_networks_bin_pack(
            bin_pack=env.unwrapped,
            num_transformer_layers=cfg.env.network.num_transformer_layers,
            transformer_num_heads=cfg.env.network.transformer_num_heads,
            transformer_key_size=cfg.env.network.transformer_key_size,
            transformer_mlp_units=cfg.env.network.transformer_mlp_units,
            factor_iterations=cfg.env.network.factor_iterations,
            factor_vocab_size=cfg.env.network.factor_vocab_size,
        )
    else:
        raise ValueError(f"Environment name not found. Got {cfg.env.name}.")
    return actor_networks


def setup_evaluators(cfg: DictConfig, agent: Agent) -> Tuple[Evaluator, Evaluator]:
    env = _make_raw_env(cfg)
    stochastic_eval = Evaluator(
        eval_env=env,
        agent=agent,
        total_batch_size=cfg.env.evaluation.eval_total_batch_size,
        stochastic=True,
    )
    greedy_eval = Evaluator(
        eval_env=env,
        agent=agent,
        total_batch_size=cfg.env.evaluation.greedy_eval_total_batch_size,
        stochastic=False,
    )
    return stochastic_eval, greedy_eval


def setup_training_state(
    env: Environment, agent: Agent, key: chex.PRNGKey
) -> TrainingState:
    params_key, reset_key, acting_key = jax.random.split(key, 3)

    # Initialize params.
    params_state = agent.init_params(params_key)

    # Initialize environment states.
    num_local_devices = jax.local_device_count()
    num_global_devices = jax.device_count()
    num_workers = num_global_devices // num_local_devices
    local_batch_size = agent.total_batch_size // num_global_devices
    reset_keys = jax.random.split(reset_key, agent.total_batch_size).reshape(
        (
            num_workers,
            num_local_devices,
            local_batch_size,
            -1,
        )
    )
    reset_keys_per_worker = reset_keys[jax.process_index()]
    env_state, timestep = jax.pmap(env.reset, axis_name="devices")(
        reset_keys_per_worker
    )

    # Initialize acting states.
    acting_key_per_device = jax.random.split(acting_key, num_global_devices).reshape(
        num_workers, num_local_devices, -1
    )
    acting_key_per_worker_device = acting_key_per_device[jax.process_index()]
    acting_state = ActingState(
        state=env_state,
        timestep=timestep,
        key=acting_key_per_worker_device,
        episode_count=jnp.zeros(num_local_devices, float),
        env_step_count=jnp.zeros(num_local_devices, float),
    )

    # Build the training state.
    training_state = TrainingState(
        params_state=jax.device_put_replicated(params_state, jax.local_devices()),
        acting_state=acting_state,
    )
    return training_state
