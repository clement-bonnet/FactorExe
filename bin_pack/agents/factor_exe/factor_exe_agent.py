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

from typing import Any, Callable, Dict, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jumanji.env import Environment
from jumanji.environments.packing.bin_pack import Observation as BinPackObservation
from jumanji.environments.packing.bin_pack import State as BinPackState
from jumanji.environments.packing.bin_pack.space import Space
from jumanji.environments.packing.bin_pack.types import (
    EMS,
    Location,
    item_from_space,
    location_from_space,
)
from jumanji.training.agents.base import Agent
from jumanji.training.networks.parametric_distribution import CategoricalDistribution
from jumanji.training.types import ActingState, TrainingState
from networks.actor_factor_exe import ActorFactorExeNetworks
from training_types import ParamsState, Transition


class FactorExeAgent(Agent):
    def __init__(
        self,
        env: Environment,
        n_steps: int,
        total_batch_size: int,
        actor_factor_exe_networks: ActorFactorExeNetworks,
        optimizer: optax.GradientTransformation,
        factor_iterations: int,
        reinforce_loss_coeff: float,
        reinforce_estimators: int,
        kl_loss_which_factor: str,
        factors_entropy_coeff: float,
        reinforce_which_factor: str,
    ) -> None:
        super().__init__(total_batch_size=total_batch_size)
        self.env = env
        self.observation_spec = env.observation_spec()
        self.n_steps = n_steps
        self.actor_factor_exe_networks = actor_factor_exe_networks
        self.optimizer = optimizer
        self.factor_iterations = factor_iterations
        self.reinforce_loss_coeff = reinforce_loss_coeff
        self.reinforce_estimators = reinforce_estimators
        self.kl_loss_which_factor = kl_loss_which_factor
        self.factors_entropy_coeff = factors_entropy_coeff
        self.reinforce_which_factor = reinforce_which_factor

    def init_params(self, key: chex.PRNGKey) -> ParamsState:
        dummy_obs = jax.tree_util.tree_map(
            lambda x: x[None, ...], self.observation_spec.generate_value()
        )  # Add batch dim
        keys = jax.random.split(key, 1)
        params = self.actor_factor_exe_networks.policy_network.init(
            key, dummy_obs, keys
        )
        opt_state = self.optimizer.init(params)
        params_state = ParamsState(
            params=params,
            opt_state=opt_state,
            update_count=jnp.array(0, float),
        )
        return params_state

    def run_epoch(self, training_state: TrainingState) -> Tuple[TrainingState, Dict]:
        if not isinstance(training_state.params_state, ParamsState):
            raise TypeError(
                "Expected params_state to be of type ParamsState, got "
                f"type {type(training_state.params_state)}."
            )
        grad, (acting_state, metrics) = jax.grad(self.factor_exe_loss, has_aux=True)(
            training_state.params_state.params,
            training_state.acting_state,
        )
        grad, metrics = jax.lax.pmean((grad, metrics), "devices")
        updates, opt_state = self.optimizer.update(
            grad, training_state.params_state.opt_state
        )
        params = optax.apply_updates(training_state.params_state.params, updates)
        training_state = TrainingState(
            params_state=ParamsState(
                params=params,
                opt_state=opt_state,
                update_count=training_state.params_state.update_count + 1,
            ),
            acting_state=acting_state,
        )
        return training_state, metrics

    def factor_exe_loss(  # noqa: CCR001
        self,
        params: hk.Params,
        acting_state: ActingState,
    ) -> Tuple[float, Tuple[ActingState, Dict]]:
        parametric_action_distribution = (
            self.actor_factor_exe_networks.parametric_action_distribution
        )

        acting_state, data = self.rollout(acting_state)  # [T, B, ...]
        key, entropy_key = jax.random.split(acting_state.key, 2)
        keys = jax.random.split(
            key, self.reinforce_estimators * self.n_steps * self.batch_size_per_device
        ).reshape(
            self.reinforce_estimators, self.n_steps, self.batch_size_per_device, -1
        )

        logits, factors, factors_logits = jax.vmap(
            jax.vmap(
                self.actor_factor_exe_networks.policy_network.apply,
                in_axes=(None, 0, 0),
            ),
            in_axes=(None, None, 0),
        )(params, data.observation, keys)

        # Compute the entropy.
        entropy = jnp.mean(parametric_action_distribution.entropy(logits, entropy_key))
        target_entropy = jnp.mean(
            parametric_action_distribution.entropy(data.target_logits, entropy_key)
        )
        num_optimal_actions = jnp.mean(jnp.sum(data.target_logits == 0, axis=-1))

        metrics: Dict = {}
        # Compute the KL loss.
        kl_losses = CategoricalDistribution(data.target_logits).kl_divergence(
            CategoricalDistribution(logits)
        )
        if self.kl_loss_which_factor == "all":
            kl_loss = jnp.mean(kl_losses)
        elif self.kl_loss_which_factor == "one":
            kl_loss = jnp.mean(kl_losses[0])
        elif self.kl_loss_which_factor == "best":
            best_indices = jnp.argmin(kl_losses, axis=0)
            kl_loss = jnp.mean(
                jnp.take_along_axis(kl_losses, best_indices[None], axis=0)
            )
        else:
            raise ValueError

        # Compute the reinforce loss.
        if self.reinforce_estimators > 1:
            kl_losses = kl_losses - kl_losses.mean(axis=0, keepdims=True)
        factors_log_prob = CategoricalDistribution(factors_logits).log_prob(
            jax.lax.stop_gradient(factors)
        )

        # Log empirical factor entropy
        x = jnp.bincount(factors[0].reshape(-1), length=factors_logits.shape[-1])
        p = x / (len(factors[0].reshape(-1)))
        factors_empirical_entropy = -jnp.sum(jnp.where(p == 0, 0.0, p * jnp.log(p)))

        reinforce_losses = (
            jax.lax.stop_gradient(kl_losses)[..., None] * factors_log_prob
        )
        # Sum over factors (i.e. "episodes")
        reinforce_losses = jnp.sum(reinforce_losses, axis=-1)
        factors_entropies = CategoricalDistribution(factors_logits).entropy()
        if self.reinforce_which_factor == "all":
            reinforce_loss = jnp.mean(reinforce_losses)
        elif self.reinforce_which_factor == "one":
            reinforce_loss = jnp.mean(reinforce_losses[0])
            factors_entropies = factors_entropies[0]
        elif self.reinforce_which_factor == "best":
            best_indices = jnp.argmin(kl_losses, axis=0)
            reinforce_losses = jnp.take_along_axis(
                reinforce_losses, best_indices[None], axis=0
            )
            factors_entropies = jnp.take_along_axis(
                factors_entropies, best_indices[None, ..., None], axis=0
            )
            reinforce_loss = jnp.mean(reinforce_losses)
        else:
            raise ValueError

        metrics.update(
            **{
                f"factor_{i}_entropy": jnp.mean(factors_entropies[..., i])
                for i in range(factors_entropies.shape[-1])
            }
        )
        factors_entropy = jnp.mean(factors_entropies)
        if self.factors_entropy_coeff:
            reinforce_loss += self.factors_entropy_coeff * (-factors_entropy)

        total_loss = kl_loss + self.reinforce_loss_coeff * reinforce_loss

        metrics.update(
            kl_loss=kl_loss,
            reinforce_loss=reinforce_loss,
            total_loss=total_loss,
            entropy=entropy,
            factors_entropy=factors_entropy,
            factors_empirical_entropy=factors_empirical_entropy,
            target_entropy=target_entropy,
            num_optimal_actions=num_optimal_actions,
        )
        if data.extras:
            metrics.update(data.extras)
        return total_loss, (acting_state, metrics)

    def make_policy(
        self,
        params: hk.Params,
        stochastic: bool = True,
    ) -> Callable[
        [Any, chex.PRNGKey], Tuple[chex.Array, Tuple[chex.Array, chex.Array]]
    ]:
        policy_network = self.actor_factor_exe_networks.policy_network
        parametric_action_distribution = (
            self.actor_factor_exe_networks.parametric_action_distribution
        )

        @jax.vmap
        def policy(
            observation: Any, key: chex.PRNGKey
        ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
            forward_key, key = jax.random.split(key)
            observation, forward_key = jax.tree_map(
                lambda x: x[None], (observation, forward_key)
            )
            logits, *_ = policy_network.apply(params, observation, forward_key)
            logits = jnp.squeeze(logits, axis=0)
            if stochastic:
                raw_action = parametric_action_distribution.sample_no_postprocessing(
                    logits, key
                )
                log_prob = parametric_action_distribution.log_prob(logits, raw_action)
            else:
                del key
                raw_action = parametric_action_distribution.mode_no_postprocessing(
                    logits
                )
                # log_prob is log(1), i.e. 0, for a greedy policy (deterministic distribution).
                log_prob = jnp.zeros_like(
                    parametric_action_distribution.log_prob(logits, raw_action)
                )
            action = parametric_action_distribution.postprocess(raw_action)
            return action, (log_prob, logits)

        return policy  # type: ignore

    def make_target_policy(
        self,
        stochastic: bool = True,
    ) -> Callable[
        [Any, BinPackState, chex.PRNGKey],
        Tuple[chex.Array, Tuple[chex.Array, chex.Array]],
    ]:
        parametric_action_distribution = (
            self.actor_factor_exe_networks.parametric_action_distribution
        )

        @jax.vmap
        def policy(
            observation: BinPackObservation,
            bin_pack_solution: BinPackState,
            key: chex.PRNGKey,
        ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
            def unnormalize_obs_ems(obs_ems: Space, container: Space) -> Space:
                x_len, y_len, z_len = item_from_space(container)
                norm_space = Space(
                    x1=x_len, x2=x_len, y1=y_len, y2=y_len, z1=z_len, z2=z_len
                )
                obs_ems: Space = jax.tree_util.tree_map(
                    lambda x, c: jnp.round(x * c).astype(jnp.int32),
                    obs_ems,
                    norm_space,
                )
                return obs_ems

            def is_optimal_action(
                ems: EMS, solution_item_location: Location
            ) -> chex.Array:
                ems = unnormalize_obs_ems(ems, bin_pack_solution.container)
                ems_location = location_from_space(ems)
                return (
                    (ems_location.x == solution_item_location.x)
                    & (ems_location.y == solution_item_location.y)
                    & (ems_location.z == solution_item_location.z)
                )

            actions_are_optimal = (
                jax.vmap(
                    jax.vmap(is_optimal_action, in_axes=(None, 0)), in_axes=(0, None)
                )(observation.ems, bin_pack_solution.items_location)
                & observation.action_mask
            )

            logits = jnp.where(actions_are_optimal, 0.0, jnp.finfo(jnp.float32).min)
            logits = logits.reshape(*logits.shape[:-2], -1)

            if stochastic:
                raw_action = parametric_action_distribution.sample_no_postprocessing(
                    logits, key
                )
                log_prob = parametric_action_distribution.log_prob(logits, raw_action)
            else:
                del key
                raw_action = parametric_action_distribution.mode_no_postprocessing(
                    logits
                )
                # log_prob is log(1), i.e. 0, for a greedy policy (deterministic distribution).
                log_prob = jnp.zeros_like(
                    parametric_action_distribution.log_prob(logits, raw_action)
                )
            action = parametric_action_distribution.postprocess(raw_action)
            return action, (log_prob, logits)

        return policy  # type: ignore

    def rollout(
        self,
        acting_state: ActingState,
    ) -> Tuple[ActingState, Transition]:
        """Rollout for training purposes.
        Returns:
            shape (n_steps, batch_size_per_device, *)
        """
        target_policy = self.make_target_policy(stochastic=True)

        def run_one_step(
            acting_state: ActingState, key: chex.PRNGKey
        ) -> Tuple[ActingState, Transition]:
            timestep = acting_state.timestep
            bin_pack_solution = timestep.extras["bin_pack_solution"]
            batch_size = timestep.reward.shape[0]
            keys = jax.random.split(key, batch_size)
            action, (_, target_logits) = target_policy(
                timestep.observation, bin_pack_solution, keys
            )
            next_env_state, next_timestep = self.env.step(acting_state.state, action)

            acting_state = ActingState(
                state=next_env_state,
                timestep=next_timestep,
                key=key,
                episode_count=acting_state.episode_count,
                env_step_count=acting_state.env_step_count,
            )

            transition = Transition(
                observation=timestep.observation,
                action=action,
                reward=next_timestep.reward,
                logits=None,
                target_logits=target_logits,
                extras=next_timestep.extras,
            )

            return acting_state, transition

        acting_keys = jax.random.split(acting_state.key, self.n_steps).reshape(
            (self.n_steps, -1)
        )
        acting_state, data = jax.lax.scan(run_one_step, acting_state, acting_keys)
        return acting_state, data
