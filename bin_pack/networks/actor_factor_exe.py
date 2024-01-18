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

from typing import NamedTuple, Optional, Sequence, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jumanji.environments.packing.bin_pack import BinPack, Observation
from jumanji.environments.packing.bin_pack.types import EMS, Item
from jumanji.training.networks.base import FeedForwardNetwork
from jumanji.training.networks.parametric_distribution import (
    FactorisedActionSpaceParametricDistribution,
    ParametricDistribution,
)
from jumanji.training.networks.transformer_block import TransformerBlock


class ActorFactorExeNetworks(NamedTuple):
    """The assumption is that the networks are given a batch of observations."""

    policy_network: FeedForwardNetwork
    parametric_action_distribution: ParametricDistribution


def make_actor_factor_exe_networks_bin_pack(
    bin_pack: BinPack,
    num_transformer_layers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
    factor_iterations: int,
    factor_vocab_size: int,
) -> ActorFactorExeNetworks:
    """Make actor networks for the `BinPack` environment."""
    num_values = np.asarray(bin_pack.action_spec().num_values)
    parametric_action_distribution = FactorisedActionSpaceParametricDistribution(
        action_spec_num_values=num_values
    )
    policy_network = make_actor_factor_exe_network_bin_pack(
        num_transformer_layers=num_transformer_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
        factor_iterations=factor_iterations,
        factor_vocab_size=factor_vocab_size,
    )
    return ActorFactorExeNetworks(
        policy_network=policy_network,
        parametric_action_distribution=parametric_action_distribution,
    )


class BinPackFactorTorso(hk.Module):
    def __init__(
        self,
        num_transformer_layers: int,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_mlp_units: Sequence[int],
        factor_iterations: int,
        factor_vocab_size: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_transformer_layers = num_transformer_layers
        self.transformer_num_heads = transformer_num_heads
        self.transformer_key_size = transformer_key_size
        self.transformer_mlp_units = transformer_mlp_units
        self.factor_iterations = factor_iterations
        self.factor_vocab_size = factor_vocab_size
        self.model_size = transformer_num_heads * transformer_key_size
        self.factor_embedding_layer = hk.Embed(
            vocab_size=self.factor_vocab_size,
            embed_dim=self.model_size,
            name="factor_embedding",
        )

    def __call__(
        self, observation: Observation, key: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        transformer_blocks = [
            TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=2 / self.num_transformer_layers,
                model_size=self.model_size,
                name=f"self_attention_block_{block_id}",
            )
            for block_id in range(self.num_transformer_layers)
        ]
        factor_embedding_projection = hk.Linear(
            self.factor_vocab_size, name="factor_embedding_projection"
        )

        # EMS encoder
        ems_mask = observation.ems_mask
        ems_embeddings = self.embed_ems(observation.ems)

        # Item encoder
        items_mask = observation.items_mask & ~observation.items_placed
        items_embeddings = self.embed_items(observation.items)

        # Joint embeddings
        embeddings = jnp.concatenate([ems_embeddings, items_embeddings], axis=-2)

        def factor_scan_fn(
            carry: Tuple[chex.Array, chex.Array], inputs: Tuple[int, chex.PRNGKey]
        ) -> Tuple[Tuple[chex.Array, chex.Array], chex.Array]:
            factors, mask = carry
            i, key = inputs
            factor_embeddings = self.embed_factors(factors)
            all_embeddings = jnp.concatenate([embeddings, factor_embeddings], axis=-2)

            mask = mask.at[..., -(self.factor_iterations - i)].set(True)
            # TODO: check that the mask is correct.
            self_attention_mask = self._make_self_attention_mask(mask)

            # Transformer encoder
            for transformer_block in transformer_blocks:
                all_embeddings = transformer_block(
                    all_embeddings, all_embeddings, all_embeddings, self_attention_mask
                )

            _, factor_embeddings = jnp.split(
                all_embeddings,
                (embeddings.shape[-2],),
                axis=-2,
            )
            factor_logits = factor_embedding_projection(factor_embeddings[..., i, :])
            factor = jax.random.categorical(key, factor_logits, axis=-1)
            factors = factors.at[..., i].set(factor)
            return (factors, mask), factor_logits

        factors = jnp.zeros((*embeddings.shape[:-2], self.factor_iterations), jnp.int32)
        mask = jnp.concatenate(
            [
                ems_mask,
                items_mask,
                jnp.zeros((*ems_mask.shape[:-1], self.factor_iterations), bool),
            ],
            axis=-1,
        )
        keys = jax.random.split(key, self.factor_iterations)
        (factors, mask), factors_logits = hk.scan(
            factor_scan_fn,
            (factors, mask),
            (jnp.arange(self.factor_iterations), keys),
        )

        # Last inference for the final action.
        factor_embeddings = self.embed_factors(factors)
        all_embeddings = jnp.concatenate([embeddings, factor_embeddings], axis=-2)

        self_attention_mask = self._make_self_attention_mask(mask)

        # Transformer encoder
        for transformer_block in transformer_blocks:
            all_embeddings = transformer_block(
                all_embeddings, all_embeddings, all_embeddings, self_attention_mask
            )

        embeddings, _ = jnp.split(
            all_embeddings,
            (embeddings.shape[-2],),
            axis=-2,
        )

        return embeddings, factors, factors_logits

    def embed_ems(self, ems: EMS) -> chex.Array:
        # Stack the 6 EMS attributes into a single vector [x1, x2, y1, y2, z1, z2].
        ems_leaves = jnp.stack(jax.tree_util.tree_leaves(ems), axis=-1)
        # Projection of the EMSs.
        embeddings = hk.Linear(self.model_size, name="ems_embedding")(ems_leaves)
        return embeddings

    def embed_items(self, items: Item) -> chex.Array:
        # Stack the 3 items attributes into a single vector [x_len, y_len, z_len].
        items_leaves = jnp.stack(jax.tree_util.tree_leaves(items), axis=-1)
        # Projection of the EMSs.
        embeddings = hk.Linear(self.model_size, name="item_embedding")(items_leaves)
        return embeddings

    def embed_factors(self, factors: chex.Array) -> chex.Array:
        # Projection of the factors.
        embeddings = self.factor_embedding_layer(factors)
        # TODO: add positional encoding.
        return embeddings

    def _make_self_attention_mask(self, mask: chex.Array) -> chex.Array:
        # Use the same mask for the query and the key.
        mask = jnp.einsum("...i,...j->...ij", mask, mask)
        # Expand on the head dimension.
        mask = jnp.expand_dims(mask, axis=-3)
        return mask


def make_actor_factor_exe_network_bin_pack(
    num_transformer_layers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
    factor_iterations: int,
    factor_vocab_size: int,
) -> FeedForwardNetwork:
    def network_fn(
        observation: Observation, key: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        torso = BinPackFactorTorso(
            num_transformer_layers=num_transformer_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            factor_iterations=factor_iterations,
            factor_vocab_size=factor_vocab_size,
            name="policy_torso",
        )
        embeddings, factors, factors_logits = hk.vmap(torso, split_rng=False)(observation, key)
        ems_embeddings, items_embeddings = jnp.split(
            embeddings, (observation.ems_mask.shape[-1],), axis=-2
        )

        # EMS projection.
        ems_embeddings = hk.Linear(torso.model_size, name="ems_projection")(ems_embeddings)
        # Items projection.
        items_embeddings = hk.Linear(torso.model_size, name="items_projection")(items_embeddings)

        # Outer-product between the embeddings to obtain logits.
        logits = jnp.einsum("...ek,...ik->...ei", ems_embeddings, items_embeddings)
        logits = jnp.where(observation.action_mask, logits, jnp.finfo(jnp.float32).min)
        return logits.reshape(*logits.shape[:-2], -1), factors, factors_logits

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
