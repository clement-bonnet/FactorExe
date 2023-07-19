from typing import NamedTuple, Tuple

import chex
import jax.numpy as jnp
from jumanji.environments.packing.bin_pack import BinPack, Observation, State
from jumanji.types import TimeStep, restart


class AugmentedState(NamedTuple):
    original_state: State
    bin_pack_solution: State
    key: chex.PRNGKey


class BinPackSolutionWrapper(BinPack):
    """Add solution to timestep.extras."""

    def reset(self, key: chex.PRNGKey) -> Tuple[AugmentedState, TimeStep[Observation]]:
        bin_pack_solution = self.generator.generate_solution(key)
        state = self.generator._unpack_items(bin_pack_solution)
        # Make the observation.
        state, observation, extras = self._make_observation_and_extras(state)

        ### Newly added
        augmented_state = AugmentedState(
            original_state=state, bin_pack_solution=bin_pack_solution, key=state.key
        )
        ###

        extras.update(invalid_action=jnp.array(False))
        if self.debug:
            extras.update(invalid_ems_from_env=jnp.array(False))

        ### Newly added
        extras.update(bin_pack_solution=bin_pack_solution)
        ###

        timestep = restart(observation, extras)

        return augmented_state, timestep

    def step(
        self, augmented_state: AugmentedState, action: chex.Array
    ) -> Tuple[AugmentedState, TimeStep[Observation]]:
        bin_pack_solution = augmented_state.bin_pack_solution
        state, timestep = super().step(augmented_state.original_state, action)
        augmented_state = AugmentedState(
            original_state=state, bin_pack_solution=bin_pack_solution, key=state.key
        )
        timestep.extras.update(bin_pack_solution=bin_pack_solution)
        return augmented_state, timestep
