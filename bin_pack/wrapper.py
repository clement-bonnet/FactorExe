from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp
from jumanji.environments.packing.bin_pack import BinPack, Observation, State
from jumanji.types import TimeStep, restart
from jumanji.wrappers import VmapAutoResetWrapper


class AugmentedState(NamedTuple):
    original_state: State
    bin_pack_solution: State
    key: chex.PRNGKey


class BinPackSolutionWrapper(BinPack):
    """Add solution to timestep.extras."""

    def reset(self, key: chex.PRNGKey) -> Tuple[AugmentedState, TimeStep[Observation]]:
        bin_pack_solution = self.generator.generate_solution(key)
        # Avoid side effects on the bin_pack_solution.
        bin_pack_solution_copy = State(**vars(bin_pack_solution))
        state = self.generator._unpack_items(bin_pack_solution)
        # Make the observation.
        state, observation, extras = self._make_observation_and_extras(state)

        ### Newly added
        augmented_state = AugmentedState(
            original_state=state,
            bin_pack_solution=bin_pack_solution_copy,
            key=state.key,
        )
        ###

        extras.update(invalid_action=jnp.array(False))
        if self.debug:
            extras.update(invalid_ems_from_env=jnp.array(False))

        ### Newly added
        extras.update(bin_pack_solution=bin_pack_solution_copy)
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


class VmapAutoResetWrapperBinPackSolution(VmapAutoResetWrapper):
    def __init__(self, env: BinPackSolutionWrapper):
        super().__init__(env)

    def _auto_reset(
        self, state: AugmentedState, timestep: TimeStep
    ) -> Tuple[AugmentedState, TimeStep[Observation]]:
        """Reset the state and overwrite `timestep.observation` with the reset observation
        if the episode has terminated.
        """
        if not hasattr(state, "key"):
            raise AttributeError(
                "This wrapper assumes that the state has attribute key which is used"
                " as the source of randomness for automatic reset"
            )

        # Make sure that the random key in the environment changes at each call to reset.
        # State is a type variable hence it does not have key type hinted, so we type ignore.
        key, _ = jax.random.split(state.key)
        state, reset_timestep = self._env.reset(key)

        # Replace observation with reset observation.
        timestep.extras.update(
            bin_pack_solution=reset_timestep.extras["bin_pack_solution"]
        )
        timestep = timestep.replace(  # type: ignore
            observation=reset_timestep.observation,
        )

        return state, timestep
