from typing import Tuple

import chex
from jumanji.environments.packing.bin_pack import BinPack, Observation, State
from jumanji.types import TimeStep


class AugmentedState(State):
    bin_pack_solution: State


class BinPackSolutionWrapper(BinPack):
    """Add solution to timestep.extras."""

    def reset(self, key: chex.PRNGKey) -> Tuple[AugmentedState, TimeStep[Observation]]:
        state, timestep = super().reset(key)
        bin_pack_solution = self.generator.generate_solution(key)
        state = AugmentedState(bin_pack_solution=bin_pack_solution)
        timestep.extras["bin_pack_solution"] = bin_pack_solution
        return state, timestep

    def step(
        self, state: AugmentedState, action: chex.Array
    ) -> Tuple[AugmentedState, TimeStep[Observation]]:
        new_state, new_timestep = super().step(state, action)
        new_state = AugmentedState(bin_pack_solution=state.bin_pack_solution)
        new_timestep.extras["bin_pack_solution"] = new_state.bin_pack_solution
        return state, new_timestep
