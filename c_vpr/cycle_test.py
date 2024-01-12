import chex
import jax
import jax.numpy as jnp
import pytest
from pytest import fixture

from c_vpr.cycle import Cycle


@fixture
def cycle() -> Cycle:
    return Cycle(input_length=10)


def test__c_vpr_sample(cycle: Cycle) -> None:
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    seq1 = jax.jit(cycle.sample)(key1)
    assert isinstance(seq1, chex.Array)
    seq2 = cycle.sample(key2)
    assert not jnp.array_equal(seq1, seq2)


def test__c_vpr_sample_n_hops(cycle: Cycle) -> None:
    key1, key2, key = jax.random.split(jax.random.PRNGKey(0), 3)
    num_hops = 2
    seq1 = jax.jit(cycle.sample_n_hops, static_argnums=1)(key1, num_hops)
    assert isinstance(seq1, chex.Array)
    seq2 = cycle.sample_n_hops(key2, num_hops)
    assert not jnp.array_equal(seq1, seq2)
    sequence, target = jax.jit(
        cycle.sample_n_hops, static_argnames=("num_hops", "return_target")
    )(key, num_hops, return_target=True)
    assert target.shape == ()
    assert jnp.isin(target, sequence)


@pytest.mark.parametrize("num_hops", [1, 2, 3, 5, 8, 9])
def test_functional__c_vpr_sample_n_hops(cycle: Cycle, num_hops: int) -> None:
    key = jax.random.PRNGKey(0)
    sequence = cycle.sample_n_hops(key, num_hops)
    assert sequence.shape == (cycle.input_length,)
