import chex
import jax
import jax.numpy as jnp
import pytest
from pytest import fixture

from c_vpr.env import C_VPR


@fixture
def c_vpr() -> C_VPR:
    return C_VPR(input_length=10)


def test__c_vpr_sample(c_vpr: C_VPR) -> None:
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    example1 = jax.jit(c_vpr.sample)(key1)
    assert isinstance(example1, chex.Array)
    example2 = c_vpr.sample(key2)
    assert not jnp.array_equal(example1, example2)


def test__c_vpr_sample_n_hops(c_vpr: C_VPR) -> None:
    key1, key2, key = jax.random.split(jax.random.PRNGKey(0), 3)
    num_hops = 2
    example1 = jax.jit(c_vpr.sample_n_hops, static_argnums=1)(key1, num_hops)
    assert isinstance(example1, chex.Array)
    example2 = c_vpr.sample_n_hops(key2, num_hops)
    assert not jnp.array_equal(example1, example2)
    example, target = jax.jit(
        c_vpr.sample_n_hops, static_argnames=("num_hops", "return_target")
    )(key, num_hops, return_target=True)
    assert target.shape == ()
    assert jnp.isin(target, example)


def test__c_vpr_get_num_hops(c_vpr: C_VPR) -> None:
    example = jnp.array([1, 3, 5, 4, 1], int)
    num_hops = jax.jit(c_vpr.get_num_hops)(example)
    assert num_hops == 3


@pytest.mark.parametrize("num_hops", [1, 2, 3, 5, 8, 9])
def test_functional__c_vpr_sample_n_hops(c_vpr: C_VPR, num_hops: int) -> None:
    key = jax.random.PRNGKey(0)
    example = c_vpr.sample_n_hops(key, num_hops)
    assert c_vpr.get_num_hops(example) == num_hops


if __name__ == "__main__":
    test__c_vpr_sample_n_hops(C_VPR(input_length=10))
