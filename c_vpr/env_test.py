from pytest import fixture

import jax
import jax.numpy as jnp

from c_vpr.env import C_VPR


@fixture
def c_vpr():
    return C_VPR(K=10, L=10)


def test__c_vpr_sample(c_vpr: C_VPR):
    key = jax.random.PRNGKey(0)
    example = c_vpr.sample(key)


def test__c_vpr_get_num_hops(c_vpr: C_VPR):
    example = jnp.array([1, 3, 5, 4, 1], int)
    num_hops = c_vpr.get_num_hops(example)
    assert num_hops == 3


if __name__ == "__main__":
    test__c_vpr_get_num_hops(C_VPR(K=10, L=10))
