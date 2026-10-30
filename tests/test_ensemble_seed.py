import random
import numpy as np
import torch
from src.utils.seed import derive_member_seed, resolve_seed_configuration, set_configured_seeds


def test_member_streams_are_distinct_and_repeatable():
    seeds = [resolve_seed_configuration(0, i) for i in range(3)]
    assert len({s.member_seed for s in seeds}) == 3
    for s in seeds:
        set_configured_seeds(s)
        first = random.random(), np.random.rand(), torch.rand(2)
        set_configured_seeds(s)
        second = random.random(), np.random.rand(), torch.rand(2)
        assert first[:2] == second[:2]
        torch.testing.assert_close(first[2], second[2])
        assert derive_member_seed(0, s.member_id) == s.member_seed
