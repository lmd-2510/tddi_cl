import json

import numpy as np
import pytest

from src.data.ddinet_fingerprints import (
    DDINetFingerprintConfig,
    build_ddinet_fingerprint_cache,
    canonicalize_smiles,
    load_ddinet_fingerprint_cache,
    morgan_fingerprint,
    ordered_pair_fingerprint,
)


def test_morgan_fingerprint_is_deterministic_and_fail_fast():
    config = DDINetFingerprintConfig()
    canonical_a, first = morgan_fingerprint("CCO", config)
    canonical_b, second = morgan_fingerprint("OCC", config)

    assert canonical_a == canonical_b == "CCO"
    assert first.dtype == np.uint8
    assert first.shape == (1024,)
    np.testing.assert_array_equal(first, second)
    assert set(np.unique(first)).issubset({0, 1})

    with pytest.raises(ValueError, match="could not parse"):
        canonicalize_smiles("not-a-smiles")


def test_ordered_pair_and_immutable_cache_contract(tmp_path):
    config = DDINetFingerprintConfig()
    direct_ab = ordered_pair_fingerprint("CCO", "CCN", config)
    direct_ba = ordered_pair_fingerprint("CCN", "CCO", config)
    assert direct_ab.shape == (2048,)
    assert direct_ab.dtype == np.float32
    assert not np.array_equal(direct_ab, direct_ba)

    root = tmp_path / "cache"
    cache = build_ddinet_fingerprint_cache(
        root,
        [("drug-b", "CCN"), ("drug-a", "CCO"), ("drug-a", "OCC")],
        config=config,
    )
    np.testing.assert_array_equal(cache.ordered_pair("drug-a", "drug-b"), direct_ab)
    np.testing.assert_array_equal(
        cache.ordered_pairs(["drug-a", "drug-b"], ["drug-b", "drug-a"]),
        np.stack((direct_ab, direct_ba)),
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["label_free"] is True
    assert manifest["config"]["paper_status"]["n_bits"] == "paper_confirmed"
    assert [record["drug_id"] for record in manifest["records"]] == ["drug-a", "drug-b"]
    load_ddinet_fingerprint_cache(root, expected_config=config)

    with pytest.raises(ValueError, match="configuration mismatch"):
        load_ddinet_fingerprint_cache(
            root,
            expected_config=DDINetFingerprintConfig(radius=3),
        )


def test_cache_rejects_conflicting_structure_for_one_drug(tmp_path):
    with pytest.raises(ValueError, match="conflicting canonical SMILES"):
        build_ddinet_fingerprint_cache(
            tmp_path / "cache",
            [("drug-a", "CCO"), ("drug-a", "CCN")],
        )
