"""Backbone-neutral public API for deterministic Morgan fingerprint caches.

The implementation originated with the DDINet adapter.  These aliases expose
the same verified artifact for other fingerprint backbones while the explicit
``consumer`` field keeps cache provenance and namespaces distinct.
"""

from src.data.ddinet_fingerprints import (
    DDINetFingerprintCache as MorganFingerprintCache,
    DDINetFingerprintConfig as MorganFingerprintConfig,
    build_ddinet_fingerprint_cache as build_morgan_fingerprint_cache,
    canonicalize_smiles,
    load_ddinet_fingerprint_cache as load_morgan_fingerprint_cache,
    morgan_fingerprint,
    ordered_pair_fingerprint,
)


__all__ = [
    "MorganFingerprintCache",
    "MorganFingerprintConfig",
    "build_morgan_fingerprint_cache",
    "canonicalize_smiles",
    "load_morgan_fingerprint_cache",
    "morgan_fingerprint",
    "ordered_pair_fingerprint",
]
