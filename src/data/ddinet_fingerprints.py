"""Deterministic Morgan-fingerprint preprocessing for the DDINet adapter.

This module is intentionally independent from the numerical QSAR scaler used
by T-DDI.  Fingerprints are derived only from molecular structure, contain no
labels, and may therefore be shared by continual-learning methods without
fitting on future task targets.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
except ImportError:  # pragma: no cover - exercised by the runtime guard
    Chem = None
    DataStructs = None
    rdFingerprintGenerator = None


DDINET_FINGERPRINT_KIND = "ddi_cil_morgan_fingerprint_cache"
DDINET_FINGERPRINT_SCHEMA_VERSION = 1
DDINET_FINGERPRINT_MANIFEST = "manifest.json"
DDINET_FINGERPRINT_MATRIX = "fingerprints.npy"


@dataclass(frozen=True)
class DDINetFingerprintConfig:
    """Explicit provisional fingerprint contract for the DDINet adaptation.

    The accessible paper text fixes ``n_bits=1024`` but does not expose all
    RDKit arguments.  Radius two (ECFP4), non-chiral Morgan bits and ordered
    concatenation are therefore recorded as deliberate, replaceable choices.
    """

    radius: int = 2
    n_bits: int = 1024
    include_chirality: bool = False
    pair_composition: str = "ordered_concatenation"
    consumer: str = "ddinet"

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("Morgan radius must be positive.")
        if self.n_bits <= 0:
            raise ValueError("Morgan fingerprint size must be positive.")
        if self.pair_composition != "ordered_concatenation":
            raise ValueError("Morgan cache currently supports ordered concatenation only.")
        if not isinstance(self.consumer, str) or not self.consumer.strip():
            raise ValueError("Morgan fingerprint consumer must be a non-empty string.")

    @property
    def pair_input_dim(self) -> int:
        return 2 * self.n_bits

    def manifest(self) -> dict[str, object]:
        return {
            **asdict(self),
            "fingerprint_family": "morgan_ecfp",
            "paper_status": {
                "n_bits": "paper_confirmed",
                "radius": "provisional_adaptation",
                "include_chirality": "provisional_adaptation",
                "pair_composition": "provisional_adaptation",
            },
        }


def _require_rdkit() -> None:
    if Chem is None or DataStructs is None or rdFingerprintGenerator is None:
        raise ImportError(
            "DDINet fingerprint preprocessing requires RDKit. Install the "
            "project requirements before building the fingerprint cache."
        )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _exclusive_lock(path: Path, timeout: float = 600.0):
    started = time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(descriptor)
            break
        except FileExistsError:
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"Timed out waiting for DDINet cache lock: {path}")
            time.sleep(0.2)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def canonicalize_smiles(smiles: str) -> str:
    """Return a canonical isomeric SMILES or fail without silent imputation."""

    _require_rdkit()
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string.")
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return str(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))


def morgan_fingerprint(
    smiles: str,
    config: DDINetFingerprintConfig = DDINetFingerprintConfig(),
) -> tuple[str, np.ndarray]:
    """Canonicalise one molecule and return its immutable uint8 bit vector."""

    _require_rdkit()
    canonical = canonicalize_smiles(smiles)
    molecule = Chem.MolFromSmiles(canonical)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=config.radius,
        fpSize=config.n_bits,
        includeChirality=config.include_chirality,
    )
    bit_vector = generator.GetFingerprint(molecule)
    array = np.zeros((config.n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bit_vector, array)
    if array.shape != (config.n_bits,) or not np.isin(array, (0, 1)).all():
        raise ValueError("RDKit returned an invalid Morgan bit vector.")
    array.flags.writeable = False
    return canonical, array


def ordered_pair_fingerprint(
    smiles_a: str,
    smiles_b: str,
    config: DDINetFingerprintConfig = DDINetFingerprintConfig(),
) -> np.ndarray:
    """Build the provisional paper-adapted [fingerprint_A, fingerprint_B]."""

    _, fingerprint_a = morgan_fingerprint(smiles_a, config)
    _, fingerprint_b = morgan_fingerprint(smiles_b, config)
    return np.concatenate((fingerprint_a, fingerprint_b)).astype(np.float32, copy=False)


def _normalise_records(
    records: Mapping[str, str] | Iterable[tuple[str, str]],
    config: DDINetFingerprintConfig,
) -> list[dict[str, object]]:
    items = records.items() if isinstance(records, Mapping) else records
    by_id: dict[str, tuple[str, np.ndarray]] = {}
    for raw_drug_id, raw_smiles in items:
        drug_id = str(raw_drug_id).strip()
        if not drug_id:
            raise ValueError("Drug IDs must be non-empty strings.")
        canonical, fingerprint = morgan_fingerprint(str(raw_smiles), config)
        previous = by_id.get(drug_id)
        if previous is not None and previous[0] != canonical:
            raise ValueError(f"Drug ID {drug_id!r} maps to conflicting canonical SMILES.")
        by_id[drug_id] = (canonical, fingerprint)
    if not by_id:
        raise ValueError("At least one drug/SMILES record is required.")
    return [
        {
            "drug_id": drug_id,
            "canonical_smiles": canonical,
            "fingerprint": fingerprint,
        }
        for drug_id, (canonical, fingerprint) in sorted(by_id.items())
    ]


def build_ddinet_fingerprint_cache(
    root: str | Path,
    records: Mapping[str, str] | Iterable[tuple[str, str]],
    *,
    config: DDINetFingerprintConfig = DDINetFingerprintConfig(),
    lock_timeout: float = 600.0,
) -> "DDINetFingerprintCache":
    """Create or validate one immutable, method-neutral fingerprint cache."""

    root = Path(root).resolve()
    manifest_path = root / DDINET_FINGERPRINT_MANIFEST
    if manifest_path.is_file():
        return load_ddinet_fingerprint_cache(root, expected_config=config)

    with _exclusive_lock(root.parent / f".{root.name}.lock", timeout=lock_timeout):
        if manifest_path.is_file():
            return load_ddinet_fingerprint_cache(root, expected_config=config)
        root.mkdir(parents=True, exist_ok=True)
        unexpected = [path.name for path in root.iterdir()]
        if unexpected:
            raise FileExistsError(
                f"DDINet cache directory is non-empty without a manifest: {unexpected}"
            )

        normalised = _normalise_records(records, config)
        matrix = np.stack([record.pop("fingerprint") for record in normalised]).astype(
            np.uint8,
            copy=False,
        )
        for row_index, record in enumerate(normalised):
            record["row_index"] = row_index

        matrix_path = root / DDINET_FINGERPRINT_MATRIX
        temporary_matrix = root / f".{DDINET_FINGERPRINT_MATRIX}.{uuid.uuid4().hex}.tmp"
        with temporary_matrix.open("wb") as handle:
            np.save(handle, matrix, allow_pickle=False)
        os.replace(temporary_matrix, matrix_path)

        payload: dict[str, object] = {
            "artifact_kind": DDINET_FINGERPRINT_KIND,
            "schema_version": DDINET_FINGERPRINT_SCHEMA_VERSION,
            "config": config.manifest(),
            "matrix": {
                "filename": DDINET_FINGERPRINT_MATRIX,
                "dtype": "uint8",
                "shape": list(matrix.shape),
                "sha256": _sha256_file(matrix_path),
            },
            "records": normalised,
            "label_free": True,
        }
        payload["payload_sha256"] = _digest(payload)
        temporary_manifest = root / f".{DDINET_FINGERPRINT_MANIFEST}.{uuid.uuid4().hex}.tmp"
        temporary_manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
    return load_ddinet_fingerprint_cache(root, expected_config=config)


class DDINetFingerprintCache:
    """Verified read-only lookup over one fingerprint cache artifact."""

    def __init__(self, root: Path, manifest: dict[str, object], matrix: np.ndarray) -> None:
        self.root = root
        self.manifest = manifest
        self.matrix = matrix
        records = manifest["records"]
        self._row_by_drug_id = {
            str(record["drug_id"]): int(record["row_index"])
            for record in records
        }

    @property
    def input_dim(self) -> int:
        return 2 * int(self.matrix.shape[1])

    def fingerprint(self, drug_id: str) -> np.ndarray:
        try:
            row = self._row_by_drug_id[str(drug_id)]
        except KeyError as error:
            raise KeyError(f"Drug ID is absent from the DDINet fingerprint cache: {drug_id}") from error
        return np.asarray(self.matrix[row], dtype=np.uint8)

    def ordered_pair(self, drug_id_a: str, drug_id_b: str) -> np.ndarray:
        return np.concatenate(
            (self.fingerprint(drug_id_a), self.fingerprint(drug_id_b))
        ).astype(np.float32, copy=False)

    def ordered_pairs(
        self,
        drug_ids_a: Sequence[str],
        drug_ids_b: Sequence[str],
    ) -> np.ndarray:
        if len(drug_ids_a) != len(drug_ids_b):
            raise ValueError("Drug-pair ID sequences must have equal length.")
        if not drug_ids_a:
            return np.empty((0, self.input_dim), dtype=np.float32)
        return np.stack(
            [self.ordered_pair(a, b) for a, b in zip(drug_ids_a, drug_ids_b, strict=True)]
        )


def load_ddinet_fingerprint_cache(
    root: str | Path,
    *,
    expected_config: DDINetFingerprintConfig | None = None,
) -> DDINetFingerprintCache:
    root = Path(root).resolve()
    manifest_path = root / DDINET_FINGERPRINT_MANIFEST
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksum = payload.pop("payload_sha256", None)
    if checksum != _digest(payload):
        raise ValueError("DDINet fingerprint manifest checksum mismatch.")
    payload["payload_sha256"] = checksum
    if payload.get("artifact_kind") != DDINET_FINGERPRINT_KIND:
        raise ValueError("Unexpected DDINet fingerprint artifact kind.")
    if payload.get("schema_version") != DDINET_FINGERPRINT_SCHEMA_VERSION:
        raise ValueError("Unsupported DDINet fingerprint cache schema.")
    if payload.get("label_free") is not True:
        raise ValueError("DDINet fingerprint cache must be explicitly label-free.")
    if expected_config is not None and payload.get("config") != expected_config.manifest():
        raise ValueError("DDINet fingerprint configuration mismatch.")

    matrix_info = payload["matrix"]
    matrix_path = root / str(matrix_info["filename"])
    if _sha256_file(matrix_path) != matrix_info["sha256"]:
        raise ValueError("DDINet fingerprint matrix checksum mismatch.")
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    if (
        matrix.dtype != np.uint8
        or list(matrix.shape) != matrix_info["shape"]
        or matrix.ndim != 2
        or not np.isin(matrix, (0, 1)).all()
    ):
        raise ValueError("DDINet fingerprint matrix contract mismatch.")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != matrix.shape[0]:
        raise ValueError("DDINet fingerprint record coverage mismatch.")
    if [int(record["row_index"]) for record in records] != list(range(len(records))):
        raise ValueError("DDINet fingerprint row indices are not contiguous.")
    drug_ids = [str(record["drug_id"]) for record in records]
    if len(drug_ids) != len(set(drug_ids)):
        raise ValueError("DDINet fingerprint cache contains duplicate drug IDs.")
    return DDINetFingerprintCache(root, payload, matrix)
