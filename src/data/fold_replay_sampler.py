"""Opt-in replay-fraction/cap sampler; no changes to legacy FixedReplaySampler.

One instance per task, with immutable current/memory IDs. Epoch RNG streams do
not consume model/global RNG. No feature matrices or trainer dependencies.
"""

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
from typing import Iterator, Sequence

import numpy as np

from src.utils.seed import resolve_seed_configuration

try:
    from torch.utils.data import Sampler
except ImportError:  # pragma: no cover
    Sampler = object


SAMPLER_POLICY = "fold_fraction_capped_rotating_v1"
ROTATING_CURRENT_SAMPLER_POLICY = "fold_fraction_capped_rotating_current_v2"
RNG_DERIVATION = "seedsequence_pcg64_member_task_epoch_stream_v1"


def _integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= int(value) < 2**32:
        raise ValueError(f"{name} must be a uint32 integer.")
    return int(value)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    array = np.asarray(values)
    if array.ndim != 1 or not all(isinstance(s, (str, np.str_)) and s for s in array):
        raise ValueError(f"{name} must be a one-dimensional sequence of nonempty string IDs.")
    result = tuple(str(s) for s in array)
    if len(set(result)) != len(result):
        raise ValueError(f"Duplicate {name}.")
    return result


@dataclass(frozen=True)
class ReplayEpochPlan:
    """Indices address concat(current_rows, replay_rows); no rows are dropped."""

    indices: tuple[int, ...]
    audit: dict


class FoldReplayFractionSampler(Sampler):
    """Capped class-round-robin replay with full or rotating current rows.

    Default target R=floor(N/7), actual=min(R,cap*memory_size), task0 R=0.
    Task0 rejects a nonempty memory to catch stale/wrong task state. A zero-sized
    current set is allowed by this primitive (zero draws); trainer handles whether
    empty training tasks are legal. Memory is fixed within the task.

    Resume is at EPOCH boundaries, not within a DataLoader-prefetched epoch. An
    iterator must be exhausted (or explicitly closed to abandon/retry that epoch)
    before state_dict/set_epoch. A new task requires a NEW sampler, starting epoch0.
    """

    def __init__(self, *, current_sample_ids: Sequence[str], replay_sample_ids: Sequence[str],
                 replay_raw_labels: Sequence[int], experiment_seed: int, member_id: int,
                 task_id: int, replay_fraction: float = 0.125, repeat_cap: int = 3,
                 rotate_current_when_capacity_limited: bool = False) -> None:
        self._current = _ids(current_sample_ids, "current sample IDs")
        self._replay = _ids(replay_sample_ids, "replay sample IDs")
        if set(self._current) & set(self._replay):
            raise ValueError("Current and replay sample IDs overlap.")
        labels = np.asarray(replay_raw_labels)
        if (labels.shape != (len(self._replay),) or (labels.size and labels.dtype.kind not in "iu")
                or any(not -(2**63) <= int(c) < 2**63 for c in labels)):
            raise ValueError("replay_raw_labels must be aligned raw int64 class IDs.")
        self._labels = tuple(int(c) for c in labels)
        task_id, member_id = _integer(task_id, "task_id"), _integer(member_id, "member_id")
        if member_id not in (0, 1, 2):
            raise ValueError("member_id must be 0, 1 or 2.")
        if task_id == 0 and self._replay:
            raise ValueError("Task0 must have empty replay memory.")
        repeat_cap = _integer(repeat_cap, "repeat_cap")
        if repeat_cap == 0:
            raise ValueError("repeat_cap must be positive.")
        try:
            fraction = Fraction(str(replay_fraction))
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError("replay_fraction must be finite and in [0,1).") from error
        if not 0 <= fraction < 1:
            raise ValueError("replay_fraction must be in [0,1).")
        if not isinstance(rotate_current_when_capacity_limited, (bool, np.bool_)):
            raise ValueError("rotate_current_when_capacity_limited must be boolean.")
        rotate_current_when_capacity_limited = bool(rotate_current_when_capacity_limited)
        seeds = resolve_seed_configuration(_integer(experiment_seed, "experiment_seed"), member_id)
        self._seed, self._task_id, self._cap = seeds.member_seed, task_id, repeat_cap
        self._target = 0 if task_id == 0 else len(self._current) * fraction.numerator // (fraction.denominator - fraction.numerator)
        self._actual = min(self._target, repeat_cap * len(self._replay))
        self._rotate_current = rotate_current_when_capacity_limited
        if task_id == 0 or not rotate_current_when_capacity_limited or self._actual >= self._target:
            self._current_draws = len(self._current)
        elif self._actual == 0 or fraction.numerator == 0:
            self._current_draws = len(self._current)
        else:
            # If replay capacity cannot support f of a full-current epoch, reduce
            # current draws to R*(1-f)/f.  At f=1/8 this is seven current draws
            # per replay draw.  Epoch windows rotate, so rows are deferred rather
            # than discarded from the task trajectory.
            matched = self._actual * (fraction.denominator - fraction.numerator) // fraction.numerator
            self._current_draws = min(len(self._current), matched)
        classes = sorted(set(self._labels))
        self._class_order = tuple(classes[i] for i in self._rng(0, 0).permutation(len(classes)))
        self._queues = {}
        for c in self._class_order:
            indices = sorted((i for i, label in enumerate(self._labels) if label == c), key=lambda i: self._replay[i])
            unsigned = c % 2**64
            order = self._rng(1, 0, unsigned & 0xFFFFFFFF, unsigned >> 32).permutation(len(indices))
            self._queues[c] = tuple(indices[i] for i in order)
        self._canonical_current = np.asarray(sorted(range(len(self._current)), key=lambda i: self._current[i]), dtype=np.int64)
        policy = ROTATING_CURRENT_SAMPLER_POLICY if rotate_current_when_capacity_limited else SAMPLER_POLICY
        self._metadata = {
            "policy": policy, "schema_version": 2 if rotate_current_when_capacity_limited else 1,
            "seeds": asdict(seeds), "task_id": task_id,
            "fraction_numerator": fraction.numerator, "fraction_denominator": fraction.denominator,
            "target_fraction": float(fraction), "rounding": "floor_N_times_f_over_1_minus_f",
            "repeat_cap": repeat_cap, "current_count": len(self._current), "memory_count": len(self._replay),
            "current_order_sha256": _digest(self._current),
            "replay_order_label_sha256": _digest(list(zip(self._replay, self._labels))),
            "rng_derivation": RNG_DERIVATION, "rng_state_policy": "stateless_keyed_streams_no_global_rng",
            "queue_policy": "task_fixed_seeded_ID_sorted_cyclic_queues",
            "class_policy": "rotating_round_robin_skip_epoch_saturated_classes",
            "mix_policy": "random_positions_preserve_replay_relative_order", "drop_last": False,
            "state_boundary": "completed_epoch_only", "reset_policy": "new_instance_at_each_task_epoch0",
        }
        if rotate_current_when_capacity_limited:
            self._metadata.update({
                "current_draws_per_epoch": self._current_draws,
                "rotate_current_when_capacity_limited": True,
                "current_policy": "cyclic_ID_sorted_windows_seeded_order_capacity_limited",
            })
        self._next_epoch = 0
        self._class_cursor = 0
        self._cursors = {c: 0 for c in self._class_order}
        self._active = False
        self._last_audit = None

    def _rng(self, stream: int, epoch: int, *extra: int) -> np.random.Generator:
        sequence = np.random.SeedSequence([self._seed, self._task_id, 0x5245504C, stream, epoch, *extra])
        return np.random.Generator(np.random.PCG64(sequence))

    @property
    def metadata(self) -> dict:
        return deepcopy(self._metadata)

    @property
    def next_epoch(self) -> int:
        return self._next_epoch

    @property
    def last_audit(self) -> dict | None:
        """Only the last FULLY consumed epoch, not a prefetched/abandoned plan."""
        return deepcopy(self._last_audit)

    def __len__(self) -> int:
        return self._current_draws + self._actual

    def _advance(self, class_cursor: int, cursors: dict[int, int], *, collect: bool):
        """Round-robin one replay slot at a time; remove capped classes this epoch."""
        next_cursors = dict(cursors)
        draws = {c: 0 for c in self._class_order}
        indices = []
        if self._actual == 0:
            return indices, class_cursor, next_cursors
        count = len(self._class_order)
        active = deque((class_cursor + i) % count for i in range(count))
        for _ in range(self._actual):
            position = active.popleft()
            c = self._class_order[position]
            queue = self._queues[c]
            if collect:
                indices.append(queue[next_cursors[c]])
            next_cursors[c] = (next_cursors[c] + 1) % len(queue)
            draws[c] += 1
            class_cursor = (position + 1) % count
            if draws[c] < self._cap * len(queue):
                active.append(position)
        return indices, class_cursor, next_cursors

    def _schedule_before(self, epoch: int):
        class_cursor, cursors = 0, {c: 0 for c in self._class_order}
        for _ in range(epoch):
            _, class_cursor, cursors = self._advance(class_cursor, cursors, collect=False)
        return class_cursor, cursors

    def _plan(self, epoch: int, class_cursor: int, cursors: dict[int, int]):
        replay_indices, next_class, next_cursors = self._advance(class_cursor, cursors, collect=True)
        if self._current_draws == len(self._current):
            current = self._canonical_current[self._rng(2, epoch).permutation(len(self._current))]
        else:
            start = (epoch * self._current_draws) % len(self._current)
            positions = (start + np.arange(self._current_draws, dtype=np.int64)) % len(self._current)
            current = self._canonical_current[positions]
            current = current[self._rng(2, epoch).permutation(self._current_draws)]
        # Preserve cyclic replay ordering: shuffling all indices would allow an
        # exemplar repeat to appear before other exemplars in its current cycle.
        positions = np.zeros(len(self), dtype=bool)
        positions[self._rng(3, epoch).permutation(len(self))[:self._actual]] = True
        indices = np.empty(len(self), dtype=np.int64)
        indices[~positions] = current
        indices[positions] = np.asarray(replay_indices, dtype=np.int64) + len(self._current)
        repetitions = np.bincount(replay_indices, minlength=len(self._replay)) if replay_indices else np.zeros(len(self._replay), dtype=np.int64)
        per_class = {}
        for c in sorted(self._queues):
            counts = repetitions[list(self._queues[c])]
            unique = int(np.count_nonzero(counts))
            per_class[str(c)] = {"draws": int(counts.sum()), "unique_exemplars": unique,
                                 "memory_count": len(counts), "exemplar_coverage": unique / len(counts),
                                 "max_repeat": int(counts.max())}
        used_classes = sum(v["draws"] > 0 for v in per_class.values())
        unique = int(np.count_nonzero(repetitions))
        all_ids = self._current + self._replay
        audit = {
            "policy": self._metadata["policy"], "task_id": self._task_id, "epoch": epoch,
            "current_draws": self._current_draws, "target_replay_draws": self._target,
            "actual_replay_draws": self._actual, "total_draws": len(self),
            "target_fraction": self._metadata["target_fraction"],
            "actual_fraction": self._actual / len(self) if len(self) else 0.0,
            "capacity_limited": self._actual < self._target, "repeat_cap": self._cap,
            "memory_size": len(self._replay), "unique_replay_exemplars": unique,
            "exemplar_coverage": unique / len(self._replay) if self._replay else 0.0,
            "eligible_class_count": len(per_class), "replayed_class_count": used_classes,
            "class_coverage": used_classes / len(per_class) if per_class else 0.0,
            "coverage_denominator": "classes_and_exemplars_present_in_buffer",
            "per_class": per_class,
            "repeat_histogram": {str(n): int(np.sum(repetitions == n)) for n in range(self._cap + 1)},
            "max_repeat": int(repetitions.max()) if len(repetitions) else 0,
            "replay_ids_sha256": _digest([self._replay[i] for i in replay_indices]),
            "draw_order_ids_sha256": _digest([all_ids[i] for i in indices]),
        }
        if self._rotate_current:
            audit.update({
                "current_count": len(self._current),
                "current_coverage": self._current_draws / len(self._current) if self._current else 0.0,
                "current_rotation_start": (
                    (epoch * self._current_draws) % len(self._current) if self._current else 0
                ),
            })
        if audit["max_repeat"] > self._cap or len(indices[~positions]) != self._current_draws:
            raise RuntimeError("Sampler cap/current coverage invariant failed.")
        return ReplayEpochPlan(tuple(int(i) for i in indices), audit), next_class, next_cursors

    def plan_epoch(self, epoch: int) -> ReplayEpochPlan:
        """Pure preview: deterministic local epoch, no cursor/RNG/state mutation."""
        epoch = _integer(epoch, "epoch")
        class_cursor, cursors = self._schedule_before(epoch)
        return self._plan(epoch, class_cursor, cursors)[0]

    def set_epoch(self, epoch: int) -> None:
        """Seek/reconstruct local epoch state, independent of earlier task lengths."""
        if self._active:
            raise RuntimeError("Cannot set_epoch during an active iterator.")
        epoch = _integer(epoch, "epoch")
        class_cursor, cursors = self._schedule_before(epoch)
        self._next_epoch, self._class_cursor, self._cursors = epoch, class_cursor, cursors
        self._last_audit = None

    def __iter__(self) -> Iterator[int]:
        if self._active:
            raise RuntimeError("An epoch iterator is already active.")
        plan, next_class, next_cursors = self._plan(self._next_epoch, self._class_cursor, self._cursors)
        self._active = True
        return _EpochIterator(self, plan, next_class, next_cursors)

    def state_dict(self) -> dict:
        if self._active:
            raise RuntimeError("Sampler checkpoint requires a completed epoch; active/prefetched iterator exists.")
        state = {
            "metadata": self.metadata, "next_epoch": self._next_epoch,
            "class_order": list(self._class_order), "class_cursor": self._class_cursor,
            "exemplar_queues": {str(c): [self._replay[i] for i in queue] for c, queue in self._queues.items()},
            "exemplar_cursors": {str(c): position for c, position in self._cursors.items()},
            "rng_state": {"derivation": RNG_DERIVATION, "mutable_rng_state": None,
                          "next_epoch_key": self._next_epoch},
        }
        state["state_sha256"] = _digest(state)
        return state

    @classmethod
    def from_state_dict(cls, state: dict, **kwargs):
        """Validate identity/config/cursors/queues and reconstruct an epoch boundary."""
        result = cls(**kwargs)
        try:
            if state.get("state_sha256") != _digest({k: v for k, v in state.items() if k != "state_sha256"}):
                raise ValueError("Sampler state SHA256 mismatch.")
            if state["metadata"] != result.metadata:
                raise ValueError("Sampler metadata/IDs/labels/seed/task/config mismatch.")
            result.set_epoch(state["next_epoch"])
            if state != result.state_dict():
                raise ValueError("Sampler cursor/queue/RNG state mismatch.")
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("Malformed sampler state.") from error
        return result


class _EpochIterator:
    """Commit cursors only on StopIteration; close() abandons the planned epoch."""

    def __init__(self, owner, plan, next_class, next_cursors):
        self.owner, self.plan = owner, plan
        self.next_class, self.next_cursors = next_class, next_cursors
        self.position, self.closed = 0, False

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        if self.position == len(self.plan.indices):
            self.owner._class_cursor = self.next_class
            self.owner._cursors = self.next_cursors
            self.owner._next_epoch += 1
            self.owner._last_audit = deepcopy(self.plan.audit)
            self.close()
            raise StopIteration
        index = self.plan.indices[self.position]
        self.position += 1
        return index

    def close(self):
        if not self.closed:
            self.owner._active = False
            self.closed = True

    def __del__(self):  # best effort on abandonment; explicit close is portable
        self.close()
