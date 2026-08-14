"""Generic state-key collision certificates for ARC graph exploration.

The graph explorer normally keys a node by the current visible frame. Some live
games have hidden progress where two distinct visible histories reach the same
frame hash. This helper extends a key only after it observes that alias inside
one run. It uses visible history tokens and the agent's own actions only.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping, Sequence


class HashSubstitutionError(RuntimeError):
    """Raised when a supplied history digest contradicts the raw history tokens."""


@dataclass
class _HistoryRecord:
    digest: str
    raw_history: str
    action_history: tuple[str, ...]
    assigned_key: str


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _action_label(step: Mapping[str, Any]) -> str:
    action = int(step.get("action", 0))
    data = step.get("data")
    if data is None:
        return str(action)
    return f"{action}@{_stable_json(data)}"


def _history_digest(history: Sequence[Any]) -> str:
    return hashlib.sha256(_stable_json(list(history)).encode()).hexdigest()


class StateKeyCollisionCertifier:
    """Certify and apply a minimal action suffix only after a live key alias.

    The class keeps raw canonical histories as well as hashes. That extra copy is
    deliberate: if a digest is substituted or unstable, trusting the digest alone
    would let a false certificate pass or hide a true collision.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        max_suffix_k: int = 4,
        history_digest_func: Callable[[Sequence[Any]], str] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_suffix_k = max(1, int(max_suffix_k))
        self._history_digest_func = history_digest_func or _history_digest
        self._records_by_base: dict[str, list[_HistoryRecord]] = {}
        self._digest_to_raw: dict[str, str] = {}
        self._raw_to_digest: dict[str, str] = {}
        self._certificate_rows: list[dict[str, Any]] = []
        self._hash_substitution_detected = False
        self._hash_instability_detected = False
        self._refused_certificate_count = 0

    def reset(self) -> None:
        self._records_by_base.clear()
        self._digest_to_raw.clear()
        self._raw_to_digest.clear()
        self._certificate_rows.clear()
        self._hash_substitution_detected = False
        self._hash_instability_detected = False
        self._refused_certificate_count = 0

    def state_key(
        self,
        base_key: str,
        observation_history: Sequence[Any],
        action_history: Sequence[Mapping[str, Any]],
    ) -> str:
        if not self.enabled:
            return str(base_key)

        base = str(base_key)
        raw_history = _stable_json(list(observation_history))
        digest = str(self._history_digest_func(observation_history))
        self._check_digest(raw_history, digest)

        actions = tuple(_action_label(step) for step in action_history)
        records = self._records_by_base.setdefault(base, [])
        for record in records:
            if record.raw_history == raw_history:
                return record.assigned_key

        if not records:
            records.append(_HistoryRecord(digest, raw_history, actions, base))
            return base

        trial_records = records + [_HistoryRecord(digest, raw_history, actions, base)]
        suffix_k = self._minimal_suffix_k(trial_records)
        if suffix_k is None:
            self._refused_certificate_count += 1
            assigned = base
        else:
            assigned = self._extended_key(base, actions, suffix_k)
            self._certificate_rows.append(
                {
                    "base_key": base,
                    "observation_history_hashes": [r.digest for r in trial_records],
                    "alias_evidence": {
                        "known_history_count": len(trial_records),
                        "new_history_hash": digest,
                        "prior_history_hashes": [r.digest for r in records],
                        "base_key_reused": True,
                    },
                    "minimal_suffix_k": int(suffix_k),
                    "minimal_suffix_selected": list(actions[-suffix_k:]) if suffix_k else [],
                    "forbidden_inputs": [],
                }
            )
        records.append(_HistoryRecord(digest, raw_history, actions, assigned))
        return assigned

    def certificate_rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._certificate_rows]

    def diagnostics(self) -> dict[str, Any]:
        max_k = 0
        for row in self._certificate_rows:
            max_k = max(max_k, int(row.get("minimal_suffix_k") or 0))
        return {
            "enabled": self.enabled,
            "accepted_certificate_count": len(self._certificate_rows),
            "refused_certificate_count": self._refused_certificate_count,
            "hash_substitution_detected": self._hash_substitution_detected,
            "hash_instability_detected": self._hash_instability_detected,
            "max_suffix_k_used": max_k,
            "base_key_count": len(self._records_by_base),
        }

    def _check_digest(self, raw_history: str, digest: str) -> None:
        prior_raw = self._digest_to_raw.get(digest)
        if prior_raw is not None and prior_raw != raw_history:
            self._hash_substitution_detected = True
            raise HashSubstitutionError("distinct histories produced the same observation digest")
        prior_digest = self._raw_to_digest.get(raw_history)
        if prior_digest is not None and prior_digest != digest:
            self._hash_instability_detected = True
            raise HashSubstitutionError("one history produced more than one observation digest")
        self._digest_to_raw[digest] = raw_history
        self._raw_to_digest[raw_history] = digest

    def _minimal_suffix_k(self, records: Iterable[_HistoryRecord]) -> int | None:
        rows = list(records)
        for k in range(1, self.max_suffix_k + 1):
            suffixes = [self._suffix_signature(row.action_history, k) for row in rows]
            if len(set(suffixes)) == len(suffixes):
                return k
        return None

    @staticmethod
    def _suffix_signature(actions: Sequence[str], k: int) -> str:
        if not actions:
            return "<root>"
        return "|".join(actions[-k:])

    def _extended_key(self, base_key: str, actions: Sequence[str], k: int) -> str:
        suffix = self._suffix_signature(actions, k)
        suffix_hash = hashlib.sha256(suffix.encode()).hexdigest()[:16]
        return f"{base_key}|certk:{int(k)}:{suffix_hash}"
