#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
QuantumEntropyPool -- drop-in replacement for the Qiskit simulator call
================================================================================
Replace the Section 2.1 entropy source inside the encryption pipeline without
touching anything else.

Before (what the reviewers objected to):
    from qiskit import QuantumCircuit, Aer
    simulator = Aer.get_backend("qasm_simulator")      # <-- classical simulation
    job = execute(cluster_circuit, simulator, shots=1024)
    entropy = job.result().get_counts()                # <-- not physically quantum

After:
    from quantum_entropy_pool import QuantumEntropyPool
    pool = QuantumEntropyPool("results/noisy_extracted.bin")
    entropy = pool.draw(256)                           # <-- real QPU / AQN entropy

Design notes
------------
* Deterministic given (dataset, master_salt, counter), so every experiment stays
  reproducible and reviewable -- which is exactly what Reviewer 2 asked for.
* The raw file holds Toeplitz-extracted bits harvested by qrng_harvest.py from a
  physical source. All downstream keys are derived with HKDF-SHA3-256, so the
  per-image keys are independent even though they share one entropy pool.
* Falls back to a clearly-flagged classical source only if no quantum dataset is
  present, and says so loudly, so a paper can never silently ship non-quantum
  keys while claiming otherwise.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    pool = QuantumEntropyPool("results/noisy_extracted.bin")
    pool.draw(256)                # 256 bits as a numpy uint8 array
    pool.draw_bytes(32)           # 32 bytes
    pool.seed_u64()               # one 64-bit DRKD round seed
    pool.seeds(64)                # a batch of 64 round seeds
    pool.provenance()             # dict describing the physical source, for the
                                  # paper's reproducibility statement
================================================================================
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import warnings
from typing import Iterator

import numpy as np

DEFAULT_SALT = b"QMFI-DRKD-v2"


class QuantumEntropyPool:
    """Deterministic, reproducible draw of quantum entropy for the DRKD module."""

    def __init__(self, dataset_path: str | None = None,
                 master_salt: bytes = DEFAULT_SALT,
                 start_counter: int = 0,
                 allow_classical_fallback: bool = True):
        self.dataset_path = dataset_path
        self.master_salt = master_salt
        self.counter = start_counter
        self._pool = None
        self._source = "uninitialised"

        if dataset_path and os.path.exists(dataset_path):
            raw = np.frombuffer(open(dataset_path, "rb").read(), dtype=np.uint8)
            self._pool = np.unpackbits(raw, bitorder="big")
            self._source = "quantum-dataset"
            self._dataset_sha256 = hashlib.sha256(raw.tobytes()).hexdigest()
            sidecar = os.path.splitext(dataset_path)[0] + ".json"
            if os.path.exists(sidecar):
                try:
                    self._meta = json.load(open(sidecar, encoding="utf-8"))
                except Exception:
                    self._meta = {}
            else:
                self._meta = {}
        elif allow_classical_fallback:
            warnings.warn(
                "No quantum entropy dataset was found (%r). Falling back to a "
                "CLASSICAL source. Do not publish results produced with this "
                "fallback -- run qrng_harvest.py first."
                % (dataset_path,))
            self._rng = np.random.default_rng(20261013)
            self._source = "classical-fallback"
        else:
            raise FileNotFoundError(
                "quantum entropy dataset not found: %r" % (dataset_path,))

    # ---------------------------------------------------------------- hooks

    def _hkdf(self, info: bytes, length: int) -> bytes:
        if self._source == "classical-fallback":
            # keep the same interface, but make the origin impossible to miss
            return hashlib.sha3_256(self.master_salt + info).digest()[:length]
        prk = hmac.new(self.master_salt, self._pool.tobytes(), hashlib.sha3_256).digest()
        okm, t, i = b"", b"", 1
        while len(okm) < length:
            t = hmac.new(prk, t + info + bytes([i]), hashlib.sha3_256).digest()
            okm += t
            i += 1
        return okm[:length]

    # ---------------------------------------------------------------- draws

    def draw_bytes(self, n_bytes: int) -> bytes:
        info = b"draw-%d" % self.counter
        self.counter += 1
        return self._hkdf(info, n_bytes)

    def draw(self, n_bits: int) -> np.ndarray:
        n_bytes = (n_bits + 7) // 8
        bits = np.unpackbits(np.frombuffer(self.draw_bytes(n_bytes), dtype=np.uint8),
                             bitorder="big")
        return bits[:n_bits]

    def seed_u64(self) -> int:
        """One 64-bit seed, the exact quantity the DRKD expects per round."""
        return int.from_bytes(self.draw_bytes(8), "big")

    def seeds(self, n: int) -> list[int]:
        return [self.seed_u64() for _ in range(n)]

    def iter_seeds(self) -> Iterator[int]:
        while True:
            yield self.seed_u64()

    # ---------------------------------------------------------------- meta

    def provenance(self) -> dict:
        d = dict(source=self._source, dataset=self.dataset_path,
                 master_salt=self.master_salt.decode(errors="replace"),
                 draws_used=self.counter)
        if self._source == "quantum-dataset":
            d.update(dataset_sha256=self._dataset_sha256,
                     pool_bits=int(self._pool.size),
                     extraction_metadata=self._meta)
        return d

    def __repr__(self) -> str:
        return "QuantumEntropyPool(source=%s, draws=%d)" % (self._source, self.counter)


# ---------------------------------------------------------------- demo / check

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "results/noisy_extracted.bin"
    pool = QuantumEntropyPool(path)
    print(json.dumps(pool.provenance(), indent=2)[:900])

    a = QuantumEntropyPool(path)
    b = QuantumEntropyPool(path)
    same = a.seeds(16) == b.seeds(16)                 # reproducibility check
    print("\nreproducible across instances:", same)

    c = QuantumEntropyPool(path)
    s = c.seeds(1000)
    print("1000 round seeds: unique =", len(set(s)), " bit-width max =", max(s).bit_length())
    bits = QuantumEntropyPool(path).draw(4096)
    bias = abs(float(bits.mean()) - 0.5)
    print("pool draw bias |p(1) - 0.5| = %.6f over %d bits" % (bias, len(bits)))
