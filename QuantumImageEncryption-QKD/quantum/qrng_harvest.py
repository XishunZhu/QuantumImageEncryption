#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
Dual-Source Quantum Entropy Harvesting, Post-Processing and Validation
================================================================================
Target paper : Adaptive Image Encryption Algorithm (Section 2.1 QMFI upgrade)
Purpose      : Replace the *classical Qiskit simulator* entropy source used in
               the submitted manuscript with physically genuine quantum entropy
               harvested from two mutually independent physical processes, and
               cross-validate them against a classical PRNG control.

Sources
  A. ibm    IBM Quantum superconducting QPU -- 4x4 grid cluster (graph) state,
            Z-basis collapse measurement. Requires QISKIT_IBM_TOKEN.
  B. aqn    ANU Quantum Numbers API        -- vacuum quadrature fluctuation.
            Requires ANU_AQN_API_KEY (free signup at quantumnumbers.anu.edu.au).
            NOTE: the legacy qrng.anu.edu.au endpoint is deprecated (1 req/min).
  C. prng   numpy PCG64 -- classical control, expected to pass but proves
            that SP 800-22 alone cannot certify quantumness.
  D. aer / noisy  local simulator (with or without a device-like noise model)
            -- no credentials needed, use this to validate the code path.

Pipeline
  raw bits -> bias / min-entropy estimation -> Toeplitz hashing over GF(2)
  -> extracted key material -> SP 800-22 battery -> paper-ready report

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  pip install qiskit qiskit-aer qiskit-ibm-runtime numpy scipy requests

  export QISKIT_IBM_TOKEN="..."
  export ANU_AQN_API_KEY="..."

  python qrng_harvest.py --source noisy --n-bits 1048576          # no credentials
  python qrng_harvest.py --source prng  --n-bits 1048576          # control
  python qrng_harvest.py --source aqn   --n-bits 1048576          # ANU
  python qrng_harvest.py --source ibm --backend ibm_brisbane      # real QPU
  python qrng_harvest.py --all --n-bits 1048576 --outdir results  # full study

IMPORTANT (IBM Open Plan): Session mode is not permitted on the free plan.
This script always submits in job mode via Sampler(mode=backend). Budget about
one minute of QPU time per 2^20 raw bits consumed from the 10-minute allowance.

CAVEAT TO STATE IN THE PAPER
  Z-basis outcomes of an ideal graph state are exactly uniform, so a classical
  model reproduces them perfectly. Statistical tests therefore CANNOT certify
  quantumness. The non-classical origin is certified separately by the CHSH
  inequality violation implemented in e91_qkd.py. Both parts are required.
================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
from dataclasses import dataclass
from math import erfc, log, sqrt
from typing import Callable

import numpy as np

try:
    from scipy.stats import norm
    from scipy.special import erfc as _erfc, gammaincc
    _HAS_SCIPY = True
except ImportError:            # pragma: no cover
    _HAS_SCIPY = False
    from math import erfc as _erfc


# =============================================================================
# 0.  GF(2) utilities -- bit packing helpers for fast Toeplitz hashing
# =============================================================================

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _as_bits(x) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    return a & np.uint8(1)


def _pack64(bits: np.ndarray) -> np.ndarray:
    bits = _as_bits(bits)
    pad = (-len(bits)) % 64
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits, bitorder="little").view(np.uint64).astype(np.uint64)


def _parity(a: np.ndarray) -> int:
    """Parity (0 or 1) of the total number of set bits in a uint64 word array."""
    a = a.astype(np.uint64)
    try:
        cnt = np.bitwise_count(a)                       # numpy >= 2.0
    except AttributeError:                              # numpy 1.x fallback
        cnt = _POP8[a.view(np.uint8)].astype(np.uint64)
    return int(np.bitwise_xor.reduce(cnt % np.uint64(2)))


# =============================================================================
# 1.  Toeplitz universal hashing over GF(2)
# =============================================================================

def toeplitz_hash(raw_bits: np.ndarray, m: int, rng: np.random.Generator,
                  progress: bool = False) -> np.ndarray:
    """
    The random extractor specified in Section 2.1 of the paper.

        y = T . x   (mod 2)

    T is an m x L Toeplitz matrix generated from (L + m - 1) uniformly random
    bits k, with T[i, j] = k[(m - 1 - i) + j]. By the Leftover Hash Lemma, if
    the input carries min-entropy, the output is epsilon-close to uniform with

        epsilon = 2 ** (-(k_min - m) / 2)
    """
    x = _as_bits(raw_bits)
    L = len(x)
    if m <= 0 or m > L:
        raise ValueError("require 0 < m <= L, got m=%d, L=%d" % (m, L))

    n = L + m - 1
    k = rng.integers(0, 2, size=n, dtype=np.uint8)

    K = np.zeros((n + 63) // 64 + 2, dtype=np.uint64)
    kp = np.packbits(np.concatenate([k, np.zeros((-n) % 64, dtype=np.uint8)]),
                     bitorder="little").view(np.uint64).astype(np.uint64)
    K[:len(kp)] = kp

    X = _pack64(x)
    Lw = len(X)
    tail = L % 64
    tailmask = np.uint64((1 << tail) - 1) if tail else _MASK64

    out_words = np.zeros((m + 63) // 64, dtype=np.uint64)
    t0 = time.time()
    for i in range(m):
        s = (m - 1) - i                       # bit offset of row i inside k
        q, r = divmod(s, 64)
        if r == 0:
            w = K[q:q + Lw].copy()
        else:
            w = (K[q:q + Lw] >> np.uint64(r)) | (K[q + 1:q + Lw + 1] << np.uint64(64 - r))
        w[-1] &= tailmask
        if _parity(X & w):
            wi, bi = divmod(i, 64)
            out_words[wi] |= np.uint64(1 << bi)
        if progress and i and i % 20000 == 0:
            el = time.time() - t0
            print("      toeplitz %d/%d (%.0f%%)  elapsed %.1fs"
                  % (i, m, 100.0 * i / m, el), flush=True)

    return np.unpackbits(out_words.view(np.uint8), bitorder="little")[:m]


def toeplitz_hash_naive(raw_bits: np.ndarray, m: int, rng) -> np.ndarray:
    """
    Straightforward reference implementation used only for unit testing.

    NOTE: the dtype passed to rng.integers matters. numpy uses Lemire bounded
    rejection sampling whose bit width follows the requested dtype, so drawing
    without dtype= consumes a different amount of the underlying stream and
    silently produces a *different* Toeplitz seed. Always keep both
    implementations on dtype=np.uint8.
    """
    x = _as_bits(raw_bits)
    L = len(x)
    n = L + m - 1
    k = rng.integers(0, 2, size=n, dtype=np.uint8)
    out = np.zeros(m, dtype=np.uint8)
    for i in range(m):
        out[i] = int(np.bitwise_xor.reduce(x & k[(m - 1 - i):(m - 1 - i) + L]))
    return out


def bits_to_bytes(bits: np.ndarray) -> bytes:
    b = _as_bits(bits)
    pad = (-len(b)) % 8
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    return bytes(np.packbits(b, bitorder="big").tolist())


def bytes_to_bits(data: bytes) -> np.ndarray:
    a = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(a, bitorder="big")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# =============================================================================
# 2.  Entropy diagnostics -- decide a safe extraction rate
# =============================================================================

@dataclass
class EntropyEstimate:
    n_bits: int
    bias: float
    shannon_bit: float
    min_entropy_bit: float
    von_neumann_rate: float

    def as_dict(self) -> dict:
        return dict(n_bits=self.n_bits, bias=round(self.bias, 6),
                    shannon_bit=round(self.shannon_bit, 6),
                    min_entropy_bit=round(self.min_entropy_bit, 6),
                    von_neumann_rate=round(self.von_neumann_rate, 4))


def estimate_entropy(bits: np.ndarray) -> EntropyEstimate:
    x = _as_bits(bits)
    L = len(x)
    p1 = float(x.mean())
    p0 = 1.0 - p1
    terms = [p for p in (p0, p1) if p > 0]
    shannon = -sum(p * log(p, 2) for p in terms)
    peak = max(p0, p1)
    minent = -log(peak, 2) if peak > 0 else 1.0
    return EntropyEstimate(L, abs(p1 - 0.5), shannon, minent, 2.0 * p0 * p1)


def safe_rate(est: EntropyEstimate, headroom: float = 0.5) -> float:
    """
    Conservative Toeplitz compression ratio. We keep only the min-entropy
    fraction of the raw stream, then apply extra headroom so that the leftover
    hash distance stays comfortably below 2^-128.
    """
    return max(1e-4, headroom * est.min_entropy_bit - 1e-3)


# =============================================================================
# 3.  SP 800-22 statistical test battery (self-contained subset)
# =============================================================================

@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    note: str = ""

    def passed(self, alpha: float = 0.01) -> bool:
        return self.p_value >= alpha

    def as_dict(self) -> dict:
        return dict(name=self.name, statistic=round(self.statistic, 4),
                    p_value=round(self.p_value, 6),
                    verdict="PASS" if self.passed() else "FAIL", note=self.note)


def _pm2(x):
    return 2 * _as_bits(x).astype(np.int64) - 1


def _phi(bits: np.ndarray, m: int) -> float:
    windows = np.lib.stride_tricks.sliding_window_view(bits, m)
    weights = (1 << np.arange(m - 1, -1, -1, dtype=np.int64))
    vals = windows.astype(np.int64) @ weights
    cnt = np.bincount(vals, minlength=2 ** m).astype(float)
    cnt = cnt[cnt > 0] / max(len(vals), 1)
    return float((cnt * np.log(cnt)).sum())


def _psi2(bits: np.ndarray, m: int) -> float:
    n = len(bits)
    windows = np.lib.stride_tricks.sliding_window_view(bits, m)
    weights = (1 << np.arange(m - 1, -1, -1, dtype=np.int64))
    vals = windows.astype(np.int64) @ weights
    cnt = np.bincount(vals, minlength=2 ** m).astype(float)
    return float((2 ** m / (n - m + 1)) * (cnt ** 2).sum() - (n - m + 1))


def test_frequency(bits):
    s = int(_pm2(bits).sum())
    n = len(bits)
    p = _erfc(abs(s) / sqrt(2.0 * n))
    return TestResult("Frequency (monobit)", abs(s) / sqrt(n), float(p))


def test_block_frequency(bits, M=128):
    bits = _as_bits(bits)
    nb = len(bits) // M
    blocks = bits[:nb * M].reshape(nb, M)
    prop = blocks.mean(axis=1)
    chi2 = float((4 * M * (prop - 0.5) ** 2).sum())
    p = float(gammaincc(nb / 2.0, chi2 / 2.0)) if _HAS_SCIPY else float("nan")
    return TestResult("Block frequency (M=%d)" % M, chi2, p)


def test_runs(bits):
    x = _pm2(bits)
    n = len(x)
    pi = float((x > 0).mean())
    if abs(pi - 0.5) >= 2 / sqrt(n):
        return TestResult("Runs", 0.0, 0.0, "monobit precondition failed")
    v_obs = int((x[1:] != x[:-1]).sum()) + 1
    p = float(_erfc(abs(v_obs - 2 * n * pi * (1 - pi)) /
                    (2 * sqrt(2 * n) * pi * (1 - pi))))
    return TestResult("Runs", v_obs, p, "pi=%.4f" % pi)


def _max_run_ones(vec: np.ndarray) -> int:
    """Length of the longest contiguous run of 1s inside a single vector."""
    padded = np.concatenate(([0], vec.astype(np.uint8), [0]))
    d = np.diff(padded)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    if starts.size == 0:
        return 0
    return int((ends - starts).max())


def test_longest_run(bits):
    """
    NIST SP 800-22 longest-run-of-ones test.

    Binning is M-dependent and is the classic source of false failures.
    Both the thresholds and the index base must match NIST exactly:

      M=128 ->  classes [<5,  5,  6,  7,  8,  >=9],  K=5
      M=8   ->  classes [<2,  2,  3, >=4],           K=3

    Getting this wrong shifts every block into the wrong bucket and drives the
    chi-square statistic to absurd values (e.g. 1.2e4 instead of ~13).
    """
    bits = _as_bits(bits)
    n = len(bits)
    if n >= 6272:
        M, K = 128, 5
        pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
        cut = 4                       # classes below/at 4 collapse into bin 0
    else:
        M, K = 8, 3
        pi = [0.2148, 0.3672, 0.2305, 0.1875]
        cut = 1

    nb = n // M
    blocks = bits[:nb * M].reshape(nb, M)
    counts = np.zeros(K + 1, dtype=np.int64)
    for b in range(nb):
        longest = _max_run_ones(blocks[b])
        counts[0 if longest <= cut else min(longest - cut, K)] += 1

    exp = np.array(pi) * nb
    chi2 = float((((counts - exp) ** 2) / np.maximum(exp, 1e-9)).sum())
    p = float(gammaincc(K / 2.0, chi2 / 2.0)) if _HAS_SCIPY else float("nan")
    return TestResult("Longest run of ones (M=%d)" % M, chi2, p)


def test_spectral(bits):
    x = _pm2(bits).astype(float)
    n = len(x)
    s = np.abs(np.fft.fft(x))[:n // 2]
    threshold = sqrt(2.995732274 * n)
    n0 = int((s < threshold).sum())
    expected = 0.95 * len(s)
    d = (n0 - expected) / sqrt(len(s) * 0.95 * 0.05)
    return TestResult("Spectral (DFT)", d, float(_erfc(abs(d) / sqrt(2))))


def test_approximate_entropy(bits, m=10):
    bits = _as_bits(bits)
    apen = _phi(bits, m + 1) - _phi(bits, m)
    chi2 = 2 * len(bits) * (log(2) + apen)
    p = float(gammaincc(2 ** (m - 1), chi2 / 2.0)) if _HAS_SCIPY else float("nan")
    return TestResult("Approximate entropy (m=%d)" % m, chi2, p)


def test_serial(bits, m=8):
    b = _as_bits(bits)
    pm = _psi2(b, m) - _psi2(b, m - 1)
    p = float(gammaincc(2 ** (m - 2), pm / 2.0)) if _HAS_SCIPY else float("nan")
    return TestResult("Serial (m=%d)" % m, pm, p)


def test_cumulative_sums(bits, reverse=False):
    x = _pm2(bits).astype(float)
    if reverse:
        x = x[::-1]
    zmax = float(np.abs(np.cumsum(x)).max())
    n = len(x)
    sqrtn = sqrt(n)
    lo = int(np.floor((-n / zmax + 1) / 4))
    hi = int(np.floor((n / zmax - 1) / 4))
    total = 0.0
    for k in range(lo, hi + 1):
        total += norm.cdf((4 * k + 1) * zmax / sqrtn) - norm.cdf((4 * k - 1) * zmax / sqrtn)
        total -= norm.cdf((4 * k + 3) * zmax / sqrtn) - norm.cdf((4 * k + 1) * zmax / sqrtn)
    return TestResult("Cumulative sums (%s)" % ("reverse" if reverse else "forward"),
                      zmax, float(1.0 - min(total, 1.0)))


def autocorrelation(bits, max_lag=32):
    x = _pm2(bits).astype(float)
    x = x - x.mean()
    denom = float((x ** 2).sum()) or 1.0
    return [(lag, round(float((x[:-lag] * x[lag:]).sum()) / denom, 6))
            for lag in range(1, max_lag + 1)]


def run_battery(bits: np.ndarray, label: str = "", n_max: int = 1 << 20) -> list:
    b = _as_bits(bits)[:n_max]
    tasks: list[Callable] = [
        test_frequency, test_block_frequency, test_runs, test_longest_run,
        test_spectral, test_approximate_entropy, test_serial,
        lambda x: test_cumulative_sums(x, reverse=False),
        lambda x: test_cumulative_sums(x, reverse=True),
    ]
    res = []
    for fn in tasks:
        try:
            r = fn(b)
            r.name = label + r.name
            res.append(r)
        except Exception as exc:
            res.append(TestResult(label + fn.__name__, float("nan"),
                                  float("nan"), "error: %s" % exc))
    return res


# =============================================================================
# 4.  Source A -- IBM Quantum QPU, 4x4 grid cluster (graph) state
# =============================================================================

def cluster_grid_circuit(rows: int = 4, cols: int = 4):
    """
    4x4 two-dimensional grid cluster state used in Section 2.1.

        |G> = ( prod_{(a,b) in E} CZ_ab ) |+>^(N)

    Every CZ contributes only a +/-1 phase to a computational-basis amplitude,
    so an ideal graph state gives *exactly uniform* Z-basis outcomes. The
    randomness therefore comes from the irreducible collapse of the
    superposition at measurement time, not from the entanglement pattern.
    Device noise introduces a small bias removed downstream by Toeplitz hashing.
    """
    from qiskit import QuantumCircuit
    n = rows * cols
    qc = QuantumCircuit(n, n, name="cluster_%dx%d" % (rows, cols))
    qc.h(range(n))
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols:
                qc.cz(i, i + 1)
            if r + 1 < rows:
                qc.cz(i, i + cols)
    qc.barrier()
    qc.measure(range(n), range(n))
    return qc


def _bitstrings_to_bits(bitstrings: list) -> np.ndarray:
    """
    Flatten per-shot bitstrings while PRESERVING the temporal shot order.

    Ordering matters: several SP 800-22 tests (runs, serial, approximate
    entropy) depend on the sequence, and expanding a counts dictionary instead
    would cluster identical outcomes together and give misleading results.
    """
    clean = [s.replace(" ", "") for s in bitstrings]
    if not clean:
        raise RuntimeError("no per-shot measurement results returned")
    joined = "".join(clean).encode()
    return (np.frombuffer(joined, dtype=np.uint8) - np.uint8(48)).astype(np.uint8)


def _counts_to_bits(counts: dict, width: int) -> np.ndarray:
    """
    Fallback when per-shot memory is unavailable: expand a counts dict.

    WARNING: this loses the original shot ordering. Use only when memory is not
    offered by the backend, and prefer running the ordering-sensitive tests on
    a memory-ordered stream.
    """
    chunks = []
    for bitstring, cnt in counts.items():
        key = bitstring.replace(" ", "")
        row = (np.frombuffer(key.encode(), dtype=np.uint8) - np.uint8(48)).astype(np.uint8)
        chunks.append(np.tile(row[None, :], (int(cnt), 1)).reshape(-1))
    if not chunks:
        raise RuntimeError("no measurement results returned")
    return np.concatenate(chunks)


def _pick_method(obj, *names):
    """Return the first existing attribute among names, evaluated lazily."""
    for n in names:
        fn = getattr(obj, n, None)
        if fn is not None:
            return fn
    raise AttributeError("none of %s available on %s"
                         % (list(names), type(obj).__name__))


def harvest_local(n_bits: int, shots: int = 4096, noise: bool = False,
                  seed: int = 20261013, verbose: bool = True) -> dict:
    """Aer local simulation, optionally with an IBM-device-like noise model."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (NoiseModel, depolarizing_error,
                                  thermal_relaxation_error, ReadoutError)

    n_qubits = 16
    qc = cluster_grid_circuit(4, 4)
    n_jobs = max(1, int(np.ceil(n_bits / (shots * n_qubits))))

    if noise:
        nm = NoiseModel()
        # qiskit-aer renamed these helpers to singular between releases; pick
        # whichever exists. Note getattr() evaluates its default eagerly, so
        # a lazy helper is required here.
        add_qerr = _pick_method(nm, "add_all_qubit_quantum_error",
                                "add_all_qubits_quantum_error")
        add_rerr = _pick_method(nm, "add_all_qubit_readout_error",
                                "add_all_qubits_readout_error")
        # Roughly mirrors a Heron-class superconducting device.
        # The readout asymmetry is the point: on real hardware P(measure 0 |
        # prepared 1) is 3-5x P(measure 1 | prepared 0), because the excited
        # state T1-decays towards |0>. A SYMMETRIC readout error adds noise
        # but no bias, which would silently hide the very effect that decides
        # how much entropy the Toeplitz stage can actually extract.
        p1q, p2q = 3.0e-4, 8.0e-3
        p01, p10 = 8.0e-3, 2.5e-2      # |0>->|1> and |1>->|0> readout error
        t1, t2 = 250e3, 180e3          # ns, Heron-class
        dt1q, dt2q = 60.0, 400.0       # ns
        add_qerr(thermal_relaxation_error(t1, t2, dt1q).compose(depolarizing_error(p1q, 1)), ["h"])
        add_qerr(thermal_relaxation_error(t1, t2, dt2q).compose(depolarizing_error(p2q, 2)), ["cz"])
        add_rerr(ReadoutError([[1 - p01, p01], [p10, 1 - p10]]))
        sim = AerSimulator(noise_model=nm, seed_simulator=seed)
    else:
        sim = AerSimulator(seed_simulator=seed)

    isa = transpile(qc, sim, optimization_level=1)
    if verbose:
        print("    ISA circuit depth=%d  ops=%s" % (isa.depth(), dict(isa.count_ops())),
              flush=True)

    t0 = time.time()
    total_shots = shots * n_jobs
    job = sim.run(isa, shots=total_shots, memory=True)
    res = job.result()
    try:
        bits = _bitstrings_to_bits(res.get_memory())
    except Exception:
        bits = _counts_to_bits(res.get_counts(), n_qubits)
    bits = bits[:n_bits]
    meta = dict(source="noisy-aer" if noise else "aer",
                backend="AerSimulator(%s)" % ("device-like-noise" if noise else "ideal"),
                n_jobs=1, shots_per_job=shots * n_jobs, wall_time_s=round(time.time() - t0, 2),
                transpiled_depth=int(isa.depth()),
                transpiled_ops={k: int(v) for k, v in isa.count_ops().items()},
                circuit="4x4 grid cluster state, H layer + 24 CZ edges, Z-basis measurement")
    return dict(bits=bits, meta=meta)


def harvest_ibm_qpu(n_bits: int, backend_name: str | None = None,
                    shots: int = 4096, optimization_level: int = 1,
                    verbose: bool = True) -> dict:
    """
    Harvest from a real IBM Quantum processor (job mode, Open Plan compatible).

    Set credentials either via the QISKIT_IBM_TOKEN environment variable or by
    running QiskitRuntimeService.save_account() once beforehand.
    """
    from qiskit import QuantumCircuit
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    except ImportError as exc:
        raise RuntimeError("pip install qiskit-ibm-runtime to use --source ibm") from exc

    token = os.environ.get("QISKIT_IBM_TOKEN")
    if not token:
        try:
            saved = QiskitRuntimeService.saved_accounts()
        except Exception:
            saved = {}
        if not saved:
            raise RuntimeError(
                "no IBM Quantum credential found.\n"
                "  1) free account + API token at https://quantum.cloud.ibm.com\n"
                "  2) supply it in one of three ways:\n"
                "       export QISKIT_IBM_TOKEN=<token>          # this shell only\n"
                "       setx QISKIT_IBM_TOKEN <token>            # persist (new shell)\n"
                "       python -c \"from qiskit_ibm_runtime import "
                "QiskitRuntimeService as S; S.save_account(token='<token>')\"\n"
                "  Open Plan: ~10 min QPU per 28 days, job mode only, which is\n"
                "  exactly what Sampler(mode=backend) below uses.")

    try:
        service = (QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
                   if token else QiskitRuntimeService())
    except Exception as exc:                       # bad token, offline, DNS ...
        raise RuntimeError("could not authenticate to IBM Quantum: %s" % exc) from exc

    def _pick_backend():
        """least_busy() iterates every visible backend and calls status() on
        each -- a single flaky backend kills the whole call, which is what
        broke the first harvest attempt. Fall back to a fixed list (ibm_fez
        is known-good for this account) before giving up."""
        if backend_name:
            return service.backend(backend_name), "explicit --backend"
        try:
            return (service.least_busy(operational=True, simulator=False,
                                       min_num_qubits=16),
                    "least_busy(>=16q)")
        except Exception as exc:
            print("    least_busy failed (%s); trying a fixed list"
                  % str(exc)[:100], flush=True)
        for name in ("ibm_fez", "ibm_torino", "ibm_marrakesh",
                     "ibm_brisbane", "ibm_sherbrooke"):
            try:
                b = service.backend(name)
                if b.status().operational:
                    return b, "fixed list (%s)" % name
            except Exception:
                continue
        for b in service.backends(simulator=False, operational=True,
                                  min_num_qubits=16):
            return b, "first operational >=16q"
        raise RuntimeError("no operational >=16q backend visible to this account")

    try:
        backend, how = _pick_backend()
    except Exception as exc:
        raise RuntimeError(
            "authenticated, but no usable backend: %s\n"
            "  pick one explicitly with --backend, e.g. ibm_fez,\n"
            "  ibm_torino, ibm_brisbane. Check availability in the dashboard."
            % exc) from exc
    print("    selected backend: %s  (via %s)" % (backend.name, how), flush=True)

    n_qubits = 16
    qc = cluster_grid_circuit(4, 4)
    print("    transpiling for %s ..." % backend.name, flush=True)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
    isa = pm.run(qc)

    per_job = shots * n_qubits
    n_jobs = max(1, int(np.ceil(n_bits / per_job)))
    if verbose:
        print("    ISA depth=%d  ops=%s  jobs=%d x %d shots"
              % (isa.depth(), dict(isa.count_ops()), n_jobs, shots), flush=True)
        print("    estimated QPU-seconds ~ %.1f  (check your monthly allowance)"
              % (n_jobs * 2.0), flush=True)

    sampler = Sampler(mode=backend)          # job mode: allowed on Open Plan
    sampler.options.default_shots = shots
    t0 = time.time()
    job = sampler.run([isa] * n_jobs)
    print("    job id: %s  (queued at %s)" % (job.job_id(), time.strftime("%H:%M:%S")), flush=True)
    result = job.result()
    print("    job finished in %.1fs" % (time.time() - t0), flush=True)

    # SamplerV2 stores the measurement BitArray under the *classical register
    # name*, which for QuantumCircuit(n, n) is "c" -- NOT "meas". The old code
    # accessed pub.data.meas in both the primary and the fallback branch, so
    # the AttributeError was raised twice and surfaced only after the QPU
    # time had already been spent. Resolve the field by name instead.
    all_bits = []
    for pub in result:
        data = pub.data
        field = None
        for cand in ("c", "meas"):
            if hasattr(data, cand):
                field = getattr(data, cand)
                break
        if field is None:                    # unknown creg name: first BitArray
            for name in dir(data):
                if name.startswith("_"):
                    continue
                obj = getattr(data, name)
                if hasattr(obj, "get_bitstrings"):
                    field = obj
                    break
        if field is None:
            raise RuntimeError(
                "no measurement field in SamplerV2 result (available: %s)"
                % [n for n in dir(data) if not n.startswith("_")])
        try:
            all_bits.append(_bitstrings_to_bits(list(field.get_bitstrings())))
        except Exception:
            all_bits.append(_counts_to_bits(field.get_counts(), n_qubits))
    bits = np.concatenate(all_bits)[:n_bits]

    # Calibration metadata is nice-to-have: never let it kill a job whose QPU
    # time has already been spent. props.t1(q) can also return None for qubits
    # with incomplete calibration, which used to crash np.median().
    try:
        cfg = backend.configuration()
        props = backend.properties()
        nq = len(props.qubits) if (props and props.qubits is not None) else 0
        tq = [t for t in (props.t1(q) for q in range(min(16, nq))) if t]
        median_t1_us = round(float(np.median(tq)) * 1e6, 2) if tq else None
        calib = str(getattr(props, "last_update_date", "")) if props else ""
        nq_proc = int(getattr(cfg, "n_qubits", -1)) if cfg else -1
    except Exception as exc:
        print("    WARNING: calibration metadata unavailable: %s"
              % str(exc)[:100], flush=True)
        median_t1_us, calib, nq_proc = None, "", -1
    meta = dict(source="ibm-qpu", backend=backend.name,
                n_qubits_processor=nq_proc,
                shots_per_job=shots, n_jobs=n_jobs,
                wall_time_s=round(time.time() - t0, 2),
                job_id=str(job.job_id()),
                transpiled_depth=int(isa.depth()),
                transpiled_ops={k: int(v) for k, v in isa.count_ops().items()},
                median_t1_us=median_t1_us,
                calibration_time=calib,
                circuit="4x4 grid cluster state, H layer + 24 CZ edges, Z-basis measurement")
    return dict(bits=bits, meta=meta)


# =============================================================================
# 5.  Source B -- ANU Quantum Numbers (vacuum fluctuation) API
# =============================================================================

AQN_URL = "https://api.quantumnumbers.anu.edu.au"
LEGACY_URL = "https://qrng.anu.edu.au/API/jsonI.php"     # deprecated: 1 req/min


def harvest_aqn(n_bits: int, legacy: bool = False, verbose: bool = True,
               throttle: float = 0.5, max_retries: int = 25,
               checkpoint: str = None) -> dict:
    """
    Fetch quantum random bytes generated by measuring vacuum-field quadrature
    fluctuations at the ANU physics laboratory.

    Modern endpoint (needs a free API key, 1024 values per request):
        https://api.quantumnumbers.anu.edu.au?length=1024&type=uint8
        header: x-api-key

    Robustness:
      * ``throttle`` inserts a fixed delay between requests to stay under the
        free tier's rate limit (the default 0.5 s is ~120 req/min).
      * ``max_retries`` + exponential backoff (capped at 30 s) absorb transient
        429s instead of aborting.
      * ``checkpoint`` (a file path) makes the fetch resumable: successfully
        fetched bytes are persisted every 10 requests, and on a hard failure the
        partial buffer is saved so a re-run continues from where it stopped
        rather than discarding everything. The checkpoint is removed on success.
    """
    import requests

    n_bytes = int(np.ceil(n_bits / 8))
    key = os.environ.get("ANU_AQN_API_KEY")
    if legacy:
        url, headers, per_req = LEGACY_URL, None, 1024
    else:
        if not key:
            raise RuntimeError("set ANU_AQN_API_KEY (free key: quantumnumbers.anu.edu.au)")
        url, headers, per_req = AQN_URL, {"x-api-key": key}, 1024

    buf = bytearray()
    if checkpoint and os.path.exists(checkpoint):
        with open(checkpoint, "rb") as fh:
            buf = bytearray(fh.read())
        if verbose:
            print("    resumed from checkpoint: %d/%d bytes"
                  % (len(buf), n_bytes), flush=True)

    t0 = time.time()
    session = requests.Session()
    req_done = 0
    while len(buf) < n_bytes:
        want = min(per_req, n_bytes - len(buf))
        params = {"length": int(want), "type": "uint8"}
        backoff, last_err = 1.0, None
        for attempt in range(max_retries):
            try:
                r = session.get(url, params=params, headers=headers, timeout=30)
                r.raise_for_status()
                payload = r.json()
                if not payload.get("success", False):
                    raise RuntimeError("AQN reported failure: %s" % payload)
                data = payload.get("data", [])
                if len(data) != want:
                    raise RuntimeError("AQN short response: asked %d bytes, got %d"
                                       % (want, len(data)))
                buf.extend(bytes(data))
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt == max_retries - 1:
                    break
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
        if last_err is not None:
            if checkpoint:
                with open(checkpoint, "wb") as fh:
                    fh.write(bytes(buf))
                if verbose:
                    print("    saved checkpoint (%d bytes) before abort"
                          % len(buf), flush=True)
            raise RuntimeError(
                "AQN request failed after %d retries: %s"
                % (max_retries, last_err)) from last_err
        req_done += 1
        if checkpoint and req_done % 10 == 0:
            with open(checkpoint, "wb") as fh:
                fh.write(bytes(buf))
        if verbose and req_done % 10 == 0:
            print("    AQN %d bytes (%.1f%%)"
                  % (len(buf), 100.0 * len(buf) / n_bytes), flush=True)
        time.sleep(1.5 if legacy else throttle)

    if checkpoint and os.path.exists(checkpoint):
        try:
            os.remove(checkpoint)
        except OSError:
            pass

    bits = bytes_to_bits(bytes(buf))[:n_bits]
    meta = dict(source="anu-aqn" + ("-legacy" if legacy else ""),
                backend=url, n_requests=req_done,
                physical_process="vacuum electromagnetic field quadrature measurement",
                n_bits=len(bits), wall_time_s=round(time.time() - t0, 2))
    return dict(bits=bits, meta=meta)

    bits = bytes_to_bits(bytes(buf))[:n_bits]
    meta = dict(source="anu-aqn" + ("-legacy" if legacy else ""),
                backend=url, n_requests=n_req,
                physical_process="vacuum electromagnetic field quadrature measurement",
                n_bits=len(bits), wall_time_s=round(time.time() - t0, 2))
    return dict(bits=bits, meta=meta)


# =============================================================================
# 6.  Source C -- classical PRNG control
# =============================================================================

def harvest_prng(n_bits: int, seed: int = 20261013) -> dict:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
    meta = dict(source="prng-control", backend="numpy PCG64", seed=seed,
                note="classical negative control")
    return dict(bits=bits, meta=meta)


# =============================================================================
# 7.  Key material derivation for the encryption pipeline
# =============================================================================

def hkdf_sha3(extracted: bytes, info: bytes, length: int = 32,
              salt: bytes = b"QMFI-DRKD-v2") -> bytes:
    """HKDF built on SHA3-256, matching the hash family used by the DRKD module."""
    prk = hmac.new(salt, extracted, hashlib.sha3_256).digest()
    okm, t, counter = b"", b"", 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha3_256).digest()
        okm += t
        counter += 1
    return okm[:length]


def derive_round_seeds(extracted_bits: np.ndarray, n_seeds: int = 256,
                       context: str = "round") -> list:
    """Derive one 64-bit DRKD round seed per request from the harvested pool."""
    pool = bits_to_bytes(extracted_bits)
    return [int.from_bytes(hkdf_sha3(pool, b"%s-%d" % (context.encode(), i), 8), "big")
            for i in range(n_seeds)]


# =============================================================================
# 8.  Reporting
# =============================================================================

@dataclass
class SourceReport:
    source: str
    meta: dict
    raw_entropy: dict
    rate: float
    extracted_bits: int
    raw_sha256: str
    extracted_sha256: str
    tests_raw: list
    tests_extracted: list
    autocorr_max: float

    def as_dict(self) -> dict:
        return dict(source=self.source, meta=self.meta, raw_entropy=self.raw_entropy,
                    extraction_rate=round(self.rate, 4),
                    extracted_bits=self.extracted_bits,
                    raw_sha256=self.raw_sha256, extracted_sha256=self.extracted_sha256,
                    tests_raw=[t.as_dict() for t in self.tests_raw],
                    tests_extracted=[t.as_dict() for t in self.tests_extracted],
                    max_autocorr_abs=round(self.autocorr_max, 6))


def process_source(name: str, harvest_fn: Callable, outdir: str,
                   n_bits: int, seed: int, rate: float | None,
                   save: bool = True) -> SourceReport:
    print("\n=== source: %s ===" % name, flush=True)
    got = harvest_fn()
    bits = _as_bits(got["bits"])
    meta = got["meta"]
    print("    harvested %d raw bits" % len(bits), flush=True)

    est = estimate_entropy(bits)
    print("    bias=%.5f  H=%.6f bit  H_min=%.6f bit" %
          (est.bias, est.shannon_bit, est.min_entropy_bit), flush=True)

    used_rate = rate if rate is not None else safe_rate(est)
    m = int(len(bits) * used_rate)
    print("    Toeplitz: L=%d -> m=%d (rate %.4f)" % (len(bits), m, used_rate), flush=True)

    t0 = time.time()
    rng = np.random.default_rng(seed)
    extracted = toeplitz_hash(bits, m, rng, progress=True)
    print("    extraction took %.1fs" % (time.time() - t0), flush=True)

    raw_bytes, ext_bytes = bits_to_bytes(bits), bits_to_bytes(extracted)
    if save:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "%s_raw.bin" % name), "wb") as fh:
            fh.write(raw_bytes)
        with open(os.path.join(outdir, "%s_extracted.bin" % name), "wb") as fh:
            fh.write(ext_bytes)

    ac = autocorrelation(extracted, 32)
    acmax = max(abs(v) for _, v in ac)
    rep = SourceReport(name, meta, est.as_dict(), used_rate, m,
                       sha256_hex(raw_bytes), sha256_hex(ext_bytes),
                       run_battery(bits, label="[raw] "),
                       run_battery(extracted, label="[ext] "),
                       acmax)

    n_fail = sum(0 if t.passed() else 1 for t in rep.tests_extracted)
    print("    extracted stream: %d/%d tests passed, max|autocorr|=%.6f"
          % (len(rep.tests_extracted) - n_fail, len(rep.tests_extracted), acmax),
          flush=True)
    return rep


def md_table(reports: list) -> str:
    names = [r.source for r in reports]
    head = "| Test | " + " | ".join(names) + " |"
    sep = "|---|" + "---|" * len(names)
    rows = [head, sep]
    for i, t0 in enumerate(reports[0].tests_extracted):
        base = t0.name.replace("[ext] ", "")
        cells = []
        for r in reports:
            t = r.tests_extracted[i]
            cells.append("%.4f (%s)" % (t.p_value, "PASS" if t.passed() else "FAIL"))
        rows.append("| %s | " % base + " | ".join(cells) + " |")
    rows.append("| Bias of raw stream | " +
                " | ".join("%.5f" % r.raw_entropy["bias"] for r in reports) + " |")
    rows.append("| Min-entropy / bit (raw) | " +
                " | ".join("%.6f" % r.raw_entropy["min_entropy_bit"] for r in reports) + " |")
    rows.append("| Extraction rate | " +
                " | ".join("%.4f" % r.rate for r in reports) + " |")
    rows.append("| Max abs. autocorrelation | " +
                " | ".join("%.6f" % r.autocorr_max for r in reports) + " |")
    return "\n".join(rows)


# =============================================================================
# 9.  Self-test
# =============================================================================

def self_test():
    print("self-test: Toeplitz fast implementation vs naive reference")
    rng = np.random.default_rng(7)
    for L, m in ((512, 64), (4096, 331), (8192, 1024)):
        x = rng.integers(0, 2, size=L, dtype=np.uint8)
        fast = toeplitz_hash(x, m, np.random.default_rng(11))
        naive = toeplitz_hash_naive(x, m, np.random.default_rng(11))
        ok = np.array_equal(fast, naive) and len(fast) == m
        print("  L=%5d m=%5d -> %s" % (L, m, "OK" if ok else "MISMATCH"))
        assert ok, "Toeplitz implementation is wrong at L=%d m=%d" % (L, m)

    print("self-test: biased input should be corrected after extraction")
    rng = np.random.default_rng(3)
    biased = (rng.random(400000) < 0.62).astype(np.uint8)
    est = estimate_entropy(biased)
    print("  raw bias=%.4f  H_min=%.4f" % (est.bias, est.min_entropy_bit))
    out = toeplitz_hash(biased, int(len(biased) * 0.45), np.random.default_rng(5))
    est2 = estimate_entropy(out)
    print("  extracted bias=%.6f" % est2.bias)
    assert abs(est2.bias) < 0.005, "bias survived extraction"
    print("self-test PASSED\n", flush=True)


# =============================================================================
# 10.  CLI
# =============================================================================

SOURCE_BUILDERS = {
    "aer": lambda n, a: (lambda: harvest_local(n, noise=False, seed=a.seed)),
    "noisy": lambda n, a: (lambda: harvest_local(n, noise=True, seed=a.seed)),
    "prng": lambda n, a: (lambda: harvest_prng(n, seed=a.seed)),
    "aqn": lambda n, a: (lambda: harvest_aqn(n, legacy=a.legacy)),
    "ibm": lambda n, a: (lambda: harvest_ibm_qpu(n, backend_name=a.backend, shots=a.shots)),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dual-source quantum entropy harvesting")
    ap.add_argument("--source", choices=sorted(SOURCE_BUILDERS), default="noisy")
    ap.add_argument("--all", action="store_true",
                    help="run noisy + prng + every credentialed source available")
    ap.add_argument("--n-bits", type=int, default=1 << 20)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--rate", type=float, default=None,
                    help="extraction rate; default = safe_rate estimate")
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--backend", default=None, help="IBM backend name")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--legacy", action="store_true", help="use deprecated ANU endpoint")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if a.self_test:
        self_test()
        return 0

    if a.all:
        names = ["noisy", "prng"]
        if os.environ.get("ANU_AQN_API_KEY"):
            names.append("aqn")
        if os.environ.get("QISKIT_IBM_TOKEN"):
            names.append("ibm")
    else:
        names = [a.source]

    os.makedirs(a.outdir, exist_ok=True)
    reports = []
    for name in names:
        builder = SOURCE_BUILDERS[name](a.n_bits, a)
        try:
            reports.append(process_source(name, builder, a.outdir,
                                          a.n_bits, a.seed, a.rate))
        except Exception as exc:
            print("  !! source '%s' failed: %s" % (name, exc), file=sys.stderr, flush=True)

    if not reports:
        print("no source produced data", file=sys.stderr)
        return 1

    payload = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                   n_bits_requested=a.n_bits,
                   reports=[r.as_dict() for r in reports])
    with open(os.path.join(a.outdir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    table = md_table(reports)
    with open(os.path.join(a.outdir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("# Quantum entropy cross-validation\n\n")
        fh.write(table + "\n")
    print("\n" + table)
    print("\nwrote %s/report.json and %s/report.md" % (a.outdir, a.outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
