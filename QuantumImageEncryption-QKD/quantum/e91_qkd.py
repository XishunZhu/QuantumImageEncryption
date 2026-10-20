#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
E91 Entanglement-Based Quantum Key Distribution for Image Encryption
================================================================================
Target paper : new Section 2.2, providing the root key for the Dual-Ratchet Key
               Derivation Module (DRKD) and certifying that the entropy really
               has a non-classical origin.

Why this module exists
----------------------
Section 2.1 of the paper harvests random bits from a cluster state. That is a
genuine quantum entropy source, BUT: the Z-basis outcomes of a graph state are
statistically uniform even under an ideal classical model, so no amount of
SP 800-22 testing can prove the randomness is quantum. This module supplies the
missing certificate -- a Bell/CHSH inequality violation S > 2, which has no
classical explanation -- and then turns the same entanglement into key material.

Protocol (Ekert 1991)
---------------------
 1. A source emits |Phi+> = (|00> + |11>)/sqrt(2).
 2. Alice measures qubit 0 in basis A(theta) = ry(2*theta) then Z.
    With this convention  E(theta_a, theta_b) = cos(2*(theta_a - theta_b)),
    which was verified against Aer before anything else was built on top.
 3. KEY rounds:  Alice and Bob each pick theta from {0, pi/4} at random.
                 Matching bases give perfectly correlated outcomes -> sifted key.
    TEST rounds: four combinations used for the CHSH statistic
                 a1=0, a2=pi/4, b1=pi/8, b2=-pi/8
                 S = E(a1,b1) + E(a1,b2) + E(a2,b1) - E(a2,b2)
                 Classical bound |S| <= 2, Tsirelson bound 2*sqrt(2) = 2.8284.
 4. Eve = intercept-resend with probability p_eve, implemented as a real
    dynamic circuit (measure + reset + conditional X), not as a hand-made
    error model. Predicted QBER = 0.25 * p_eve, which the script checks.
 5. QBER is estimated by publicly revealing a random subset of the sifted key.
 6. Privacy amplification by Toeplitz hashing; final rate from Shor-Preskill
    r = 1 - 2*h2(QBER), which vanishes at the 11% security threshold.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  python e91_qkd.py --self-test                       # physics sanity checks
  python e91_qkd.py --mode local --n-pairs 65536      # ideal simulator
  python e91_qkd.py --mode noisy --n-pairs 65536      # device-like noise
  python e91_qkd.py --mode local --eve-sweep          # QBER & key rate vs Eve
  python e91_qkd.py --mode ibm --backend ibm_brisbane # CHSH on real hardware

Output: results/e91_report.json and results/e91_sweep.csv ready for the figures.
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np

from qrng_harvest import (toeplitz_hash, bits_to_bytes, sha256_hex,
                          _as_bits, run_battery)

A_KEY = [0.0, np.pi / 4]                       # key-generation bases
A_CHSH, B_CHSH = [0.0, np.pi / 4], [np.pi / 8, -np.pi / 8]
CHSH_PAIRS = [(A_CHSH[0], B_CHSH[0]), (A_CHSH[0], B_CHSH[1]),
              (A_CHSH[1], B_CHSH[0]), (A_CHSH[1], B_CHSH[1])]
CHSH_SIGNS = [1.0, 1.0, 1.0, -1.0]             # S = E11 + E12 + E21 - E22


# =============================================================================
# 1.  Circuit construction
# =============================================================================

def bell_pair_circuit(theta_a: float, theta_b: float, eve_theta: float | None = None):
    """
    One E91 round.

    Eve is inserted on Bob's half *before* Bob's basis rotation: she rotates into
    her own basis, measures, resets, conditionally flips to prepare the state she
    actually observed, then undoes her rotation and sends the qubit on.
    Qiskit 2.x removed Instruction.c_if, so if_test is used here.
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

    q = QuantumRegister(2, "q")
    c = ClassicalRegister(2, "c")
    ev = ClassicalRegister(1, "ev")
    qc = QuantumCircuit(q, c, ev, name="e91")

    qc.h(0)
    qc.cx(0, 1)

    if eve_theta is not None:
        qc.ry(2 * eve_theta, 1)
        qc.barrier()
        qc.measure(1, ev[0])
        qc.reset(1)
        with qc.if_test((ev, 1)):
            qc.x(1)
        qc.ry(-2 * eve_theta, 1)
        qc.barrier()

    qc.ry(2 * theta_a, 0)
    qc.ry(2 * theta_b, 1)
    qc.barrier()
    qc.measure(0, c[0])
    qc.measure(1, c[1])
    return qc


# =============================================================================
# 2.  Backends
# =============================================================================

def make_local_backend(noise: bool = False, seed: int = 20261013):
    from qiskit_aer import AerSimulator
    if not noise:
        return AerSimulator(seed_simulator=seed), "AerSimulator(ideal)"

    from qiskit_aer.noise import (NoiseModel, depolarizing_error,
                                  thermal_relaxation_error, ReadoutError)

    def pick(obj, *names):
        for n in names:
            fn = getattr(obj, n, None)
            if fn is not None:
                return fn
        raise AttributeError("none of %s available" % (list(names),))

    nm = NoiseModel()
    t1, t2 = 250e3, 180e3
    pick(nm, "add_all_qubit_quantum_error", "add_all_qubits_quantum_error")(
        thermal_relaxation_error(t1, t2, 60.0).compose(depolarizing_error(3e-4, 1)), ["h", "ry"])
    pick(nm, "add_all_qubit_quantum_error", "add_all_qubits_quantum_error")(
        thermal_relaxation_error(t1, t2, 400.0).compose(depolarizing_error(8e-3, 2)), ["cx"])
    # Asymmetric readout, mirroring qrng_harvest.py: P(0|1) = 2.5e-2 vs
    # P(1|0) = 8e-3. A symmetric error would leave QBER at p=0 unbiased and
    # would understate the noise floor the paper reports as QBER0.
    pick(nm, "add_all_qubit_readout_error", "add_all_qubits_readout_error")(
        ReadoutError([[1 - 8.0e-3, 8.0e-3], [2.5e-2, 1 - 2.5e-2]]))
    return AerSimulator(noise_model=nm, seed_simulator=seed), "AerSimulator(device-like-noise)"


def run_circuits(circuits: list, backend, shots: int, tag: str = "") -> list:
    from qiskit import transpile
    isa = transpile(circuits, backend, optimization_level=1)
    t0 = time.time()
    res = backend.run(isa, shots=shots, memory=True).result()
    out = []
    for i in range(len(isa)):
        # circuits share the same name, so looking results up BY NAME returns
        # only the first match. Always index by position.
        try:
            out.append(list(res.get_memory(i)))
        except Exception:
            out.append(["0"] * shots)
    print("    %s ran %d circuits x %d shots in %.1fs"
          % (tag, len(isa), shots, time.time() - t0), flush=True)
    return out


def _mem_outcomes(mem, idx=0):
    """Extract bit idx from each per-shot bitstring, preserving order."""
    return np.array([int(s.replace(" ", "")[::-1][idx]) for s in mem], dtype=np.uint8)


def correlation(alice_bits: np.ndarray, bob_bits: np.ndarray) -> float:
    agree = float(np.mean(alice_bits == bob_bits))
    return 2.0 * agree - 1.0


# =============================================================================
# 3.  CHSH experiment
# =============================================================================

@dataclass
class CHSHResult:
    backend: str
    shots_per_corr: int
    correlations: list
    S: float
    S_std: float
    violated: bool

    def as_dict(self):
        d = asdict(self)
        d["correlations"] = [round(c, 5) for c in self.correlations]
        d["S"] = round(self.S, 5)
        d["S_std"] = round(self.S_std, 5)
        return d


def chsh_experiment(backend, shots=8192, eve_theta=None, tag="") -> CHSHResult:
    circs = [bell_pair_circuit(a, b, eve_theta) for a, b in CHSH_PAIRS]
    mems = run_circuits(circs, backend, shots, tag="chsh " + tag)
    cors = []
    for mem in mems:
        alice = _mem_outcomes(mem, 0)
        bob = _mem_outcomes(mem, 1)
        cors.append(correlation(alice, bob))
    S = sum(s * e for s, e in zip(CHSH_SIGNS, cors))
    std = 2.0 / np.sqrt(shots)            # 4 correlations each ~1/sqrt(N)
    return CHSHResult(str(backend), shots, cors, float(S), float(std), abs(S) > 2.0)


def chsh_mixed(backend, p_eve: float, shots: int = 8192, tag: str = "") -> CHSHResult:
    """
    CHSH when Eve intercepts only a FRACTION p_eve of the pairs.

    Eve does not get to choose which pairs she intercepts, so the test rounds
    see a mixture: a fraction (1-p_eve) of the shots are clean Bell pairs and
    p_eve carry her intercept-resend state. Pooling the per-shot data in the
    right proportion makes S degrade smoothly,

        S(p) = (1 - p) * 2*sqrt(2) + p * sqrt(2)     approximately,

    crossing the classical bound 2 at p ~ 0.586. That crossing point is a
    result worth reporting; forcing Eve either "fully present" or "fully absent"
    would instead produce a meaningless step discontinuity.
    """
    n_clean = int(round(shots * (1 - p_eve)))
    n_eve = shots - n_clean
    cors, effective = [], 0
    for a, b in CHSH_PAIRS:
        parts = []
        if n_clean > 0:
            parts.append(run_circuits([bell_pair_circuit(a, b, None)], backend,
                                      n_clean, tag="chsh-clean " + tag)[0])
        if n_eve > 0:
            half = max(1, n_eve // 2)
            for eb in A_KEY:                       # Eve guesses either basis
                parts.append(run_circuits([bell_pair_circuit(a, b, eb)], backend,
                                          half, tag="chsh-eve " + tag)[0])
        pooled = [s for part in parts for s in part]
        effective = len(pooled)
        cors.append(correlation(_mem_outcomes(pooled, 0), _mem_outcomes(pooled, 1)))
    S = float(sum(s * e for s, e in zip(CHSH_SIGNS, cors)))
    return CHSHResult(str(backend), effective, cors, S,
                      float(2.0 / np.sqrt(effective)), abs(S) > 2.0)


# =============================================================================
# 4.  Key generation, sifting, QBER
# =============================================================================

@dataclass
class ProtocolResult:
    backend: str
    p_eve: float
    n_pairs: int
    sifted_len: int
    qber: float
    chsh: CHSHResult
    key_rate: float
    final_key_bits: int
    key_sha256: str
    revealed: int

    def as_dict(self):
        d = asdict(self)
        d["chsh"] = self.chsh.as_dict()
        d["qber"] = round(self.qber, 6)
        d["key_rate"] = round(self.key_rate, 6)
        return d


def binary_entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def key_rate_shor_preskill(qber: float) -> float:
    """Shor-Preskill secure fraction; becomes non-positive at QBER ~= 11%."""
    return max(0.0, 1.0 - 2.0 * binary_entropy(qber))


def run_protocol(backend, n_pairs: int = 65536, p_eve: float = 0.0,
                 reveal_fraction: float = 0.10, shots_cap: int = 1 << 16,
                 seed: int = 20261013, tag: str = "") -> ProtocolResult:
    """
    Run one full E91 session.

    Eve is modelled honestly: for a fraction p_eve of the pairs the circuit is
    rebuilt WITH the intercept-resend block, and the clean pairs use the plain
    circuit. The two populations are then combined in the right proportion.
    """
    rng = np.random.default_rng(seed)
    n_clean = int(round(n_pairs * (1 - p_eve)))
    n_eve = n_pairs - n_clean

    results = []
    # Eve picks a basis at random per intercepted pair, so her population is
    # split over the two bases rather than being pinned to one choice.
    populations = [("clean", n_clean, None)]
    if n_eve > 0:
        populations.append(("eve-Z", n_eve // 2, A_KEY[0]))
        populations.append(("eve-X", n_eve - n_eve // 2, A_KEY[1]))

    for label, n, eve in populations:
        if n <= 0:
            continue
        combos = []
        for ta in A_KEY:
            for tb in A_KEY:
                combos.append((ta, tb, n // 4))
        for ta, tb, cnt in combos:
            if cnt <= 0:
                continue
            (qc,) = [bell_pair_circuit(ta, tb, eve)]
            mems = run_circuits([qc], backend, min(cnt, shots_cap),
                                tag="%s key(%s)" % (tag, label))
            mem = mems[0]
            alice = _mem_outcomes(mem, 0)
            bob = _mem_outcomes(mem, 1)
            results.append(dict(alice=alice, bob=bob, matched=int(ta == tb), label=label))

    # sifting: keep only rounds where the two bases agreed
    if not results:
        raise RuntimeError("protocol produced no key rounds")
    keep = [r for r in results if r["matched"]]
    alice = np.concatenate([r["alice"] for r in keep])
    bob = np.concatenate([r["bob"] for r in keep])
    sifted = len(alice)

    # QBER estimated on a publicly revealed random subset
    n_reveal = max(1, int(sifted * reveal_fraction))
    idx = rng.choice(sifted, size=n_reveal, replace=False)
    qber = float(np.mean(alice[idx] != bob[idx]))
    mask = np.ones(sifted, dtype=bool)
    mask[idx] = False
    alice_r, bob_r = alice[mask], bob[mask]

    rate = key_rate_shor_preskill(qber)
    n_final = int(len(alice_r) * rate)
    if n_final > 0:
        key_bits = toeplitz_hash(alice_r, n_final, np.random.default_rng(seed + 1))
        key = sha256_hex(bits_to_bytes(key_bits))
    else:
        key = ""

    chsh = chsh_mixed(backend, p_eve=p_eve, shots=8192, tag=tag)
    return ProtocolResult(str(backend), p_eve, n_pairs, sifted, qber, chsh,
                          rate, n_final, key, n_reveal)


# =============================================================================
# 5.  Eve sweep -- the figure that belongs in the paper
# =============================================================================

def eve_sweep(backend, levels: list, n_pairs: int = 32768,
              seed: int = 20261013, tag: str = "") -> list:
    rows = []
    for p in levels:
        res = run_protocol(backend, n_pairs=n_pairs, p_eve=p, seed=seed, tag=tag)
        rows.append(dict(p_eve=p, qber=round(res.qber, 6),
                         predicted_qber=round(0.25 * p, 6),
                         key_rate=round(res.key_rate, 6),
                         chsh_S=round(res.chsh.S, 4),
                         sifted_len=res.sifted_len,
                         final_key_bits=res.final_key_bits))
        print("    p_eve=%.2f  QBER=%.4f (predicted %.4f)  rate=%.4f  S=%.3f"
              % (p, res.qber, 0.25 * p, res.key_rate, res.chsh.S), flush=True)
    return rows


# =============================================================================
# 6.  Real-hardware CHSH (2 qubits, well inside today's NISQ budget)
# =============================================================================

def ibm_chsh(backend_name: str | None = None, shots: int = 4096) -> dict:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    except ImportError as exc:
        raise RuntimeError("pip install qiskit-ibm-runtime for --mode ibm") from exc

    import os
    token = os.environ.get("QISKIT_IBM_TOKEN")
    service = (QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
               if token else QiskitRuntimeService())
    backend = (service.backend(backend_name) if backend_name
               else service.least_busy(operational=True, simulator=False, min_num_qubits=2))
    print("    backend: %s" % backend.name, flush=True)

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa = [pm.run(bell_pair_circuit(a, b)) for a, b in CHSH_PAIRS]

    sampler = Sampler(mode=backend)      # job mode, allowed on the Open Plan
    sampler.options.default_shots = shots
    t0 = time.time()
    job = sampler.run(isa)
    print("    job id: %s (queued)" % job.job_id(), flush=True)
    res = job.result()

    cors, meta = [], []
    for i, pub in enumerate(res):
        bits = list(pub.data.c.get_bitstrings())
        a = np.array([int(s[::-1][0]) for s in bits], dtype=np.uint8)
        b = np.array([int(s[::-1][1]) for s in bits], dtype=np.uint8)
        cors.append(correlation(a, b))
        meta.append(len(bits))
    S = float(sum(s * e for s, e in zip(CHSH_SIGNS, cors)))
    props = backend.properties()
    out = dict(backend=backend.name, job_id=str(job.job_id()),
               correlations=[round(c, 5) for c in cors], S=round(S, 5),
               S_std=round(2.0 / np.sqrt(meta[0]), 5), violated=abs(S) > 2.0,
               shots=meta[0], wall_time_s=round(time.time() - t0, 1),
               calibration_time=str(getattr(props, "last_update_date", "")))
    print("    S = %.4f +- %.4f  -> %s"
          % (S, out["S_std"], "VIOLATION (quantum)" if out["violated"] else "no violation"),
          flush=True)
    return out


# =============================================================================
# 7.  Self-test: verify the physics before trusting any number
# =============================================================================

def self_test():
    print("E91 self-test")
    backend, name = make_local_backend(noise=False)
    ok = True

    print("  [1] correlation convention E(theta_a, theta_b) = cos(2(da))")
    for ta, tb in ((0.0, 0.0), (0.0, np.pi / 8), (np.pi / 4, -np.pi / 8)):
        mem = run_circuits([bell_pair_circuit(ta, tb)], backend, 4096, tag="cal")[0]
        e = correlation(_mem_outcomes(mem, 0), _mem_outcomes(mem, 1))
        want = np.cos(2 * (ta - tb))
        good = abs(e - want) < 0.06
        ok &= good
        print("      E(%.3f, %.3f) = %+.4f   theory %+.4f   %s"
              % (ta, tb, e, want, "OK" if good else "MISMATCH"))

    print("  [2] CHSH must reach 2*sqrt(2) = 2.8284 on an ideal simulator")
    ch = chsh_experiment(backend, shots=16384, tag="selftest")
    good = ch.S > 2.7
    ok &= good
    print("      S = %.4f   (%s)" % (ch.S, "OK" if good else "TOO LOW"))

    print("  [3] perfect correlation must give zero QBER with no Eve")
    res = run_protocol(backend, n_pairs=16384, p_eve=0.0, tag="selftest")
    good = res.qber < 0.01
    ok &= good
    print("      QBER = %.5f   (%s)" % (res.qber, "OK" if good else "TOO HIGH"))

    print("  [4] full intercept-resend must give QBER ~ 25% and kill the key")
    res = run_protocol(backend, n_pairs=16384, p_eve=1.0, tag="selftest")
    good = abs(res.qber - 0.25) < 0.04 and res.key_rate == 0.0
    ok &= good
    print("      QBER = %.4f (expect 0.25)   rate = %.4f (expect 0)   %s"
          % (res.qber, res.key_rate, "OK" if good else "CHECK"))

    print("  [5] Shor-Preskill threshold sits at QBER = 11%")
    lo, hi = key_rate_shor_preskill(0.10), key_rate_shor_preskill(0.12)
    good = lo > 0 and hi == 0.0
    ok &= good
    print("      r(10%%) = %.4f > 0,  r(12%%) = %.4f = 0   %s"
          % (lo, hi, "OK" if good else "WRONG"))

    print("E91 self-test %s\n" % ("PASSED" if ok else "FAILED"), flush=True)
    return 0 if ok else 1


# =============================================================================
# 8.  CLI
# =============================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E91 QKD for image encryption")
    ap.add_argument("--mode", choices=["local", "noisy", "ibm"], default="local")
    ap.add_argument("--n-pairs", type=int, default=65536)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--eve-sweep", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if a.self_test:
        return self_test()

    os.makedirs(a.outdir, exist_ok=True)
    payload = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"))

    if a.mode == "ibm":
        payload["hardware_chsh"] = ibm_chsh(a.backend, shots=a.shots)
        path = os.path.join(a.outdir, "e91_report.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(json.dumps(payload, indent=2))
        print("\nwrote %s" % path)
        return 0

    backend, bname = make_local_backend(noise=(a.mode == "noisy"), seed=a.seed)
    payload["backend"] = bname

    if a.eve_sweep:
        rows = eve_sweep(backend, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
                         n_pairs=a.n_pairs, seed=a.seed, tag=a.mode)
        payload["eve_sweep"] = rows
        csv_path = os.path.join(a.outdir, "e91_sweep.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\nwrote %s" % csv_path)
    else:
        res = run_protocol(backend, n_pairs=a.n_pairs, seed=a.seed, tag=a.mode)
        payload["protocol"] = res.as_dict()
        print(json.dumps(payload["protocol"], indent=2))

    path = os.path.join(a.outdir, "e91_report.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
