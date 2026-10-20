#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch harvest -> frozen dataset -> Toeplitz derivation, for the QIP submission.

Why a separate script
---------------------
Harvesting once and deriving all experiment keys from the frozen stream is the
correct QRNG post-processing architecture, but it creates a statistical hazard
that has to be handled explicitly: the NIST battery is then run on a *single*
realisation, so the reported p-values carry no sampling error estimate. A
reviewer can (rightly) object that one lucky stream proves little.

The remedy used here: freeze the raw stream once, derive once, then slice the
*extracted* stream into K segments and run the full battery on every segment.
Segment-to-segment agreement is what makes the claim reproducible, and the
manifest makes it auditable.

Three modes
-----------
  --plan                 Budget the IBM Open Plan allowance (no credentials).
  --harvest              Collect a batch, derive, freeze, and write a manifest.
  --verify <dir>         Segment-wise validation of a frozen dataset + Table 1.

Usage
-----
  python harvest_dataset.py --plan
  python harvest_dataset.py --harvest --source ibm  --backend ibm_brisbane
  python harvest_dataset.py --harvest --source aqn
  python harvest_dataset.py --verify dataset
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np

import qrng_harvest as Q
from qrng_harvest import (
    _as_bits, bits_to_bytes, bytes_to_bits, estimate_entropy, run_battery,
    safe_rate, sha256_hex, toeplitz_hash,
)


# ---------------------------------------------------------------- constants --
IBM_MONTHLY_QPU_S = 600.0          # Open Plan: 10 min per 28-day rolling window
BITS_PER_SHOT = 16                 # 4x4 grid cluster state, Z basis
ALPHA = 0.01                       # NIST SP 800-22 significance level
# SP 800-22 recommends n >= 10^6 for a single-sequence run; the approximate
# entropy test with m = 10 is the binding constraint.
TARGET_EXTRACTED = 1 << 20


# ------------------------------------------------------------------- plan ---
def plan(n_bits: int, shots: int, sec_per_circuit: float) -> dict:
    pubs = int(np.ceil(n_bits / (BITS_PER_SHOT * shots)))
    total = pubs * sec_per_circuit
    return dict(
        raw_bits=n_bits,
        shots_per_circuit=shots,
        circuits=pubs,
        circuit_executions=pubs * shots,
        est_qpu_s=round(total, 1),
        pct_of_monthly=round(100.0 * total / IBM_MONTHLY_QPU_S, 1),
        within_open_plan=total <= IBM_MONTHLY_QPU_S,
    )


def cmd_plan(a) -> int:
    print("=" * 74)
    print(" IBM Quantum Open Plan budget  (10 min QPU / 28-day rolling window)")
    print("=" * 74)
    print(" circuit: 4x4 grid cluster state, 16 qubits, 24 CZ edges, Z-basis")
    print(" yield  : %d raw bits per shot\n" % BITS_PER_SHOT)

    # A 4x2^20 raw harvest under three per-circuit cost assumptions. The
    # reference figure (1.4 s per 4000-shot circuit) is for a much deeper
    # 21-qubit / 110-1320 CZ workload, so it is a pessimistic upper bound.
    print(" Raw harvest = 2^22 bits = 4,194,304 bits")
    print(" %-10s %-10s %-14s %-12s %s" %
          ("s/circuit", "circuits", "executions", "QPU (s)", "% of 10 min"))
    for s in (0.3, 0.7, 1.4):
        p = plan(1 << 22, 4096, s)
        print(" %-10.1f %-10d %-14d %-12.1f %.1f%%%s"
              % (s, p["circuits"], p["circuit_executions"], p["est_qpu_s"],
                 p["pct_of_monthly"], "" if p["within_open_plan"] else "  <-- OVER"))
    print()

    print(" E91 CHSH on hardware (2 qubits, 4 circuits, 4096 shots): < 10 s")
    print(" Total worst case ~ %.0f s of a 600 s allowance.\n"
          % (plan(1 << 22, 4096, 1.4)["est_qpu_s"] + 10))

    print("=" * 74)
    print(" Recommended split of the raw harvest")
    print("=" * 74)
    for rate in (0.50, 0.45, 0.35, 0.25):
        ext = int((1 << 22) * rate)
        print("  extraction rate %.2f -> %9d extracted bits  (target %d, %s)"
              % (rate, ext, TARGET_EXTRACTED,
                 "OK" if ext >= TARGET_EXTRACTED else "SHORT"))
    print()
    print(" 2^22 raw keeps a >= 2^20 extracted stream even if the hardware")
    print(" bias forces the rate down to 0.25, which is the safety margin.")
    print()
    print(" NOTE: queue time (typically 5-30 min) does not consume the")
    print(" allowance. Pack every circuit into ONE Sampler job so the queue")
    print(" is entered once; Open Plan allows <= 5 queued jobs at a time.")
    return 0


# ---------------------------------------------------------------- harvest ---
def harvest(a) -> int:
    outdir = a.outdir
    os.makedirs(outdir, exist_ok=True)
    n_bits = a.n_bits

    builders = {
        "ibm": lambda: Q.harvest_ibm_qpu(n_bits, a.backend, shots=a.shots),
        "aqn": lambda: Q.harvest_aqn(n_bits, legacy=a.legacy, throttle=0.5,
                                   max_retries=25,
                                   checkpoint=os.path.join(outdir, "anu_checkpoint.bin")),
        "local": lambda: Q.harvest_local(n_bits, shots=a.shots, noise=False,
                                        seed=a.seed),
        "noisy": lambda: Q.harvest_local(n_bits, shots=a.shots, noise=True,
                                        seed=a.seed),
    }
    print("=== batch harvest: %s, %d raw bits ===" % (a.source, n_bits))
    t0 = time.time()
    got = builders[a.source]()
    bits = _as_bits(got["bits"])
    meta = got["meta"]
    print("    harvested %d raw bits in %.1fs" % (len(bits), time.time() - t0))

    est = estimate_entropy(bits)
    print("    bias=%.6f  H=%.6f  H_min=%.6f"
          % (est.bias, est.shannon_bit, est.min_entropy_bit))

    # Clamp the rate to the audited floor: never assume better hardware than
    # measured, because the extraction ratio is a security parameter.
    rate = a.rate if a.rate else min(safe_rate(est), a.rate_cap)
    m = int(len(bits) * rate)
    print("    Toeplitz: L=%d -> m=%d  (rate %.4f)" % (len(bits), m, rate))

    t1 = time.time()
    # The Toeplitz seed is PUBLIC by design -- universal hashing needs a seed
    # chosen independently of the raw stream, not a secret one. It is frozen
    # in the manifest so any reader can re-derive the identical key stream.
    rng = np.random.default_rng(a.seed)
    extracted = toeplitz_hash(bits, m, rng, progress=True)
    print("    extraction took %.1fs" % (time.time() - t1))

    raw_bytes, ext_bytes = bits_to_bytes(bits), bits_to_bytes(extracted)
    raw_p = os.path.join(outdir, "%s_raw.bin" % a.source)
    ext_p = os.path.join(outdir, "%s_extracted.bin" % a.source)
    open(raw_p, "wb").write(raw_bytes)
    open(ext_p, "wb").write(ext_bytes)

    manifest = dict(
        schema="qip-qrng-dataset/v1",
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source=a.source,
        raw_bits=len(bits),
        extracted_bits=m,
        extraction_rate=round(rate, 6),
        toeplitz=dict(
            construction="Toeplitz universal hash, m x (L+m-1) seed",
            seed=int(a.seed),
            seed_is_public=True,
            security_basis="leftover hash lemma; seed independent of raw stream",
        ),
        raw_stats=est.as_dict(),
        files={
            os.path.basename(raw_p): dict(
                bytes=len(raw_bytes), sha256=sha256_hex(raw_bytes)),
            os.path.basename(ext_p): dict(
                bytes=len(ext_bytes), sha256=sha256_hex(ext_bytes)),
        },
        provenance=meta,
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            qiskit=Q.__dict__.get("__qiskit_version__", "n/a"),
        ),
    )
    mpath = os.path.join(outdir, "manifest_%s.json" % a.source)
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print("\n    frozen dataset:")
    print("      %-28s %8d bytes  sha256=%s"
          % (os.path.basename(raw_p), len(raw_bytes),
             sha256_hex(raw_bytes)[:16] + "..."))
    print("      %-28s %8d bytes  sha256=%s"
          % (os.path.basename(ext_p), len(ext_bytes),
             sha256_hex(ext_bytes)[:16] + "..."))
    print("      manifest -> %s" % mpath)
    print("\n    Publish the raw .bin files with the submission: the whole")
    print("    point of freezing the stream is that a reader can re-derive it.")
    return 0


# ----------------------------------------------------------------- verify ---
def segment_battery(bits: np.ndarray, n_seg: int) -> list:
    """Run the full battery on `n_seg` disjoint slices of one stream.

    SP 800-22 asks for ~100 samples to judge pass-rate and p-value uniformity.
    A frozen dataset gives far fewer, so the segment sweep is reported as a
    *consistency indicator*, not as a formal pass/fail verdict.
    """
    n = (len(bits) // n_seg)
    out = []
    for i in range(n_seg):
        seg = bits[i * n:(i + 1) * n]
        res = run_battery(seg, label="seg%d " % i)
        n_fail = sum(0 if t.passed(ALPHA) else 1 for t in res)
        out.append(dict(segment=i, n_bits=len(seg), n_tests=len(res),
                        n_failed=n_fail,
                        p_min=round(min(t.p_value for t in res), 6),
                        results=[t.as_dict() for t in res]))
    return out


def cmd_verify(a) -> int:
    outdir = a.dir
    print("=" * 74)
    print(" Frozen dataset verification: %s" % outdir)
    print("=" * 74)

    man = None
    for fn in sorted(os.listdir(outdir)):
        if fn.startswith("manifest_") and fn.endswith(".json"):
            man = json.load(open(os.path.join(outdir, fn), encoding="utf-8"))
            print("\n--- manifest %s ---" % fn)
            print("    source          : %s" % man["source"])
            print("    created (UTC)   : %s" % man["created_utc"])
            print("    raw bits        : %d" % man["raw_bits"])
            print("    extracted bits  : %d  (rate %.4f)"
                  % (man["extracted_bits"], man["extraction_rate"]))
            print("    raw bias        : %.6f" % man["raw_stats"]["bias"])
            print("    H_min           : %.6f bit" % man["raw_stats"]["min_entropy_bit"])

    names = sorted({fn.rsplit("_", 1)[0] for fn in os.listdir(outdir)
                    if fn.endswith("_extracted.bin")})
    print("\n--- segment-wise SP 800-22 sweep (alpha = %.2f) ---" % ALPHA)
    print("    SP 800-22 asks for ~100 samples to judge pass rate and p-value")
    print("    uniformity. A frozen dataset yields %d, so this is a"
          % a.segments)
    print("    consistency indicator, not a formal verdict.\n")

    table_rows = {}
    for name in names:
        p = os.path.join(outdir, name + "_extracted.bin")
        bits = bytes_to_bits(open(p, "rb").read())
        segs = segment_battery(bits, a.segments)
        ok = sum(1 for s in segs if s["n_failed"] == 0)
        print("  %-10s %d segments x %d bits -> %d/%d fully pass, "
              "min p = %.4f"
              % (name, len(segs), segs[0]["n_bits"], ok, len(segs),
                 min(s["p_min"] for s in segs)))
        table_rows[name] = segs

    # ---- emit Table 1 rows straight from segment 0 of each source ----------
    print("\n" + "=" * 74)
    print(" Table 1 rows (segment 0; reuse the same row order as the draft)")
    print("=" * 74)
    for name in names:
        res = table_rows[name][0]["results"]
        print("\n  %s:" % name)
        for r in res:
            print("    %-38s %s" % (r["name"], r["p_value"]))

    summary = dict(outdir=outdir, alpha=ALPHA,
                   segments=a.segments, sources=table_rows)
    out = os.path.join(outdir, "verification.json")
    json.dump(summary, open(out, "w", encoding="utf-8"), indent=2)
    print("\n    full record -> %s" % out)
    return 0


# ------------------------------------------------------------------- main ---
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="budget the QPU allowance")
    ap.add_argument("--harvest", action="store_true", help="collect + freeze")
    ap.add_argument("--verify", metavar="DIR", help="validate a frozen dataset")
    ap.add_argument("--source", choices=["ibm", "aqn", "local", "noisy"],
                    default="ibm",
                    help="local = ideal simulator (dry run); noisy = device-like "
                         "noise model, reproduces the real bias/extraction-rate "
                         "behaviour without credentials")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--n-bits", type=int, default=1 << 22)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--rate", type=float, default=None)
    ap.add_argument("--rate-cap", type=float, default=0.5,
                    help="never extract faster than this, regardless of H_min")
    ap.add_argument("--legacy", action="store_true", help="ANU deprecated endpoint")
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--outdir", default="dataset")
    a = ap.parse_args(argv)

    if a.verify:
        a.dir = a.verify
        return cmd_verify(a)
    if a.harvest:
        return harvest(a)
    return cmd_plan(a)


if __name__ == "__main__":
    sys.exit(main())
