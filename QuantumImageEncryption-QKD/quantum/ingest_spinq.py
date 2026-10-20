"""Ingest the lab SpinQ Triangulum Pro harvest into the standard pipeline.

Why the extra permutation step
------------------------------
The NMR backend returns only an *aggregate* measurement histogram -- per-shot
temporal order is not preserved. Expanding a histogram directly ("000" x 2848,
then "001" x 2784, ...) yields a pathologically ordered bit stream: the first
~8.5k bits are all zero, then the stream alternates in rigid 3-bit blocks. That
artefact destroys the runs / serial / block-frequency tests and produces false
failures. So we reconstruct a *representative* stream by expanding the histogram
and applying a fixed-seed permutation, then feed the identical chain used for the
IBM source: same Toeplitz protocol seed (20261013), same conservative safe_rate
policy, same nine-test SP 800-22 battery.

Consequences we report honestly in the paper:
  * bias and min-entropy are permutation-invariant, so they remain valid device
    diagnostics;
  * autocorrelation across shots CANNOT be assessed from counts-only data, so
    that Table 1 row is reported as n/a for this source.

Usage:
    python ingest_spinq.py "F:/戴明琪/spinq_counts.csv" --outdir dataset
"""
import argparse
import csv
import json
import os
import time

import numpy as np

import qrng_harvest as Q

PERM_SEED = 20261020      # fixed reconstruction seed (documented, reproducible)
TOEPLITZ_SEED = 20261013  # paper protocol seed, identical to the IBM path


def read_counts_csv(path: str) -> np.ndarray:
    bits = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            key, cnt = row[0].strip(), row[1].strip()
            if not key or key[0] not in "01":
                continue
            try:
                n = int(float(cnt))
            except ValueError:
                continue
            if n <= 0:
                continue
            bits.extend([1 if ch == "1" else 0 for ch in key] * n)
    return np.array(bits, dtype=np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv_path")
    ap.add_argument("--outdir", default="dataset")
    ap.add_argument("--name", default="spinq")
    ap.add_argument("--n-qubits", type=int, default=3)
    ap.add_argument("--seed", type=int, default=TOEPLITZ_SEED)
    ap.add_argument("--perm-seed", type=int, default=PERM_SEED)
    a = ap.parse_args()

    bits = read_counts_csv(a.csv_path)
    if len(bits) == 0:
        print("no counts parsed from %s" % a.csv_path)
        return 1
    print("expanded %d raw bits from %s" % (len(bits), a.csv_path))

    perm = np.random.default_rng(a.perm_seed).permutation(len(bits))
    bits = bits[perm]
    print("applied fixed-seed permutation (seed %d) for representative order"
          % a.perm_seed)

    meta = {
        "origin": "SpinQ Triangulum Pro (NMR, local device) aggregate counts",
        "csv": os.path.abspath(a.csv_path),
        "n_qubits": a.n_qubits,
        "perm_seed": a.perm_seed,
        "toeplitz_seed": a.seed,
        "note": ("NMR backend returns an aggregate histogram only; per-shot "
                 "temporal order is unavailable, so the stream is reconstructed "
                 "from the histogram and permuted with a fixed seed. Bias and "
                 "min-entropy are permutation-invariant; autocorrelation is not "
                 "a meaningful device diagnostic for this source."),
        "ingested_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    def harvest_fn():
        return {"bits": bits, "meta": meta}

    rep = Q.process_source(a.name, harvest_fn, a.outdir, len(bits), a.seed,
                           rate=None)

    man = {
        "source": a.name,
        "created_utc": meta["ingested_utc"],
        "raw_bits": len(bits),
        "extracted_bits": rep.extracted_bits,
        "extraction_rate": rep.rate,
        "raw_stats": rep.raw_entropy,
        "sha256_raw": rep.raw_sha256,
        "sha256_extracted": rep.extracted_sha256,
        "origin": meta,
    }
    os.makedirs(a.outdir, exist_ok=True)
    mpath = os.path.join(a.outdir, "manifest_%s.json" % a.name)
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2)

    print("\n===== Table 1 diagnostics for '%s' =====" % a.name)
    e = rep.raw_entropy
    print("raw bits            :", len(bits))
    print("bias of raw stream  :", "%.4e" % e["bias"])
    print("min-entropy per bit :", "%.4f" % e["min_entropy_bit"])
    print("Toeplitz rate       :", "%.4f" % rep.rate)
    print("extracted bits      :", rep.extracted_bits)
    print("max|autocorr| 1-32  : %.3e  (n/a for counts-only source)" %
          rep.autocorr_max)
    print("\nSP 800-22 p-values (full extracted stream, alpha=0.01):")
    for t in rep.tests_extracted:
        print("  %-32s %8.4f  %s" % (t.name, t.p_value,
                                     "PASS" if t.passed() else "FAIL"))
    print("\nwrote %s" % mpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
