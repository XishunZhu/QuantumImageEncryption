"""Ingest externally harvested counts into the pipeline.

For the case where the IBM Quantum *API* is unreachable (e.g. Cloudflare blocks
the sandbox egress) but the *browser* works: build the same 4x4 cluster circuit
in the Composer at https://quantum.cloud.ibm.com, run it, download the counts
CSV, and feed it here. Everything downstream (entropy estimate, Toeplitz
extraction, SP 800-22 battery, Table 1 export) is identical to a live harvest.

    python ingest_counts.py counts.csv --name composer --outdir dataset

CSV format: two columns, <bitstring>,<count> -- exactly what the Composer
"Download results" button or a SamplerV2 counts export produces. A header row
is fine; rows whose first field is not 0/1 are skipped. Any bit width works.
"""
import argparse
import csv
import json
import os
import time

import numpy as np

import qrng_harvest as Q


def read_counts_csv(path: str, max_bits: int | None = None) -> tuple:
    """Expand a counts CSV into a flat bit array (Qiskit key order preserved)."""
    bits = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            key, cnt = row[0].strip(), row[1].strip()
            if not key or key[0] not in "01":
                continue                       # header / footer / junk
            try:
                n = int(float(cnt))
            except ValueError:
                continue
            if n <= 0:
                continue
            bits.extend([1 if ch == "1" else 0 for ch in key] * n)
            if max_bits and len(bits) >= max_bits:
                bits = bits[:max_bits]
                break
    return np.array(bits, dtype=np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv_path")
    ap.add_argument("--name", default="composer",
                    help="dataset label, used for <name>_raw/_extracted.bin")
    ap.add_argument("--outdir", default="dataset")
    ap.add_argument("--seed", type=int, default=20261013,
                    help="Toeplitz seed (must match the paper protocol)")
    ap.add_argument("--max-bits", type=int, default=None,
                    help="cap the ingested bits (e.g. 4194304)")
    a = ap.parse_args()

    bits = read_counts_csv(a.csv_path, a.max_bits)
    if len(bits) == 0:
        print("no counts parsed from %s -- check the CSV format" % a.csv_path)
        return 1
    print("parsed %d bits from %s" % (len(bits), a.csv_path))

    meta = {
        "origin": "external counts ingest",
        "csv": os.path.abspath(a.csv_path),
        "ingested_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    def harvest_fn():
        return {"bits": bits, "meta": meta}

    rep = Q.process_source(a.name, harvest_fn, a.outdir, len(bits),
                           a.seed, rate=None)

    # minimal manifest so `harvest_dataset.py --verify` shows provenance
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
    print("\nwrote %s" % mpath)
    print("next: python harvest_dataset.py --verify %s --segments 4" % a.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
