"""Harvest genuine quantum-random bits from a SpinQ lab NMR device (Triangulum Pro).

Why this exists
---------------
The ANU vacuum-fluctuation QRNG could not be harvested (provider's free-tier API
hard-rate-limited us, HTTP 429). The lab has a physical SpinQ Triangulum Pro -- a
real 3-qubit NMR quantum computer, a *physically different* modality from the IBM
superconducting transmon used for Table 1. Two independent physical quantum
sources of different technology is a stronger cross-validation story than IBM +
ANU ever was, and it is local (LAN), so there is no cloud rate limit.

How to talk to the device (per the official SpinQit docs, doc.spinq.com)
----------------------------------------------------------------------
SpinQit connects to a desktop NMR unit through its "Local Quantum Computer"
backend, NOT by a device-name string. The correct calls are:

    from spinqit import get_nmr, get_compiler, Circuit, NMRConfig, H
    circ = Circuit(); q = circ.allocateQubits(3)
    for i in range(3): circ << (H, q[i])          # |+> on each qubit -> 50/50 Z
    exe  = get_compiler("native").compile(circ, 0)
    eng  = get_nmr()
    cfg  = NMRConfig()
    cfg.configure_ip("192.168.137.1"); cfg.configure_port(55444)
    cfg.configure_account("<your-account>", "<your-password>")
    cfg.configure_task("qrng", "QRNG plus-state")
    cfg.configure_shots(block)                      # backend manages relaxation
    counts = eng.execute(exe, cfg).counts           # {'000': n0, ..., '111': n7}

The Triangulum Pro exposes a local server; you must (1) enable network access and
note its IP/port from the device GUI, and (2) register a user account on the
machine. Any PC on the same LAN can then drive it. (SpinQ Cloud via
get_spinq_cloud + get_platform("triangulum_vp") is an alternative that needs an
SSH key at cloud.spinq.com, but the local NMR backend is the direct path.)

Output
------
Writes <outdir>/spinq_counts.csv in the exact two-column "<bitstring>,<count>"
format that ingest_counts.py consumes, then everything downstream (entropy
estimate, Toeplitz extraction, SP 800-22 battery, Table 1) is identical to the
IBM path. Run:

    python ingest_counts.py dataset/spinq_counts.csv --name spinq --outdir dataset

Two execution modes
-------------------
* --dry-run : no SDK, no hardware. Simulates the |+> measurement locally with
  numpy (a slightly biased Bernoulli per qubit to mimic imperfect state prep)
  so the whole CSV -> ingest -> verification chain can be validated anywhere.

* hardware  : connects via get_nmr()/NMRConfig using --ip/--port/--account/
  --password from the device. Accumulates counts in blocks of --block-size shots
  until target_bits are reached. The backend handles inter-shot relaxation, so
  --recycle-delay is only a small inter-block pause.
"""
import argparse
import csv
import json
import os
import time

import numpy as np


def build_plus_circuit(n_qubits: int):
    """Return a compiled |+>-on-each-qubit circuit (lazy import of spinqit).

    SpinQit measures all qubits implicitly (the documented first example returns
    counts without any explicit measure gate), so we only apply H to each qubit.
    Called only in hardware mode, where the SDK is importable.
    """
    from spinqit import Circuit, H, get_compiler
    circ = Circuit()
    q = circ.allocateQubits(n_qubits)
    for i in range(n_qubits):
        circ << (H, q[i])
    exe = get_compiler("native").compile(circ, 0)
    return exe


def simulate_plus_measure(n_qubits: int, n_shots: int, bias: float, rng):
    """Local stand-in for a noisy |+> measurement (dry-run only)."""
    p1 = 0.5 + bias
    draws = rng.random((n_shots, n_qubits)) < p1
    weights = (1 << np.arange(n_qubits))[::-1]
    ints = draws.astype(np.uint8) @ weights
    vals, counts = np.unique(ints, return_counts=True)
    return {format(v, "0%db" % n_qubits): int(c) for v, c in zip(vals, counts)}


def harvest_hardware(ip: str, port: int, account: str, password: str,
                     n_qubits: int, target_bits: int, block_size: int,
                     recycle_delay: float, task: str):
    """Accumulate counts from a real SpinQ NMR device via get_nmr()/NMRConfig."""
    from spinqit import get_nmr, NMRConfig
    exe = build_plus_circuit(n_qubits)
    eng = get_nmr()
    accumulated = {}
    total_shots = 0
    n_blocks = max(1, -(-target_bits // (n_qubits * block_size)))  # ceil
    for b in range(n_blocks):
        cfg = NMRConfig()
        cfg.configure_ip(ip)
        cfg.configure_port(port)
        cfg.configure_account(account, password)
        cfg.configure_task("%s_%d" % (task, b), "QRNG plus-state")
        cfg.configure_shots(block_size)
        result = eng.execute(exe, cfg)
        block = result.counts  # dict {bitstring: count}
        for k, v in block.items():
            accumulated[k] = accumulated.get(k, 0) + int(v)
        total_shots += block_size
        time.sleep(recycle_delay)
        print("  block %d/%d: +%d shots (total %d)" %
              (b + 1, n_blocks, sum(block.values()), total_shots))
    return accumulated, total_shots


def write_counts_csv(path: str, counts: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for k in sorted(counts):
            w.writerow([k, counts[k]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default="dataset")
    ap.add_argument("--csv", default=None,
                    help="output CSV path (default <outdir>/spinq_counts.csv)")
    ap.add_argument("--n-qubits", type=int, default=3,
                    help="qubits to use (Triangulum Pro = 3)")
    ap.add_argument("--target-bits", type=int, default=65536,
                    help="pilot target raw bits (right-size for NMR throughput)")
    ap.add_argument("--block-size", type=int, default=2048,
                    help="shots per execute() call (backend handles relaxation)")
    ap.add_argument("--recycle-delay", type=float, default=1.0,
                    help="seconds between blocks (light pacing)")
    ap.add_argument("--dry-run", action="store_true",
                    help="simulate locally, no SDK / no hardware")
    ap.add_argument("--dry-bias", type=float, default=0.01,
                    help="simulated per-qubit P(1)-0.5 bias for --dry-run")
    ap.add_argument("--ip", default=os.environ.get("SPINQ_IP"),
                    help="Triangulum Pro LAN IP (hardware, e.g. 192.168.137.1; env SPINQ_IP)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("SPINQ_PORT", "55444")),
                    help="Triangulum Pro server port (hardware; env SPINQ_PORT)")
    ap.add_argument("--account", default=os.environ.get("SPINQ_ACCOUNT"),
                    help="device account username (env SPINQ_ACCOUNT)")
    ap.add_argument("--password", default=os.environ.get("SPINQ_PASSWORD"),
                    help="device account password (env SPINQ_PASSWORD)")
    ap.add_argument("--task", default="qrng", help="task name prefix")
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    csv_path = a.csv or os.path.join(a.outdir, "spinq_counts.csv")

    if a.dry_run:
        rng = np.random.default_rng(a.seed)
        n_shots = max(1, a.target_bits // a.n_qubits)
        counts = simulate_plus_measure(a.n_qubits, n_shots, a.dry_bias, rng)
        mode = "dry-run (simulated |+> measurement)"
    else:
        if not (a.ip and a.account and a.password):
            print("hardware mode needs --ip / --account / --password "
                  "(from the Triangulum Pro device GUI)")
            return 1
        counts, n_shots = harvest_hardware(
            a.ip, a.port, a.account, a.password, a.n_qubits, a.target_bits,
            a.block_size, a.recycle_delay, a.task)
        mode = "hardware (%s:%d)" % (a.ip, a.port)

    write_counts_csv(csv_path, counts)
    total_bits = sum(int(v) for v in counts.values()) * a.n_qubits
    meta = {
        "mode": mode,
        "device": "SpinQ Triangulum Pro (NMR, local)",
        "n_qubits": a.n_qubits,
        "shots": int(n_shots),
        "raw_bits": int(total_bits),
        "csv": os.path.abspath(csv_path),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(a.outdir, "spinq_meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print("mode           :", mode)
    print("qubits used     :", a.n_qubits)
    print("shots           :", n_shots)
    print("raw bits        :", total_bits)
    print("distinct strings:", len(counts))
    print("wrote           :", csv_path)
    print("next            : python ingest_counts.py %s --name spinq "
          "--outdir %s" % (csv_path, a.outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
