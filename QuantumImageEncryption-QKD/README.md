# QuantumImageEncryption-QKD

Reference implementation and experimental data for the paper:

> **Adaptive Image Encryption Algorithm Based on Entanglement-Driven Quantum Key
> Distribution and High-Base Reversible Cellular Automata**
> Mingqi Dai, Xiaofen Huang, Tinggui Zhang, Xishun Zhu
> Hainan Normal University, Haikou, China

This repository contains the **quantum subsystem** of that paper: the physical
entropy sources (two independent quantum devices), the E91 entanglement-based
key-distribution protocol with its CHSH certification, the Toeplitz extraction
and NIST SP 800-22 validation chain, and the frozen raw datasets reported in
Tables 1 and 2 of the manuscript.

> **Scope note.** The classical image-transform layer of the scheme (HBRCA /
> dynamic S-box / adaptive error-correcting code) is not a quantum component and
> is documented in the manuscript only; this repository ships the parts whose
> claims are verifiable against physical quantum hardware.

## Repository layout

```
.
├── quantum/                      # source code
│   ├── quantum_entropy_pool.py   # entropy pool: drop-in replacement for the
│   │                             #   simulator call, HKDF-SHA3-256 derivation
│   ├── e91_qkd.py                # E91 protocol: Bell test, sifting, privacy
│   │                             #   amplification, Shor-Preskill key rate
│   ├── qrng_harvest.py           # physical harvest (IBM QPU / ANU API),
│   │                             #   Toeplitz extraction, NIST SP 800-22 battery
│   ├── harvest_dataset.py        # freeze-a-dataset + segment-wise validation
│   ├── ingest_counts.py          # ingest externally exported counts CSV
│   ├── ingest_spinq.py           # ingest SpinQ NMR device counts CSV
│   ├── harvest_spinq.py          # harvest from a SpinQ Triangulum Pro (NMR)
│   └── check_credentials.py      # credential self-check for the two APIs
├── data/                         # frozen raw datasets (the submitted artefacts)
│   ├── ibm_raw.bin               #   4 194 304 raw bits, IBM Quantum ibm_fez
│   ├── ibm_extracted.bin         #   1 841 079 Toeplitz-extracted bits
│   ├── manifest_ibm.json         #   provenance: jobs, calibration, T1
│   ├── spinq_counts.csv          #   SpinQ Triangulum Pro measurement histogram
│   ├── spinq_raw.bin             #   67 584 raw bits (22 528 shots x 3 qubits)
│   ├── spinq_extracted.bin       #   33 678 Toeplitz-extracted bits
│   └── manifest_spinq.json       #   provenance: device, shots, seed
└── results/                      # published validation outputs
    ├── verification.json         #   NIST SP 800-22, 4 segments x 9 tests, all pass
    ├── e91_hardware_report.json  #   CHSH on ibm_fez, S = 2.486 +- 0.022
    ├── e91_simulated_report.json #   CHSH sweep under the device noise model
    └── e91_sweep_simulated.csv   #   per-eavesdropping-fraction table
```

## Installation

Python 3.10+ is required for the Qiskit stack. (`harvest_spinq.py` needs
Python 3.8-3.10 and the vendor SDK `spinqit`, see below.)

```bash
pip install -r requirements.txt
```

No credential is needed to reproduce anything from the frozen datasets:

```bash
# verify the frozen IBM dataset (4 segments x 9 NIST tests)
python quantum/harvest_dataset.py --verify data
```

## Credentials (only for re-harvesting from hardware)

Credentials are read from environment variables, never hard-coded. Copy
`.env.example` to `.env` (or export the variables) and fill in your own:

| Variable | Purpose | Where to get it |
|---|---|---|
| `QISKIT_IBM_TOKEN` | IBM Quantum Platform | https://quantum.ibm.com (free Open Plan) |
| `ANU_AQN_API_KEY` | ANU Quantum Numbers (optional second source) | https://quantumnumbers.anu.edu.au (free signup) |

SpinQ Triangulum Pro devices are driven over the local network and take their
address and account on the command line (`--ip --port --account --password`),
or via `SPINQ_IP`, `SPINQ_PORT`, `SPINQ_ACCOUNT`, `SPINQ_PASSWORD`; see
`quantum/harvest_spinq.py --help`.

```bash
python quantum/check_credentials.py     # reports what is set and whether it authenticates
```

## Reproducing the paper's numbers

All commands are run from the repository root.

### 1. Entropy validation (Table 1)

```bash
python quantum/harvest_dataset.py --verify data
```

Runs the nine applicable NIST SP 800-22 tests on four independent segments of
the extracted streams and writes `data/verification.json`. Expected headline
numbers: IBM raw bias 4.34e-02, min-entropy 0.8799 bit/bit, Toeplitz rate
0.4389; SpinQ raw bias 4.73e-04, min-entropy 0.9986 bit/bit, Toeplitz rate
0.4983; all 36 segment-level tests pass at alpha = 0.01.

### 2. CHSH certification (Table 2)

```bash
python quantum/e91_qkd.py --help
```

*Hardware anchor* (`results/e91_hardware_report.json`): the four-setting CHSH
statistic executed on `ibm_fez` (job `db6qajg4qg6s73c571p0`, 8192 shots),
S = 2.486 +- 0.022 — above the classical bound 2, below the noise-model
prediction 2.646, as discussed in the manuscript.

*Noise-model sweep* (`results/e91_sweep_simulated.csv`): intercept-resend
fraction from 0 to 1, reproducing Table 2 of the manuscript. The simulated
model uses an asymmetric readout error P(0|1) = 8e-3, P(1|0) = 2.5e-2
characteristic of the Heron-class device, with T1 = 250 us, T2 = 180 us.

### 3. Re-harvesting from hardware (optional, needs credentials)

```bash
# pilot, then full harvest, from an IBM backend
python quantum/qrng_harvest.py --source ibm --backend ibm_fez --shots 4096 ...
python quantum/harvest_dataset.py --harvest --source ibm --backend ibm_fez

# SpinQ Triangulum Pro over the LAN (vendor SDK, Python 3.8-3.10 venv)
python quantum/harvest_spinq.py --ip 192.168.137.1 --port 55444 \
    --account <user> --password <pw> --n-qubits 3 --target-bits 65536 \
    --outdir data
python quantum/ingest_spinq.py data/spinq_counts.csv --name spinq --outdir data
```

## Notes on data provenance

* `data/ibm_raw.bin` is the raw Z-basis measurement stream of a 16-qubit grid
  cluster state on `ibm_fez` (64 jobs x 4096 shots, 2026-10-13). The
  Toeplitz extraction seed is 20261013; see `manifest_ibm.json`.
* `data/spinq_counts.csv` is the native histogram export of the SpinQ device.
  Because the NMR backend returns aggregated counts rather than a per-shot
  time series, the stream is reconstructed by expanding the histogram and then
  applying a **published fixed-seed permutation** (seed 20261020) before
  extraction; this is recorded in `manifest_spinq.json` so the result is
  exactly reproducible. Temporal-order statistics are therefore marked n/a for
  the SpinQ column of Table 1.
* The ANU vacuum-fluctuation source was evaluated as a candidate third source
  but is not part of the released dataset: the provider's free-tier API rate
  limit (HTTP 429) prevented harvesting the full sample. The two released
  sources are physically independent modalities (superconducting transmon and
  room-temperature nuclear spin).

## License

MIT, see `LICENSE`.
