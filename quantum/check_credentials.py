#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Credential self-check for the two quantum entropy sources used by the paper.

Run this AFTER you have obtained a token/key:

    python check_credentials.py

It reports, for each source, whether a credential is visible and whether it
actually authenticates, then prints the exact next command to run. Nothing is
written anywhere and no job is submitted.
"""
from __future__ import annotations

import os
import sys


def _line(title: str) -> None:
    print("\n" + "=" * 70)
    print(" " + title)
    print("=" * 70)


def check_ibm() -> bool:
    _line("IBM Quantum  (QPU cluster-state entropy + CHSH)")

    token = os.environ.get("QISKIT_IBM_TOKEN")
    saved = {}
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        try:
            saved = QiskitRuntimeService.saved_accounts() or {}
        except Exception:
            saved = {}
    except ImportError:
        print("  qiskit-ibm-runtime not installed:")
        print("      pip install qiskit-ibm-runtime")
        return False

    if token:
        print("  [env]    QISKIT_IBM_TOKEN is set (%d chars)" % len(token))
        if len(token) != 44:
            print("           !! IBM API keys are 44 characters; yours is %d." % len(token))
            print("              You may have pasted a CRN or truncated the key.")
    else:
        print("  [env]    QISKIT_IBM_TOKEN not set")

    if saved:
        print("  [saved]  %d account(s) in ~/.qiskit/qiskit-ibm.json:" % len(saved))
        for name, acct in saved.items():
            ch = acct.get("channel", "?")
            dflt = "  <- default" if acct.get("is_default_account") else ""
            print("              %-28s channel=%s%s" % (name, ch, dflt))
    else:
        print("  [saved]  no saved account")

    if not token and not saved:
        print("\n  -> NO CREDENTIAL. Get one at https://quantum.cloud.ibm.com")
        print("     (dashboard -> create API key -> it is shown ONCE, 44 chars)")
        print("     then either:")
        print("       export QISKIT_IBM_TOKEN=<key>")
        print("     or, once per machine:")
        print('       python -c "from qiskit_ibm_runtime import QiskitRuntimeService '
              'as S; S.save_account(token=\'<key>\', overwrite=True)"')
        return False

    print("\n  -> verifying against the live service ...")
    try:
        service = (QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
                   if token else QiskitRuntimeService())
        backends = [b for b in service.backends(simulator=False, operational=True)
                    if b.num_qubits >= 16]
        print("  -> OK, authenticated.")
        if backends:
            print("     %d operational backend(s) with >= 16 qubits:" % len(backends))
            for b in backends[:6]:
                try:
                    pend = b.status().pending_jobs
                except Exception:
                    pend = -1
                print("       %-18s %3d qubits  pending=%s" % (b.name, b.num_qubits, pend))
            if len(backends) > 6:
                print("       ... and %d more" % (len(backends) - 6))
        else:
            print("     !! authenticated, but no >= 16-qubit backend is operational.")
            print("        The cluster-state circuit needs 16; try again later.")
        return True
    except Exception as exc:
        print("  -> FAILED: %s" % str(exc)[:300])
        print("     Common causes: key revoked, no Open Plan instance, or the")
        print("     platform is unreachable from this network.")
        return False


def check_aqn() -> bool:
    _line("ANU Quantum Numbers  (independent cross-check source)")

    key = os.environ.get("ANU_AQN_API_KEY")
    if not key:
        print("  [env]    ANU_AQN_API_KEY not set")
        print("\n  -> NO KEY. Get one free at https://quantumnumbers.anu.edu.au")
        print("     (sign in with GitHub / Google / LinkedIn, then copy the key)")
        print("     then:")
        print("       export ANU_AQN_API_KEY=<key>")
        return False

    print("  [env]    ANU_AQN_API_KEY is set (%d chars)" % len(key))
    print("\n  -> verifying with a 16-byte live request ...")
    try:
        import requests
        r = requests.get("https://api.quantumnumbers.anu.edu.au",
                         params={"length": 16, "type": "uint8"},
                         headers={"x-api-key": key}, timeout=30)
        if r.status_code == 200:
            data = r.json()
            print("  -> OK. Sample: %s" % data.get("data"))
            print("     (the deprecated qrng.anu.edu.au endpoint is NOT used)")
            return True
        print("  -> HTTP %d: %s" % (r.status_code, r.text[:200]))
        print("     A 403 means the key is wrong; a 429 means rate limited.")
        return False
    except Exception as exc:
        print("  -> FAILED: %s" % str(exc)[:300])
        return False


def main() -> int:
    print("Quantum entropy source credential check")
    print("No jobs are submitted and nothing is written to disk.")

    ibm_ok = check_ibm()
    aqn_ok = check_aqn()

    _line("Verdict")
    if ibm_ok and aqn_ok:
        print("  Both sources ready. Run, in this order:\n")
        print("    python harvest_dataset.py --harvest --source ibm "
              "--n-bits 524288 --outdir pilot")
        print("    python harvest_dataset.py --harvest --source ibm "
              "--n-bits 4194304 --outdir dataset")
        print("    python harvest_dataset.py --harvest --source aqn "
              "--n-bits 4194304 --outdir dataset")
        print("    python e91_qkd.py --mode ibm --outdir dataset_hw")
        print("    python harvest_dataset.py --verify dataset --segments 4")
    elif ibm_ok:
        print("  IBM only. That is enough for Table 2 (hardware CHSH) and for")
        print("  the primary entropy source; the AQN cross-check can be added later.")
        print("\n    python harvest_dataset.py --harvest --source ibm "
              "--n-bits 524288 --outdir pilot")
    else:
        print("  Nothing runnable yet. Obtain at least the IBM token:")
        print("    https://quantum.cloud.ibm.com")

    return 0 if (ibm_ok or aqn_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
