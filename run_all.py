"""Master execution script.  Each stage is independently runnable (see README).

    python run_all.py                 # everything (several CPU-hours on 4 cores)
    python run_all.py --stages audit analysis figures
"""
import argparse
import subprocess
import sys

STAGES = {
    "audit": [sys.executable, "-m", "src.data_audit"],
    "tests": [sys.executable, "-m", "pytest", "-q", "tests"],
    "baselines": [sys.executable, "-m", "experiments.run_baselines"],
    "ccw": [sys.executable, "-m", "experiments.run_ccw"],
    "analysis": [sys.executable, "-m", "experiments.run_analysis"],
    "figures": [sys.executable, "-m", "src.plotting"],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=list(STAGES))
    for s in ap.parse_args().stages:
        print(f"=== stage: {s} ===", flush=True)
        subprocess.run(STAGES[s], check=True)
