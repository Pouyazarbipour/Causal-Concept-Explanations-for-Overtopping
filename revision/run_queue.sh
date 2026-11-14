#!/bin/bash
# Sequential queue of the revision experiments (started after the robustness and geometry jobs).
cd "$(dirname "$0")/.."
while pgrep -f "src.residual_robustness|src.geometry_sensitivity" >/dev/null; do sleep 30; done
python3 -u -m experiments.run_policies_v2 --mode fixed > revision/logs/policies_fixed.log 2>&1
python3 -u -m src.conformal_uncertainty > revision/logs/conformal.log 2>&1
python3 -u -m experiments.run_policies_v2 --mode tuned > revision/logs/policies_tuned.log 2>&1
echo QUEUE_DONE >> revision/logs/queue.log
