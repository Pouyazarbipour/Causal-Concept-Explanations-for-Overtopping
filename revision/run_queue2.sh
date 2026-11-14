#!/bin/bash
cd "$(dirname "$0")/.."
while pgrep -f "run_queue.sh" >/dev/null; do sleep 30; done
python3 -u -m experiments.run_policies_v2 --mode fixed > revision/logs/policies_fixed.log 2>&1
echo QUEUE2_DONE >> revision/logs/queue.log
