#!/bin/bash
# Waits for run_all.sh to finish (GPU is single), then trains on v2 data and scores everything on the v2 test.
cd "$(dirname "$0")"
PY="${PY:-python}"; export PYTHONIOENCODING=utf-8
while ! grep -q "ALL_DONE\|Error\|Traceback" results/run_all.log 2>/dev/null; do sleep 60; done
$PY dl.py --name v2 --data v2 --weighted
$PY eval_v2.py base v2
echo V2_DONE
