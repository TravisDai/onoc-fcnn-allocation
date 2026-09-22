#!/bin/sh
# Full pipeline (trace-driven main results + analytical-model sensitivity), then figures.
cd "$(dirname "$0")"
python experiments.py main > results/log_main.txt 2>&1
python experiments.py enoc > results/log_enoc.txt 2>&1
python experiments.py sens > results/log_sens.txt 2>&1
python ring_size.py > results/log_ring.txt 2>&1
python experiments.py dp > results/log_dp.txt 2>&1
