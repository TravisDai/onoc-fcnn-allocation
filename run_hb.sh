#!/bin/sh
cd "$(dirname "$0")"
python hb_experiments.py main > results/log_hb_main.txt 2>&1
python hb_experiments.py lanes > results/log_hb_lanes.txt 2>&1
python hb_experiments.py rate > results/log_hb_rate.txt 2>&1
python hb_experiments.py clusters > results/log_hb_clusters.txt 2>&1
python hb_experiments.py dcfg > results/log_hb_dcfg.txt 2>&1
echo done > results/hb_done.txt
