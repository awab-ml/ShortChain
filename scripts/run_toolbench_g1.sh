#!/usr/bin/env bash
set -e

PYTHON=".venv/bin/python3"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

echo "=========================================================="
echo " Step 1: Preparing ToolBench G1 Datasets"
echo "=========================================================="
$PYTHON scripts/prepare_toolbench_g1.py "$@"

echo ""
echo "=========================================================="
echo " Step 2: Running ToolBench G1 Benchmark Pipeline"
echo "=========================================================="
$PYTHON scripts/run_benchmark.py \
    --benchmark toolbench \
    --train-path data/toolbench/g1_train.jsonl \
    --eval-path data/toolbench/g1_test.jsonl \
    --catalog-path data/toolbench/g1_catalog.json \
    --config configs/toolbench_g1.yaml \
    --failure-negatives \
    --output models/toolbench_g1.pkl \
    --results-dir results/toolbench_g1 \
    "$@"

echo ""
echo "=========================================================="
echo " Benchmark Run Finished Successfully!"
echo " Results saved in results/toolbench_g1/toolbench_results.json"
echo "=========================================================="
