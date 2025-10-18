#!/bin/bash
# Quick check: existing shards and models

echo "=== Existing Shards ==="
ls -1 datasets/*shard*.pkl 2>/dev/null | wc -l | xargs echo "Total shards:"
echo ""
echo "CRL shards:"
ls -1 datasets/crl_pretraining_data_shard*.pkl 2>/dev/null | head -3
echo ""
echo "Challenge 2 shards:"
ls -1 datasets/challenge2_data_shard*.pkl 2>/dev/null | head -3

echo ""
echo "=== Existing Models ==="
ls -lh src/ML_pipeline_test/saved_models/*.pth 2>/dev/null | awk '{print $9, $5}'