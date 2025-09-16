#!/bin/bash

# Monitor all NPUs from 0 to 15
TIMESTAMP_SUFFIX=$(date +%m%d_%H%M%S)

# Create log directory if it doesn't exist
mkdir -p local/log/aicore

# Start monitoring for each NPU (0-15)
for DEVICE_ID in {0..7}; do
    OUTPUT_FILE="local/log/aicore/aicore-${TIMESTAMP_SUFFIX}-npu${DEVICE_ID}.csv"
    echo "Starting monitor for NPU $DEVICE_ID -> $OUTPUT_FILE"
    sudo stdbuf -oL npu-smi info watch -i $DEVICE_ID > $OUTPUT_FILE &
    # stdbuf -oL npu-smi info watch -i $DEVICE_ID > $OUTPUT_FILE &
done

echo "All NPU monitors started (NPUs 0-15)"
echo "Log files are being written to local/log/aicore/aicore-${TIMESTAMP_SUFFIX}-npu*.csv"
