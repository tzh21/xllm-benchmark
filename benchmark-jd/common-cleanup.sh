#!/bin/bash

# Common cleanup function for benchmark scripts
# Kills only child processes (background jobs) of the current script

cleanup() {
    echo "Cleaning up..."
    trap - INT TERM
    echo "Subprocesses: $(jobs -p)"
    jobs -p | xargs -r kill
    wait
}

# Set up trap to call cleanup on exit or termination
trap cleanup INT TERM