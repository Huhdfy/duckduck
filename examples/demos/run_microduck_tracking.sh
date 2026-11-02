#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Run the agreed 3+2 experiment batches. Stop before batch two on a failed run.
set -u
repo_root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$repo_root" || exit 1
: "${UV_PROJECT_ENVIRONMENT:?Set UV_PROJECT_ENVIRONMENT to the validated Isaac Lab environment}"
output_root="${1:?Pass the absolute run output directory}"
queue_dir="${2:?Pass the absolute queue log directory}"
mkdir -p "$queue_dir"
printf 'variant\tstatus\n' > "$queue_dir/status.tsv"
git rev-parse HEAD > "$queue_dir/revision.txt"
run_batch() {
    local variant pid failed=0
    local -a pids=() names=()
    for variant in "$@"; do
        printf '%s\tstarted\n' "$variant" >> "$queue_dir/status.tsv"
        uv run --no-sync python examples/demos/microduck_tracking_experiment.py \
            --variant "$variant" --num-envs 4096 --iterations 6000 --seed 43 --output-root "$output_root" \
            > "$queue_dir/$variant.log" 2>&1 &
        pid=$!
        printf '%s\n' "$pid" > "$queue_dir/$variant.pid"
        pids+=("$pid")
        names+=("$variant")
    done
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            printf '%s\tcompleted\n' "${names[$i]}" >> "$queue_dir/status.tsv"
        else
            printf '%s\tfailed\n' "${names[$i]}" >> "$queue_dir/status.tsv"
            failed=1
        fi
    done
    return "$failed"
}
run_batch yaw_obs yaw_reward pos_half || exit 1
run_batch pos_only pos_add
