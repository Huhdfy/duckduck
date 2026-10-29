# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Research scoring preserves terminal windows, trial independence and conservative gates."""

import json

import numpy as np
import pytest

from isaaclab_tasks.contrib.microduck.reward_research import (
    FORWARD,
    TURNS,
    analyze_evaluation,
    bootstrap_difference,
    diagnose_h2,
    h1_gate,
    screen_candidate,
)


def test_trace_rescoring_keeps_warmup_and_excludes_autoresets(tmp_path):
    """Training and reference rewards differ only by declared kernels; the failed trial has no reset tail."""
    columns = [
        "link_vx_b_mps",
        "link_vy_b_mps",
        "link_vz_b_mps",
        "command_x",
        "command_y",
        "command_z",
        "yaw_error_radps",
        "roll_pitch_sq_rad2ps2",
        "reward_step/track_lin_vel",
        "reward_step/track_ang_vel",
        "reward_step/action_rate_l2",
        "reward_total_step",
        "alive_before_step",
    ]
    samples = np.zeros((4, 2, len(columns)), dtype=float)
    samples[:, :, 0] = samples[:, :, 3] = 0.1
    samples[:, :, 6] = 1
    samples[:, :, 8] = 2
    samples[:, :, 9] = 2 * np.exp(-2)
    samples[:, :, 10] = -0.1
    samples[:, :, 11] = 1.9 + 2 * np.exp(-2)
    samples[:, :, 12] = 1
    samples[2:, 1, :-1] = np.nan
    samples[2:, 1, -1] = 0
    np.savez_compressed(tmp_path / "step_traces.npz", **{"forward_0.1": samples})
    (tmp_path / "trace_metadata.json").write_text(json.dumps({"columns": columns, "step_dt": 1.0, "warmup_s": 1.0}))
    (tmp_path / "trials.json").write_text(
        json.dumps(
            [
                {"case": "forward_0.1", "trial": i, "target_vx_mps": 0.1, "vx_mps": 0.1, "completed": i == 0}
                for i in range(2)
            ]
        )
    )
    result = analyze_evaluation(tmp_path, {"variant": "w_std"})
    full, failed = result["trials"]
    assert full["reference_return"] == pytest.approx(4 * (1.9 + 2 * np.exp(-2)))
    assert failed["reference_return"] == pytest.approx(full["reference_return"] / 2)
    assert full["training_return"] == pytest.approx(4 * (1.9 + 2 * np.exp(-1 / 0.1225)))
    assert result["distributions"]["forward_0.1"]["x"]["n"] == 4
    assert not result["h2"]["eligible"]
    # A terminal divergence is retained with the live kernel's finite zero reward.
    samples[1, 1, 6] = np.nan
    samples[1, 1, 9] = 0
    samples[1, 1, 11] = 1.9
    np.savez_compressed(tmp_path / "step_traces.npz", **{"forward_0.1": samples})
    failed = analyze_evaluation(tmp_path, {"variant": "w_std"})["trials"][1]
    assert failed["training_rewards"]["track_ang_vel"] == pytest.approx(2 * np.exp(-1 / 0.1225))
    assert failed["training_return"] == pytest.approx(3.8 + 2 * np.exp(-1 / 0.1225))


def test_trial_bootstrap_and_saturation_gate(tmp_path):
    """Paired errors are resampled as trials, and degradation alone cannot open the H1 branch."""
    difference = bootstrap_difference(np.arange(16) + 2, np.arange(16))
    assert difference["ci95"] == [2.0, 2.0]
    baseline = {"trials": [{"case": c, "trial": i, "angular_rms_radps": 1.0} for c in TURNS for i in range(16)]}
    changed = {
        "folder": str(tmp_path),
        "trials": [
            {"case": c, "trial": i, "angular_rms_radps": 2.0, "saturation_fraction": 0.8}
            for c in TURNS
            for i in range(16)
        ],
    }
    arrays = {
        f"{c}/{key}": np.full(16, value)
        for c in TURNS
        for key, value in (("x", 4.0), ("exp_reward_gain_for_10pct_error_reduction", 0.009))
    }
    np.savez_compressed(tmp_path / "reward_distributions.npz", **arrays)
    assert h1_gate(baseline, changed)["eligible"]
    for row in changed["trials"]:
        row["saturation_fraction"] = 0.1
    assert not h1_gate(baseline, changed)["eligible"]


def test_h2_needs_group_overlap_and_unique_competitor():
    """A return advantage supported by a competing reward opens H2; tied attribution closes it."""
    trials = []
    for group in ("low", "tracking"):
        for _ in range(4):
            rewards = {
                "track_lin_vel": 1.0 if group == "low" else 2.0,
                "track_ang_vel": 2.0,
                "action_rate_l2": 0.0 if group == "low" else -2.0,
                "pose": 0.1,
            }
            trials.append(
                {
                    "case": FORWARD[0],
                    "completed": True,
                    "motion_class": group,
                    "reference_rewards": rewards,
                    "reference_return": sum(rewards.values()),
                    "reference_positive_return": sum(v for v in rewards.values() if v > 0),
                }
            )
    result = diagnose_h2(trials)
    assert result["eligible"] and result["competitor"] == "action_rate_l2"
    assert not diagnose_h2(trials[:-1])["eligible"]
    for row in trials:
        row["reference_rewards"]["pose"] = row["reference_rewards"]["action_rate_l2"]
        row["reference_return"] = sum(row["reference_rewards"].values())
    assert not diagnose_h2(trials)["eligible"]


def test_physical_guard_rejects_a_faster_but_unstable_candidate():
    """A large tracking gain cannot bypass the straight-yaw safety guard."""
    rows = [
        {
            "case": c,
            "trial": i,
            "planar_error_mps": 0.2,
            "yaw_rms_radps": 0.1,
            "roll_pitch_rms_radps": 0.1,
            "angular_rms_radps": 0.3,
            "completed": True,
        }
        for c in ("standing",) + FORWARD + TURNS
        for i in range(16)
    ]
    treatment = [dict(r, planar_error_mps=0.1) for r in rows]
    assert screen_candidate({"trials": treatment}, {"trials": rows}, "v_std")["eligible"]
    treatment[0 + 16]["yaw_rms_radps"] = 1.0
    assert not screen_candidate({"trials": treatment}, {"trials": rows}, "v_std")["eligible"]
