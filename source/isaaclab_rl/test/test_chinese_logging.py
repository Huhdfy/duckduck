# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Chinese log presentation preserves real RSL-RL aggregation, iteration indices and relative saves."""

from types import SimpleNamespace

import pytest
import torch
from rsl_rl.utils.logger import Logger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from isaaclab_rl.utils.chinese_logging import TrainingLogAdapter


def test_chinese_logs_preserve_upstream_scalars_and_relative_save(tmp_path, capsys):
    """Read actual event files: translation changes labels only, and relative milestones use completed updates."""
    runners = []
    for language in ("en", "zh"):
        cfg = {"logger": "tensorboard", "num_steps_per_env": 24, "algorithm": {}, "run_name": "test"}
        logger = Logger(str(tmp_path / language), cfg, {}, 2, False, 1, 0, "cpu")
        logger._store_code_state = lambda: []
        saved = []
        runner = SimpleNamespace(logger=logger, save=saved.append)
        adapter = TrainingLogAdapter(runner, language, save_interval=2)
        adapter.init_logging_writer()
        for iteration in (6000, 6001):
            adapter.process_env_step(
                torch.tensor([2.0, 4.0]),
                torch.tensor([1, 1]),
                {
                    "log": {"Episode_Reward/track_lin_vel": torch.tensor(0.25)},
                },
            )
            adapter.log(
                it=iteration,
                start_it=6000,
                total_it=6002,
                collect_time=1.0,
                learn_time=0.5,
                loss_dict={"surrogate": 0.5},
                learning_rate=0.00017,
                action_std=torch.ones(2),
                rnd_weight=None,
            )
        adapter.stop_logging_writer()
        assert len(saved) == 1 and saved[0].endswith("model_6001.pt")
        events = EventAccumulator(str(tmp_path / language))
        events.Reload()
        runners.append(events)
    assert "新增轮次：2/2" in capsys.readouterr().out
    for english, chinese in (
        ("Loss/learning_rate", "优化/学习率"),
        ("Train/mean_reward", "训练/平均回报"),
        ("Episode_Reward/track_lin_vel", "奖励/线速度跟踪"),
    ):
        assert [(v.step, v.value) for v in runners[0].Scalars(english)] == [
            (v.step, v.value) for v in runners[1].Scalars(chinese)
        ]
    assert runners[1].Scalars("训练/平均回报")[0].value == pytest.approx(3.0)
