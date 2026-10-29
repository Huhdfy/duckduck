# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-run Chinese presentation of RSL-RL metrics, without changing upstream aggregation."""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

# Translate whole path components first; keep units explicit and joint identifiers unambiguous.
_NAMES = {
    "Episode_Reward": "奖励",
    "Episode_Termination": "结束原因",
    "Episode": "回合",
    "Diagnostics": "诊断",
    "Metrics": "指标",
    "Curriculum": "课程",
    "Loss": "优化",
    "Policy": "策略",
    "Perf": "性能",
    "Train": "训练",
    "all": "全部指令",
    "forward": "直行指令",
    "mean_reward": "平均回报",
    "mean_episode_length": "平均回合步数",
    "mean_std": "动作标准差",
    "learning_rate": "学习率",
    "total_fps": "每秒采样数",
    "collection_time": "采样耗时秒",
    "learning_time": "更新耗时秒",
    "time": "按耗时",
    "surrogate": "策略损失",
    "value_function": "价值损失",
    "entropy": "策略熵",
    "track_lin_vel": "线速度跟踪",
    "track_ang_vel": "角速度跟踪",
    "upright": "躯干直立",
    "pose": "腿部姿态",
    "body_ang_vel": "躯干晃动",
    "angular_momentum": "整机角动量",
    "dof_pos_limits": "关节软限位",
    "action_rate_l2": "动作变化",
    "air_time": "腾空时间",
    "foot_clearance": "足部离地",
    "foot_slip": "足部滑移",
    "foot_swing_height": "摆脚高度",
    "self_collision": "自身碰撞",
    "self_collisions": "自身碰撞",
    "head_pose_tracking": "头部姿态",
    "head_pose_bias": "头部持续偏差",
    "joint_tracking_error": "关节跟踪误差",
    "time_out": "回合超时",
    "fell_over": "摔倒",
    "nan_state": "非有限状态",
    "error_vel_xy": "平面速度误差",
    "error_vel_yaw": "偏航速度误差",
    "tracking_exceedance_fraction": "关节超容差比例",
    "tracking_deadzone_cost": "关节超差代价",
    "joint_error_rms_rad": "关节跟踪均方根弧度",
    "reward_weighted": "加权奖励",
    "total": "合计",
    "link_vx_mean_mps": "机体前向速度均值米每秒",
    "com_vx_mean_mps": "质心前向速度均值米每秒",
    "link_vx_error_mean_mps": "前向速度偏差米每秒",
    "link_vx_error_rms_mps": "前向速度误差均方根米每秒",
    "link_vy_rms_mps": "侧向速度均方根米每秒",
    "link_vz_rms_mps": "竖向速度均方根米每秒",
    "reference_vx_delta_mean_mps": "速度参考点前向差米每秒",
    "reference_velocity_delta_rms_mps": "速度参考点差均方根米每秒",
    "yaw_error_mean_radps": "偏航速度偏差弧度每秒",
    "yaw_error_rms_radps": "偏航速度误差均方根弧度每秒",
    "roll_pitch_sq_mean_rad2ps2": "横滚俯仰角速度平方和",
    "yaw_factor_mean": "偏航奖励因子",
    "roll_pitch_factor_mean": "横滚俯仰奖励因子",
    "angular_reward_raw_mean": "角速度原始奖励",
    "action_rate_weight": "动作变化权重",
    "head_pose_bias_weight": "头部偏差权重",
    "standing_envs": "站立比例",
    "head_pose_range": "头部指令范围",
    "com_range": "质心扰动范围",
    "head_com_range": "头部质心扰动范围",
    "success_rate": "成功率",
    "base_velocity": "机体速度",
    "left_hip_yaw": "左髋偏航",
    "left_hip_roll": "左髋横滚",
    "left_hip_pitch": "左髋俯仰",
    "left_knee": "左膝",
    "left_ankle": "左踝",
    "right_hip_yaw": "右髋偏航",
    "right_hip_roll": "右髋横滚",
    "right_hip_pitch": "右髋俯仰",
    "right_knee": "右膝",
    "right_ankle": "右踝",
    "neck_pitch": "颈俯仰",
    "head_pitch": "头俯仰",
    "head_yaw": "头偏航",
    "head_roll": "头横滚",
    "left_joint_error_rms_rad": "左腿关节误差均方根弧度",
    "right_joint_error_rms_rad": "右腿关节误差均方根弧度",
    "head_joint_error_rms_rad": "头部关节误差均方根弧度",
    "left_target_offset_rms_rad": "左腿目标偏移均方根弧度",
    "right_target_offset_rms_rad": "右腿目标偏移均方根弧度",
    "left_actual_offset_rms_rad": "左腿实际偏移均方根弧度",
    "right_actual_offset_rms_rad": "右腿实际偏移均方根弧度",
}
_TOKENS = {
    "left": "左",
    "right": "右",
    "hip": "髋",
    "yaw": "偏航",
    "roll": "横滚",
    "pitch": "俯仰",
    "knee": "膝",
    "ankle": "踝",
    "neck": "颈",
    "head": "头",
    "joint": "关节",
    "error": "误差",
    "target": "目标",
    "actual": "实际",
    "offset": "偏移",
    "rms": "均方根",
    "mean": "均值",
    "rad": "弧度",
    "radps": "弧度每秒",
    "mps": "米每秒",
    "m": "米",
    "s": "秒",
    "contact": "接触",
    "fraction": "比例",
    "stance": "支撑",
    "duration": "时长",
    "touchdowns": "落地次数",
    "per": "每",
    "sole": "脚底",
    "speed": "速度",
    "landing": "落点",
    "b": "机体系",
    "w": "世界系",
    "xy": "平面",
    "x": "前向",
    "y": "侧向",
    "z": "竖向",
    "cost": "代价",
    "feet": "双脚",
    "height": "高度",
    "swing": "摆动",
    "clearance": "离地",
    "slip": "滑移",
    "limits": "限位",
    "deviation": "偏离",
    "torques": "力矩",
    "l2": "平方和",
    "position": "位置",
    "velocity": "速度",
    "penalty": "惩罚",
    "reward": "奖励",
    "rate": "变化率",
    "self": "自身",
    "success": "成功",
    "value": "价值",
    "collisions": "碰撞",
    "collision": "碰撞",
    "dof": "关节",
    "pos": "位置",
    "vel": "速度",
    "tracking": "跟踪",
    "lin": "线",
    "ang": "角",
    "weight": "权重",
    "base": "机体",
}
_TEXT = {
    "Learning iteration": "训练轮次",
    "Run name:": "实验名称：",
    "Total steps:": "本次采样数：",
    "Steps per second:": "每秒采样数：",
    "Collection time:": "采样耗时：",
    "Learning time:": "更新耗时：",
    "Mean reward:": "平均回报：",
    "Mean episode length:": "平均回合步数：",
    "Mean action std:": "动作标准差：",
    "Mean surrogate loss:": "策略损失：",
    "Mean value_function loss:": "价值损失：",
    "Mean value loss:": "价值损失：",
    "Mean entropy loss:": "策略熵：",
    "Iteration time:": "本轮耗时：",
    "Time elapsed:": "已用时间：",
    "ETA:": "预计剩余：",
    "Mean episode ": "回合均值 ",
}


def chinese_tag(tag: str) -> str:
    """Translate scalar paths without changing their hierarchy or merging distinct identifiers."""
    return "/".join(
        _NAMES.get(part, "_".join(_TOKENS.get(token, token) for token in part.split("_"))) for part in tag.split("/")
    )


def chinese_text(text: str) -> str:
    """Translate RSL-RL's console labels while preserving numeric formatting."""
    for original, translated in _TEXT.items():
        text = text.replace(original, translated)
    return re.sub(r"[A-Za-z_][A-Za-z_0-9]*(?:/[A-Za-z_0-9]+)+", lambda m: chinese_tag(m[0]), text)


class _ChineseWriter:
    def __init__(self, writer):
        self.writer = writer
        self.tags: dict[str, str] = {}

    def __getattr__(self, name):
        return getattr(self.writer, name)

    def add_scalar(self, tag, value, step, *args, **kwargs):
        translated = chinese_tag(tag)
        previous = self.tags.setdefault(translated, tag)
        if previous != tag:
            raise ValueError(f"Chinese metric collision: {previous!r}, {tag!r} -> {translated!r}")
        return self.writer.add_scalar(translated, value, step, *args, **kwargs)


class TrainingLogAdapter:
    """Delegate bookkeeping to RSL-RL, translating only presentation and optional relative saves.

    Installed on one runner, never on the dependency module. Stdout capture covers only its synchronous
    log call. Metric values, iteration indices, episode aggregation and the optimizer are untouched.
    """

    def __init__(self, runner, language: str, save_interval: int | None = None):
        self.runner = runner
        self.delegate = runner.logger
        self.language = language
        self.save_interval = save_interval

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def init_logging_writer(self):
        self.delegate.init_logging_writer()
        if self.language == "zh" and self.delegate.writer is not None:
            self.delegate.writer = _ChineseWriter(self.delegate.writer)

    def stop_logging_writer(self):
        self.delegate.stop_logging_writer()
        writer = self.delegate.writer
        if writer is not None and hasattr(writer, "flush"):
            writer.flush()
            writer.close()

    def log(self, *args, **kwargs):
        if self.language == "zh":
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.delegate.log(*args, **kwargs)
            print(chinese_text(output.getvalue()), end="")
            print(f"新增轮次：{kwargs['it'] - kwargs['start_it'] + 1}/{kwargs['total_it'] - kwargs['start_it']}")
        else:
            self.delegate.log(*args, **kwargs)
        completed = kwargs["it"] - kwargs["start_it"] + 1
        if self.save_interval and completed % self.save_interval == 0 and self.writer is not None:
            self.runner.save(str(Path(self.log_dir) / f"model_{kwargs['it']}.pt"))
