# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck walking rsl_rl_ppo_cfg."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.utils import preset

from ..flat_env_cfg import MICRODUCK_STEPS_PER_ITERATION


@configclass
class MicroDuckPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for the MicroDuck velocity-tracking tasks."""

    # 每轮每个环境采样 24 个控制步；4096 个环境合计 98304 条转移，再用于 PPO 更新。
    # 此步数也用于课程阶段换算，修改时需同步考虑课程进度。
    num_steps_per_env = MICRODUCK_STEPS_PER_ITERATION
    max_iterations = 50000
    save_interval = 250
    experiment_name = preset(default="microduck_velocity_flat", backlash="microduck_velocity_flat_backlash")
    # actor 根据可部署观测生成动作，critic 利用特权观测估计价值；两者使用独立网络。
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        # 训练时用高斯动作分布探索；init_std 是动作空间中的初始标准差，不是观测噪声。
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        # 同一轮采样数据分成 4 个小批次并重复学习 5 遍；不要将学习遍数与环境采样步数混淆。
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=0.001,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        # 自适应学习率依据新旧策略的 KL 差异调节；该阈值控制更新幅度，不是奖励收敛阈值。
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class MicroDuckRoughPPORunnerCfg(MicroDuckPPORunnerCfg):
    """The same PPO recipe with a separate experiment directory for rough terrain."""

    experiment_name = preset(default="microduck_velocity_rough", backlash="microduck_velocity_rough_backlash")
