import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from strategy.sim_gym import CombatGymEnv
from env_basic.simulation_env import CombatEnv
from SimArg import InitialData, FighterDataIn, num_fighter
import os

# 创建基础 CombatEnv 实例（用于驱动底层仿真）
data_initial = InitialData()
datain = [FighterDataIn() for m in range(num_fighter)]
combat_env = CombatEnv(data_initial, datain)

# 将其封装为 Gym 兼容的 CombatGymEnv，并用 Monitor 包装
env = CombatGymEnv(combat_env, agent_id=0)
env = Monitor(env)
check_env(env)

# 设置日志和保存路径
log_dir = "./logs/"
models_root = "./models/"

# 自动查找已有编号，延续保存
if not os.path.exists(models_root):
    os.makedirs(models_root)

existing = [d for d in os.listdir(models_root) if d.startswith("exp_") and d[6:].isdigit()]
if existing:
    max_idx = max([int(d[6:]) for d in existing])
    i = max_idx + 1
else:
    i = 1

model_save_path = os.path.join(models_root, f"exp_{i}", "ppo_combat")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# 构建评估环境并包装 Monitor
eval_env = CombatGymEnv(combat_env, agent_id=0)
eval_env = Monitor(eval_env)

# 可选：添加评估回调
eval_callback = EvalCallback(
    eval_env,
    eval_freq=1000,               # 每 1000 步评估一次
    best_model_save_path=model_save_path + "_best",
    deterministic=True,
    render=False
)

# 构建 PPO 模型
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(
        net_arch=[256, 256]  # 使用两层隐藏层，每层256个神经元
    ),
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=3e-4,
    n_steps=2048,
    ent_coef=0.01,
    batch_size=64,
    n_epochs=10,
    clip_range=0.2,
    gamma=0.99,
    device='cuda' if th.cuda.is_available() else 'cpu'  # 使用 CPU 进行训练
)

# 开始训练，并添加进度条回调
model.learn(
    total_timesteps=100000,  # 总训练步数
    callback=[eval_callback, ProgressBarCallback()],  # 添加进度条
    tb_log_name="PPO_Combat"
)

# 保存最终模型
model.save(model_save_path)

print("训练完成，模型已保存")