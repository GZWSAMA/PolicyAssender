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

# ===== 新增参数：是否继续训练 =====
continue_training = True  # 设置为 True 表示继续训练；False 表示从头开始

# ==================================

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
os.makedirs(models_root, exist_ok=True)

existing = [d for d in os.listdir(models_root) if d.startswith("exp_") and d[4:].isdigit()]
if existing:
    max_idx = max([int(d[4:]) for d in existing])
    i = max_idx + 1
else:
    i = 1

# pretrained_model_path = f"models/exp_{i-1}/ppo_combat_best/best_model.zip"
pretrained_model_path = f"success_examples/7_all_alive_S_trace/ppo_combat_best/best_model.zip"

model_save_path = os.path.join(models_root, f"exp_{i}", "ppo_combat")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# 构建评估环境并包装 Monitor
eval_env = CombatGymEnv(combat_env, agent_id=0)
eval_env = Monitor(eval_env)

# 可选：添加评估回调
eval_callback = EvalCallback(
    eval_env,
    eval_freq=5000,               # 每 10000 步评估一次
    best_model_save_path=model_save_path + "_best",
    deterministic=True,
    render=False
)

# ===== 加载模型逻辑调整 =====
if continue_training and os.path.exists(pretrained_model_path):
    print(f"Loading pretrained model from {pretrained_model_path}")
    model = PPO.load(pretrained_model_path, learning_rate=3e-5, env=env, device='cpu')
else:
    print("Training from scratch.")
    model = PPO(
        "MlpPolicy",
        # "MultiInputPolicy",
        env,
        policy_kwargs=dict(
            net_arch=[512, 256]  # 使用两层隐藏层，每层256个神经元
        ),
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=4096,
        ent_coef=0.1,
        batch_size=256,
        n_epochs=20,
        clip_range=0.2,
        gamma=0.995,
        device='cpu'  # 使用 CPU 进行训练
    )
# ============================

# 开始训练，并添加进度条回调
model.learn(
    total_timesteps=20480*30,  # 总训练步数
    callback=[eval_callback, ProgressBarCallback()],  # 添加进度条
    tb_log_name="PPO_Combat"
)

# 保存最终模型
model.save(model_save_path)

print("训练完成，模型已保存")