import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from strategy.sim_gym import CombatGymEnv
from env_basic.simulation_env import CombatEnv
from SimArg import InitialData, FighterDataIn, num_fighter
from tqdm import tqdm

# 创建基础 CombatEnv 实例
data_initial = InitialData()
datain = [FighterDataIn() for m in range(num_fighter)]
combat_env = CombatEnv(data_initial, datain)

# 封装为 Gym 环境
env = CombatGymEnv(combat_env, agent_id=0)
env = Monitor(env)  # 推荐包装以记录 episode 数据

# 加载模型
model = PPO.load("models/exp_59/ppo_combat_best/best_model.zip", device='cpu')  # 替换为你自己的模型路径

# 运行评估
obs, _ = env.reset()
total_reward = 0
episode_length = 0
done = False

with tqdm(total=30000, desc="Evaluating", unit="step") as pbar:  #添加进度条
  while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    episode_length += 1
    pbar.update(1)  #更新进度条

    # print(f"Step: {episode_length}, Action: {action}, Reward: {reward:.2f}, Done: {done}")

    # 如果当前 episode 结束，重置环境
    if done or truncated:
        pbar.set_postfix({"Episode Reward": f"{total_reward:.2f}"})
        # print(f"✅ Episode finished. Total reward: {total_reward:.2f}, Length: {episode_length}")
        obs, _ = env.reset()
        total_reward = 0
        episode_length = 0