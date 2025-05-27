# 🐍强化学习空战项目Policy Ascender

## 1. 安装教程

创建 Conda 环境并安装依赖，本项目在cuda 12.1上测试：

```bash
conda create -n snake python=3.9.16
conda activate snake

conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt
```

---

## 2. 使用 TensorBoard 查看训练进度

在训练过程中，可以通过以下命令启动 TensorBoard 查看训练日志：

```bash
tensorboard --logdir=./logs/
```

访问地址：[http://localhost:6006](http://localhost:6006)

---

## 3. 使用教程

本项目使用 **PPO** 算法进行在线强化学习训练。  
相关定义如下：

- **Action Space**、**Reward Function**、**Observation Space**：定义在 `strategy/sim_gym.py` 中。

### 3.1 训练流程

1. 修改 `strategy/sim_gym.py` 中的 reward、action 或 observation 设计。
2. 运行训练脚本：

```bash
python train.py
```

- 训练完成后的最终模型保存路径为：`models/ppo_combat.zip`
- 最佳模型保存路径为：`models/ppo_combat_best/best_model.zip`

> 同时可在另一个终端运行 TensorBoard 命令查看训练进度。

---

### 3.2 测试流程

1. 修改 `test.py` 中加载的权重路径（如更换为其他模型）。
2. 运行测试脚本：

```bash
python test.py
```

---

如有问题或需要进一步配置说明，请参考项目文档或联系作者。