import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env_basic.simulation_env import CombatEnv
from SimArg import InitialData, FighterDataIn, num_fighter
from env_basic.action import Action
from math import sqrt
import torch
from torch.distributions import Categorical


class CombatGymEnv(gym.Env):
    """
    Gym 环境封装类，用于强化学习训练中的战斗模拟。
    
    该类将底层战斗环境 (CombatEnv) 封装成符合 gym.Env 接口的形式，
    提供观测空间 (observation_space)、动作空间 (action_space)、
    重置 (reset)、步进 (step)、渲染 (render) 等标准接口。
    
    设计思想：
        - 统一与 OpenAI Gym 的接口规范，便于集成 RL 框架；
        - 抽象出观测向量的构建逻辑，包含自我状态、雷达信息、近距感知、态势感知、告警等；
        - 支持多智能体训练，通过 agent_id 区分不同战机；
        - 使用 MultiDiscrete 动作空间表示离散控制指令组合；
        - 归一化距离以提高神经网络训练稳定性。
    """

    def __init__(self, env: CombatEnv, agent_id=0):
        """
        初始化 CombatGymEnv 实例。
        
        输入参数：
            env (CombatEnv): 底层战斗模拟环境实例；
            agent_id (int): 当前智能体对应的战机编号，默认为 0。
            
        输出：
            None
            
        函数作用：
            - 初始化基础环境配置；
            - 设置观测空间和动作空间；
            - 记录当前智能体 ID、对手数量、最大归一化距离；
            - 初始化数据结构用于存储战机输入输出信息。
            
        设计思想：
            - 使用 gym.Env 基类初始化；
            - 定义观测空间为连续值的 Box 空间，支持浮点数；
            - 定义动作空间为 MultiDiscrete，表示多个离散指令的组合；
            - 初始数据和输入数据使用特定类进行封装，保持代码结构清晰；
            - 用字典记录动作执行情况，方便后续调试或策略分析；
            - 用时间戳记录上次发射导弹的时间，避免频繁发射。
        """
        super(CombatGymEnv, self).__init__() #
        self.combat_env = env
        self.data_initial = InitialData()
        self.datain = [FighterDataIn() for m in range(num_fighter)]
        self.agent_id = agent_id  # 默认是 0 号战机
        self.num_opponents = num_fighter - 1  # 对手数量
        self.max_distance = 100000.0  # 归一化参考距离
        self.current_step = 0
        self.action_cont = {1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0}
        self.last_missile_time = -np.inf   # 上次发射导弹的时间

        # 定义观测空间维度
        self.obs_dim = 41 + self.num_opponents * 19  # 根据上面推导的公式
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # 定义动作空间维度
        self.action_space = spaces.MultiDiscrete([6, 2, 2, 2, 2, 2], dtype=np.int32)

    def _get_obs(self):
        """
        获取当前智能体的观测向量。
        
        输入参数：
            None
            
        输出：
            obs (np.ndarray): 表示当前战机状态及周围环境信息的一维数组，类型为 float32。
            
        函数作用：
            构建一个完整的观测向量，包含以下五部分信息：
                1. 自身战机的状态数据；
                2. 雷达探测到的敌方战机信息；
                3. 近距传感器获取的敌机信息；
                4. 态势感知系统提供的敌机位置信息；
                5. 告警系统发出的威胁提示信息；
            最终返回拼接后的观测向量，并处理缺失值为 0。

        设计思想：
            - 分模块提取信息，提高可读性和维护性；
            - 使用统一的数据结构处理单个或多个目标（如雷达/态势信息），增强鲁棒性；
            - 缺失信息填充为 0，防止神经网络训练中出现 NaN；
            - 所有观测值最终合并为一维数组，适配强化学习模型输入要求；
            - 考虑了多种传感器融合（雷达、红外、态势感知、告警），提升智能体对战场环境的理解能力；
            - 数据标准化预留接口（如 max_distance），可用于归一化处理。
        """
        # 1. 获取自身状态
        self_data = self.data_output[self.agent_id].selfdata
        obs_self = [
            self_data.fighter_side, self_data.index, self_data.control_mode,
            self_data.left_bullet, self_data.left_missile, self_data.left_bloods,
            self_data.Longitude, self_data.Latitude, self_data.Altitude,
            self_data.NorthVelocity, self_data.EastVelocity, self_data.VerticalVelocity,
            self_data.NorthAcceleration, self_data.EastAcceleration, self_data.VerticalAcceleration,
            self_data.RollAngle, self_data.PitchAngle, self_data.YawAngle,
            self_data.PathPitchAngle, self_data.PathYawAngle,
            self_data.AttackAngle, self_data.SideslipAngle,
            self_data.RollRate, self_data.PitchRate, self_data.YawRate,
            self_data.NormalLoad, self_data.LateralLoad, self_data.LongitudeinalLoad,
            self_data.NormalVelocity, self_data.LateralVelocity, self_data.LongitudianlVelocity,
            self_data.TrueAirSpeed, self_data.IndicatedAirSpeed, self_data.GroundSpeed,
            self_data.MachNumber, self_data.NumberofFuel, self_data.Thrust,
            self_data.Missile1State, self_data.Missile2State,
            self_data.Missile3State, self_data.Missile4State
        ]

         # 2. 获取敌方战机雷达信息
        radar_obs = []
        if isinstance(self.data_output[self.agent_id].radardata, list):
            radars = self.data_output[self.agent_id].radardata
        else:
            radars = [self.data_output[self.agent_id].radardata]  # 包装成列表统一处理
        for enemy in radars:
            if enemy is None:
                radar_obs.extend([0, 0, 0, 0, 0, 0])
            else:
                radar_obs.extend([
                    enemy.radar_EleAngle, enemy.radar_AziAngle, enemy.radar_Distance,
                    enemy.radar_NorthVelocity, enemy.radar_EastVelocity, enemy.radar_VerticalVelocity
                ])

        # 3. 获取近距透明信息
        close_obs = []
        if isinstance(self.data_output[self.agent_id].closedata, list):
            closes = self.data_output[self.agent_id].closedata
        else:
            closes = [self.data_output[self.agent_id].closedata]
        for enemy in closes:
            if enemy is None:
                close_obs.extend([0, 0, 0])
            else:
                close_obs.extend([
                    enemy.close_EleAngle, enemy.close_AziAngle, enemy.close_Distance
                ])

        # 4. 获取态势信息
        state_obs = []
        if isinstance(self.data_output[self.agent_id].statedata, list):
            states = self.data_output[self.agent_id].statedata
        else:
            states = [self.data_output[self.agent_id].statedata]
        for enemy in states:
            if enemy is None or not enemy.state_Survive:
                state_obs.extend([0, 0, 0, 0])
            else:
                state_obs.extend([
                    enemy.state_Longitude, enemy.state_Latitude, enemy.state_Altitude, 1
                ])

        # 5. 获取告警信息
        alert_obs = [
            self.data_output[self.agent_id].alertdata.emergency_num,
            self.data_output[self.agent_id].alertdata.emergency_EleAngle,
            self.data_output[self.agent_id].alertdata.emergency_AziAngle,
            self.data_output[self.agent_id].alertdata.emergency_missile_num,
            self.data_output[self.agent_id].alertdata.emergency_missile_EleAngle,
            self.data_output[self.agent_id].alertdata.emergency_missile_AziAngle
        ]

        full_obs = obs_self + radar_obs + close_obs + state_obs + alert_obs

        for i,ob in enumerate(full_obs):
            if isinstance(ob, list):
                if len(ob) == 0 or ob[0] is None:
                    full_obs[i] = 0.0
                else:
                    full_obs[i] = ob[0]
        # 合并为完整观测向量
        obs = np.array(full_obs).astype(np.float32).flatten()
        # print("obs", obs.shape, obs)
        return obs

    def _calculate_reward(self):
        """
        根据当前战机状态和环境信息计算强化学习智能体的奖励值。
        
        输入参数：
            None
            
        输出：
            reward (float): 当前步的奖励值，用于指导策略更新。
            
        函数作用：
            通过多维度指标评估当前动作的效果，包括：
                - 生存状态（被击中惩罚）；
                - 武器使用（发射导弹鼓励）；
                - 威胁感知（导弹告警惩罚）；
                - 战术行为（接近敌机鼓励）；
                - 动作频率控制（限制高频操作）；
            最终返回综合评分作为 RL 训练的奖励信号。
            
        设计思想：
            - 多因素奖励机制，引导智能体学习战术行为；
            - 使用历史状态比较（如血量、导弹数）判断变化并给予反馈；
            - 鼓励攻击行为（发射导弹 + 接近敌人），惩罚无效或危险行为；
            - 对某些动作设置冷却机制，防止模型滥用特定指令；
            - 支持未来扩展（如能量管理、姿态控制等）；
            - 所有奖励项经过归一化处理，便于训练稳定收敛。
        """
        reward = 0.0

        # 获取当前观测数据
        self_data = self.data_output[self.agent_id].selfdata
        self_data_enemy = self.data_output[1].selfdata
        opponent_states = self.data_output[self.agent_id].statedata
        alert_data = self.data_output[self.agent_id].alertdata

        if not isinstance(opponent_states, list):
            opponent_states = [opponent_states]
        

        # # 1. 击落敌机：每击落一架 +100 分
        # if hasattr(self, 'prev_opponent_alive'):
        #     for i in range(len(self.prev_opponent_alive)):
        #         if self.prev_opponent_alive[i] and not opponent_states[i].state_Survive:
        #             reward += 100.0  # 击落一架敌机
        # self.prev_opponent_alive = [enemy.state_Survive for enemy in opponent_states]

        # 2. 被击中：-100 分（如果血量减少）
        if hasattr(self, 'prev_health'):
            if self_data.left_bloods < self.prev_health:
                reward -= 100.0
        self.prev_health = self_data.left_bloods

        # 3. 发射导弹：+1 分（鼓励使用武器）
        if hasattr(self, 'prev_missile_count') and self_data.left_missile < self.prev_missile_count:
            reward += 5.0
        self.prev_missile_count = self_data.left_missile

        if self_data_enemy.Missile1State == 3 or self_data_enemy.Missile2State == 3 or self_data_enemy.Missile3State == 3 or self_data_enemy.Missile4State == 3:
            reward += 2.0

        if self_data.Missile1State == 2  or self_data.Missile2State == 2 or self_data.Missile3State == 2 or self_data.Missile4State == 2:
            reward += 6.0
        if self_data.Missile1State == 3 or self_data.Missile2State == 3 or self_data.Missile3State == 3 or self_data.Missile4State == 3:
            reward -= 0.5
        # 4. 生存奖励：+0.1 / step（鼓励长期存活）
        # reward += 0.1

        # 5. 导弹告警：检测到导弹威胁时 -1 分（鼓励规避）
        if alert_data.emergency_missile_num > 0:
            reward -= 3.0

        # 6. 接近敌机：根据最近敌机的距离给予动态奖励
        for enemy in opponent_states:
            if enemy.state_Survive:
                dx = enemy.state_Longitude[0] - self_data.Longitude
                dy = enemy.state_Latitude[0] - self_data.Latitude
                dz = enemy.state_Altitude[0] - self_data.Altitude
                dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        if dist > 20.0:
            reward += 100.0 * sqrt(1/(dist))  # 距离越近得分越高

        # 7. 姿态与能量控制（可选）：
        # speed = self_data.GroundSpeed
        # altitude = self_data.Altitude
        # energy = speed * speed + altitude * 9.8  # 简化动能+势能
        # if hasattr(self, 'prev_energy'):
        #     energy_diff = energy - self.prev_energy
        #     reward += 0.01 * energy_diff  # 微弱奖励能量提升
        # self.prev_energy = energy

        if self.action_cont[2] >= 100:
            reward -= 1.0
            self.action_cont[2] = 0
        
        punish_list = [1,3,4,5,6]
        for i in punish_list:
            if self.action_cont[i] >= 100:
                reward -= 3.0
                self.action_cont[i] = 0

        return float(reward)
    
   
    def reset(self, seed=None, options=None):
        """
        重置战斗模拟环境，开始新的回合。
        
        输入参数：
            seed (int): 随机种子，用于控制随机性；
            options (dict): 可选配置参数（预留接口）。
            
        输出：
            observation (np.ndarray): 初始观测向量；
            info (dict): 环境初始化信息（为空字典）。
            
        函数作用：
            - 调用底层 CombatEnv 的 reset 方法进行环境重置；
            - 返回初始观测状态供 RL 策略使用；
            - 支持 Gym API 规范的 reset 接口。

        设计思想：
            - 继承 gym.Env 的 reset 接口，兼容各种 RL 框架；
            - 通过调用底层环境实现真正的重置逻辑；
            - 初始化观测数据，保证每轮训练起点一致；
            - 提供 seed 支持以确保实验可重复性；
            - 使用 info 字段预留调试信息输出接口。
        """
        super().reset(seed=seed)
        self.data_output = self.combat_env.reset(self.data_initial, self.datain)
        return self._get_obs(), {}

    def step(self, action):
        """
        执行一个环境步进（即执行动作并返回下一状态、奖励等信息）。
        
        输入参数：
            action (np.ndarray): 由智能体输出的动作指令，形状为 (6,) 的整数数组，表示如下：
                [
                    move_action_type (0~5),   # 移动模式选择
                    missile_fire (0/1),       # 是否发射导弹
                    fire (0/1),               # 是否开火
                    roll_angle_norm (0/1/2),  # 滚转角度控制（归一化）
                    pitch_angle_norm (0/1/2), # 俯仰角控制（归一化）
                    load_factor_norm (0/1/2)  # 过载因子控制（归一化）
                ]

        输出：
            obs (np.ndarray): 下一时刻的观测向量；
            reward (float): 当前步的奖励值；
            done (bool): 是否回合结束；
            truncated (bool): 是否因时间限制而终止（当前未使用）；
            info (dict): 调试或扩展信息（当前为空字典）；

        函数作用：
            - 解析动作指令；
            - 根据动作设置战机的控制输入；
            - 更新底层战斗环境；
            - 计算奖励；
            - 返回观测、奖励、是否终止等信息以供 RL 策略更新；
            
        设计思想：
            - 使用 MultiDiscrete 动作空间实现组合式控制；
            - 将动作映射到具体的飞行行为（平飞、转弯、筋斗等）；
            - 控制敌方飞机采用固定策略，便于训练我方策略；
            - 对导弹发射进行冷却控制，避免滥用；
            - 支持 Gym 接口规范，便于集成各种强化学习算法；
            - 使用 terminal 判断游戏状态，并据此给予不同奖励。
        """
        move_action = action[0] + 1
        self.action_cont[move_action] += 1
        missle_fire_action = action[1]
        fire_action = action[2]
        roll_angle = (action[3]-1)*10
        pitch_angle = (action[4]-1)*10
        load_factor = (action[5]-1)*3

        for i in range(num_fighter):
            if i == self.agent_id:
                # 设置战机控制模式
                self.datain[i].control_mode = 3
                self.datain[i].target_index = 1
                self.datain[i].fire = fire_action
                self.datain[i].missile_fire = missle_fire_action

                arg = [0, 0]  # 默认参数

                if move_action == 1:  # 平飞
                    self.datain[i].control_input = [1, 1/9, 0, 0]

                elif move_action == 2:  # 速度追踪
                    # 使用 speed_change 控制速度变化
                    arg = [0, 0]
                    self.datain[i].control_input = Action(i).action_choose(
                        env=self.combat_env,
                        target=self.datain[i].target_index,
                        action_index=2,
                        arg=arg
                    )

                elif move_action == 3:  # 转弯
                    arg = [roll_angle, 0]
                    self.datain[i].control_input = Action(i).action_choose(
                        env=self.combat_env,
                        target=self.datain[i].target_index,
                        action_index=3,
                        arg=arg
                    )

                elif move_action == 4:  # 倾角追踪
                    arg = [pitch_angle, 0]
                    self.datain[i].control_input = Action(i).action_choose(
                        env=self.combat_env,
                        target=self.datain[i].target_index,
                        action_index=4,
                        arg=arg
                    )

                elif move_action == 5:  # 盘旋
                    arg = [pitch_angle, roll_angle]
                    self.datain[i].control_input = Action(i).action_choose(
                        env=self.combat_env,
                        target=self.datain[i].target_index,
                        action_index=5,
                        arg=arg
                    )

                elif move_action == 6:  # 筋斗
                    arg = [load_factor, 0]
                    self.datain[i].control_input = Action(i).action_choose(
                        env=self.combat_env,
                        target=self.datain[i].target_index,
                        action_index=6,
                        arg=arg
                    )

                else:
                    raise ValueError(f"Invalid move action: {move_action}")

            elif i == 1:
                # 固定策略控制 i == 1 飞机
                self.datain[i].control_mode = 3
                self.datain[i].target_index = 0
                self.datain[i].fire = 1
                arg = [0, 0]
                self.datain[i].control_input = Action(i).action_choose(
                    env=self.combat_env,
                    target=self.datain[i].target_index,
                    action_index=2,
                    arg=arg
                )
                if self.current_step % 3000 == 0:
                    self.datain[i].missile_fire = 1
                else:
                    self.datain[i].missile_fire = 0

        terminal, self.data_output = self.combat_env.update(self.datain)
        if self.current_step >= self.data_initial.len_max:
            terminal = 0
        reward = self._calculate_reward()
        if terminal == 2:  # 战斗结束
            reward += 500
        elif terminal == 1:
            reward -= 100
        elif terminal == 0:
            reward -= 20
        done = bool(terminal >= 0)
        truncated = False
        obs = self._get_obs()
        self.current_step += 1 
        # print(f"action count: {self.action_cont}, missle action: {missle_fire_action}, reward: {reward}, left_missles: {self.data_output[self.agent_id].selfdata.left_missile}")
        # print(f"current_step: {self.current_step}, action: {move_action}, reward: {reward}, missle action: {missle_fire_action}")

        return obs, reward, done, truncated, {}
    
    def render(self):
        pass

    def close(self):
        pass