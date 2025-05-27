import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env_basic.simulation_env import CombatEnv
from SimArg import InitialData, FighterDataIn, num_fighter
from env_basic.action import Action

class CombatGymEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, env: CombatEnv, agent_id=0):
        super(CombatGymEnv, self).__init__()
        self.combat_env = env
        self.data_initial = InitialData()
        self.datain = [FighterDataIn() for m in range(num_fighter)]
        self.agent_id = agent_id  # 默认是 0 号战机
        self.num_opponents = num_fighter - 1  # 对手数量
        self.max_distance = 100000.0  # 归一化参考距离
        self.current_step = 0

        # 定义观测空间维度
        self.obs_dim = 41 + self.num_opponents * 19  # 根据上面推导的公式
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # 定义动作空间维度
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([5.99, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),  
            shape=(5,),
            dtype=np.float32
        )

    def _get_obs(self):
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
        reward = 0.0

        # 获取当前观测数据
        self_data = self.data_output[self.agent_id].selfdata
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
            reward += 1.0
        self.prev_missile_count = self_data.left_missile

        # 4. 生存奖励：+0.1 / step（鼓励长期存活）
        reward += 0.1

        # 5. 导弹告警：检测到导弹威胁时 -1 分（鼓励规避）
        if alert_data.emergency_missile_num > 0:
            reward -= 1.0

        # 6. 接近敌机：根据最近敌机的距离给予动态奖励
        closest_distance = float('inf')
        for enemy in opponent_states:
            if enemy.state_Survive:
                dx = enemy.state_Longitude[0] - self_data.Longitude
                dy = enemy.state_Latitude[0] - self_data.Latitude
                dz = enemy.state_Altitude[0] - self_data.Altitude
                dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                closest_distance = min(closest_distance, dist)

        if closest_distance < 100_000:  # 单位：米
            reward += 0.01 * (100_000 - closest_distance) / 1000  # 距离越近得分越高

        # 7. 姿态与能量控制（可选）：
        speed = self_data.GroundSpeed
        altitude = self_data.Altitude
        energy = speed * speed + altitude * 9.8  # 简化动能+势能
        if hasattr(self, 'prev_energy'):
            energy_diff = energy - self.prev_energy
            reward += 0.001 * energy_diff  # 微弱奖励能量提升
        self.prev_energy = energy

        return float(reward)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.data_output = self.combat_env.reset(self.data_initial, self.datain)
        return self._get_obs(), {}

    def step(self, action):
        """
        action: np.ndarray of shape (6,)
            [
                action_type, fire,
                roll_angle_norm, pitch_angle_norm,
                load_factor_norm, speed_change_norm
            ]
        """

        move_action = int(np.clip(action[0], 0, 5))           # 动作类型
        fire_action = int(np.clip(action[1], 0, 1))           # 发射导弹

        # 参数解归一化
        roll_angle = np.clip(action[2] * 60.0, -60.0, 60.0)        # [-60, 60]
        pitch_angle = np.clip(action[3] * 30.0, -30.0, 30.0)       # [-30, 30]
        load_factor = np.clip((action[4] + 1.0) * 3.0 + 2.0, 2.0, 8.0)  # [2, 8]

        for i in range(num_fighter):
            if i == self.agent_id:
                # 设置战机控制模式
                self.datain[i].control_mode = 3
                self.datain[i].target_index = 1
                self.datain[i].fire = 1
                self.datain[i].missile_fire = fire_action

                arg = [0, 0]  # 默认参数

                if move_action == 0:  # 平飞
                    self.datain[i].control_input = [1, 1/9, 0, 0]

                elif move_action == 1:  # 速度追踪
                    # 使用 speed_change 控制速度变化
                    arg = [0, 0]
                    self.datain[i].control_input = Action(i).action_choose(
                        env=self.combat_env,
                        target=self.datain[i].target_index,
                        action_index=move_action,
                        arg=arg
                    )

                elif move_action == 2:  # 转弯
                    arg = [roll_angle, 0]
                    self.datain[i].control_input = Action(i).action_choose(
                        env=self.combat_env,
                        target=self.datain[i].target_index,
                        action_index=move_action,
                        arg=arg
                    )

                elif move_action == 3:  # 倾角追踪
                    arg = [pitch_angle, 0]
                    self.datain[i].control_input = Action(i).action_choose(
                        env=self.combat_env,
                        target=self.datain[i].target_index,
                        action_index=move_action,
                        arg=arg
                    )

                elif move_action == 4:  # 盘旋
                    arg = [pitch_angle, roll_angle]
                    self.datain[i].control_input = Action(i).action_choose(
                        env=self.combat_env,
                        target=self.datain[i].target_index,
                        action_index=move_action,
                        arg=arg
                    )

                elif move_action == 5:  # 筋斗
                    arg = [load_factor, 0]
                    self.datain[i].control_input = Action(i).action_choose(
                        env=self.combat_env,
                        target=self.datain[i].target_index,
                        action_index=move_action,
                        arg=arg
                    )

                else:
                    raise ValueError(f"Invalid move action: {move_action}")

            elif i == 1:
                # 固定策略控制 i == 1 飞机
                self.datain[i].control_mode = 2
                self.datain[i].target_index = 0
                self.datain[i].fire = 1
                arg = [0, 0]
                self.datain[i].control_input = Action(i).action_choose(
                    env=self.combat_env,
                    target=self.datain[i].target_index,
                    action_index=self.datain[i].control_mode,
                    arg=arg
                )
                if self.current_step % 3000 == 0:
                    self.datain[i].missile_fire = 1
                else:
                    self.datain[i].missile_fire = 0

        terminal, self.data_output = self.combat_env.update(self.datain)
        reward = self._calculate_reward()
        if terminal == 2:  # 战斗结束
            reward += 100
        if self.current_step >= self.data_initial.len_max:
            terminal = 0
        done = bool(terminal >= 0)
        truncated = False
        obs = self._get_obs()
        self.current_step += 1 
        print(f"current_step: {self.current_step}, reward: {reward}, done: {done}")

        return obs, reward, done, truncated, {}
    def render(self):
        pass

    def close(self):
        pass