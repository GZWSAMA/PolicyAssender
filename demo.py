#coding:utf-8
from env_basic.simulation_env import CombatEnv
from SimArg import InitialData, FighterDataIn, num_fighter
from env_basic.action import Action


if __name__ == '__main__':
    # ##################### 初始化设定 ########################
    # 初始化环境与输入量
    data_initial = InitialData()
    datain = [FighterDataIn() for m in range(num_fighter)]
    env = CombatEnv(data_initial, datain)
    # 内置机动动作实例化
    action_0 = Action(0)
    action_1 = Action(1)
    # ###################### 开始多轮仿真 ######################
    for i_episode in range(1):
        # 重置仿真
        data_output = env.reset(data_initial, datain)

        # ###################### 开始单轮仿真 ######################
        for t in range(data_initial.len_max):
            # —————————————————————————————————— 编辑输入控制数据 ——————————————————————————————————
            for i in range(num_fighter):
                if i == 0:
                    datain[i].control_mode = 3
                    datain[i].target_index = 1
                    datain[i].fire = 1
                    arg = [0, 0]
                    datain[i].control_input = action_0.action_choose(env=env, target=datain[i].target_index,
                                                                     action_index=2, arg=arg)
                    if t % 200 == 0:
                        datain[i].missile_fire = 1  # 近距弹发射
                    else:
                        datain[i].missile_fire = 0

                elif i == 1:
                    datain[i].control_mode = 3
                    datain[i].target_index = 0
                    datain[i].fire = 1
                    arg = [-20, 0]
                    datain[i].control_input = action_1.action_choose(env=env, target=datain[i].target_index,
                                                                     action_index=3, arg=arg)
                    if t % 200 == 0:
                        datain[i].missile_fire = 1  # 近距弹发射
                    else:
                        datain[i].missile_fire = 0

                # 打印输出数据示例
                print('data_output', data_output[i].radardata.target_index, data_output[i].statedata.state_Latitude)

            # 单个回合更新
            terminal, data_output = env.update(datain)    # terminal终止代码：-1正常运行;0仿真时长最大;1蓝方全灭;2红方全灭;3双方均全灭

            # 打印数据
            print('\n仿真步长：', t, '仿真时间： ', t * 0.01)

            # 仿真结束
            if terminal >= 0:
                print("Episode: \t{} ,episode len is: \t{}".format(i_episode, t))
                print(terminal)
                break

