import random

fighter_str = '.\MultiFighter.dll'
missile_str = '.\close_middle_py.dll'
epoch_max = 1               # 设定仿真轮次
len_max = 18000             # 单轮仿真的步长上限

num_blue = 1                 # 设定蓝色机数量
num_red = 1                  # 设定红色机数量
fighters_max_num = 10

num_fighter = num_red + num_blue


class InitialData(object):
    def __init__(self):

        self.dll_str = '.\MultiFighter.dll'         # 调用的模型路径，若加载失败，可改为绝对路径

        # 仿真设定
        self.epoch_max = epoch_max              # 仿真轮次
        self.len_max = len_max         # 单轮仿真长度（步长0.01s）
        # 红蓝双方战机数量
        self.num_blue = num_blue  # 蓝机数量
        self.num_red = num_red  # 红机数量
        self.originLongitude = 160.123456   # 仿真的经纬度原点位置
        self.originLatitude = 24.8976763

        # ———————————————————————————— 对抗双方的初始状态 ————————————————————————————
        # 初始位置（北东地）
        self.NED = []
        h = random.randint(8000, 12000)
        for i in range(self.num_blue):
            self.NED.append([0, 1000 * i, -h])
        for i in range(self.num_blue):
            self.NED.append([0, 1000 * i + 15000, -h])

        # 初始速度（马赫数）
        self.ma = []
        for i in range(self.num_blue + self.num_red):
            ma = random.uniform(0.8, 1.2)
            self.ma.append(ma)

        # 初始航向（0度为北向）
        self.orientation = []
        for i in range(self.num_blue + self.num_red):
            self.orientation.append(0)

        # 初始化控制模式
        # 控制模式3：[油门， 期望机体法向过载， 期望机体滚转速率， 无意义补充位]
        # 控制模式0：[油门， 纵向杆， 横向杆， 方向舵]
        self.control_mode = []
        for i in range(self.num_blue + self.num_red):
            self.control_mode.append(3)

        self.fully_combat = False      # 全透明与半透明模式的开关，当为True时，为全透明模式。导弹可以实时获取目标信息，且关闭雷达模块

class FighterDataIn(object):
    def __init__(self):
        # 控制模式输入
        self.control_mode = 3

        # 直接控制输入,依据操作模式代表不同含义
        self.control_input = [1, 1/9, 0, 0]

        # 航炮开火指令，1为发射，0为不发射
        self.fire = 0

        # 机载雷达锁定目标，以2v2对抗为例，机载雷达中，蓝色机编号依次为0,1，红色机编号为2,3
        self.target_index = 0

        # 导弹开火指令，1为发射，0为不发射
        self.missile_fire = 0         # 近距弹发射


