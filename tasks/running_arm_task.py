import numpy as np
import transforms3d as tf3  
from tasks import rewards 
from enum import Enum, auto  


class WalkModes(Enum):
    """行走模式枚举类，定义机器人可能处于的三种行走状态"""
    STANDING = auto()  # 站立模式：机器人静止站立，可能只有微调
    INPLACE = auto()  # 原地踏步模式：机器人原地踏步，没有前进速度
    FORWARD = auto()  # 前进行走模式：机器人向前行走

    def encode(self):
        """
        将行走模式编码为one-hot向量，便于神经网络输入
        
        Returns:
            np.array: 长度为3的one-hot编码向量
        """
        if self.name == 'STANDING':
            return np.array([0, 0, 1])  # 站立模式：[0,0,1]
        elif self.name == 'INPLACE':
            return np.array([0, 1, 0])  # 原地踏步：[0,1,0]
        elif self.name == 'FORWARD':
            return np.array([1, 0, 0])  # 前进模式：[1,0,0]

    def sample_ref(self):
        """
        根据当前模式采样参考值（速度或角速度）
        用于生成训练数据时的目标值
        
        Returns:
            float: 速度参考值(m/s)或偏航角速度参考值(rad/s)
        """
        if self.name == 'STANDING':
            # 站立模式：随机偏航角速度，允许小范围转动
            return np.random.uniform(-1, 1)
        if self.name == 'INPLACE':
            # 原地踏步：较小的随机偏航角速度
            return np.random.uniform(-0.5, 0.5)
        if self.name == 'FORWARD':
            # return np.random.uniform(0.5, 2.0)
            # 前进模式：固定高速6.0 m/s（类似于跑步速度）
            return 6.0


class RunningArmTask(object):
    """
    双足机器人动态稳定行走任务类
    
    主要功能：
    1. 定义行走任务的目标和约束
    2. 计算多目标奖励函数
    3. 管理行走模式的切换
    4. 检查任务终止条件
    """

    def __init__(self,
                 client=None,  # 机器人MuJoCo客户端接口
                 dt=0.025,  # 控制时间步长（40Hz控制频率）
                 neutral_foot_orient=[],  # 中立状态下的脚部朝向（四元数）
                 neutral_pose=[],  # 中立姿势（所有关节的目标角度）
                 root_body='pelvis',  # 根节点（骨盆）的体名称
                 lfoot_body='lfoot',  # 左脚的体名称
                 rfoot_body='rfoot',  # 右脚的体名称
                 head_body='head',  # 头部的体名称
                 waist_r_joint='waist_r',  # 腰部横滚（roll）关节名称
                 waist_p_joint='waist_p',  # 腰部俯仰（pitch）关节名称
                 manip_hfield=False,  # 是否操作高度场（用于地形适应训练）
                 ):

        # 保存基本参数
        self._client = client  # 机器人客户端接口
        self._control_dt = dt  # 控制时间步长
        self._neutral_foot_orient = neutral_foot_orient  # 中立脚部朝向
        self._neutral_pose = np.array(neutral_pose)  # 转换为numpy数组
        self.manip_hfield = manip_hfield  # 高度场操作标志

        # 获取机器人质量（用于动力学计算）
        self._mass = self._client.get_robot_mass()

        # 初始化运行时变量
        self.mode_ref = []  # 当前模式参考值（速度/角速度）
        self._goal_height_ref = []  # 目标高度参考值
        self._swing_duration = []  # 摆动相持续时间（脚离地）
        self._stance_duration = []  # 支撑相持续时间（脚着地）
        self._total_duration = []  # 完整步态周期时间

        # 保存身体部位名称
        self._root_body_name = root_body  # 根节点名称
        self._lfoot_body_name = lfoot_body  # 左脚名称
        self._rfoot_body_name = rfoot_body  # 右脚名称
        self._head_body_name = head_body  # 头部名称

    def calc_reward(self, prev_torque, prev_action, action):
        """
        计算综合奖励函数
        
        奖励函数由多个分量组成，每个分量关注不同的行为目标：
        - 脚步接触力：确保正确的步态相位
        - 脚步速度：减少不必要的滑动
        - 身体高度：保持适当的高度
        - 前进速度：达到目标速度
        - 方向控制：保持正确朝向
        - 姿态保持：接近中立姿势
        
        Args:
            prev_torque: 上一时刻的关节力矩
            prev_action: 上一时刻的动作
            action: 当前动作
            
        Returns:
            dict: 包含各个奖励分量的字典
        """
        # 获取脚部状态信息（世界坐标系）
        self.l_foot_vel = self._client.get_lfoot_body_vel(frame=1)[0]  # 左脚线速度
        self.r_foot_vel = self._client.get_rfoot_body_vel(frame=1)[0]  # 右脚线速度
        self.l_foot_frc = self._client.get_lfoot_grf()  # 左脚地面反作用力
        self.r_foot_frc = self._client.get_rfoot_grf()  # 右脚地面反作用力 （判断是否着地）

        # 获取颈部位置（用于高度估计）
        neck_pos = self._client.get_object_xpos_by_name("NECK_Y_S", 'OBJ_BODY')

        # 获取头部和根节点位置（只取x,y平面坐标）
        head_pos = self._client.get_object_xpos_by_name(self._head_body_name, 'OBJ_BODY')[0:2]
        root_pos = self._client.get_object_xpos_by_name(self._root_body_name, 'OBJ_BODY')[0:2]

        # 获取当前关节位置
        current_pose = np.array(self._client.get_act_joint_positions())

        # 获取步态相位时钟函数
        # 这些函数根据当前相位返回脚的状态（支撑/摆动）
        r_frc = self.right_clock[0]  # 右脚接触力相位函数
        l_frc = self.left_clock[0]   # 左脚接触力相位函数
        r_vel = self.right_clock[1]  # 右脚速度相位函数
        l_vel = self.left_clock[1]   # 左脚速度相位函数

        # 站立模式下调整相位函数
        if self.mode == WalkModes.STANDING:
            # 双脚始终着地，施加力
            r_frc = (lambda _: 1)  # 常数函数，始终返回1（始终支撑）
            l_frc = (lambda _: 1)  # 常数函数，始终返回1
            # 速度相位惩罚函数，始终惩罚
            r_vel = (lambda _: -1)  # 常数函数，始终返回-1
            l_vel = (lambda _: -1)

        # 根据行走模式设置目标速度参考
        if self.mode == WalkModes.STANDING:
            self._goal_speed_ref = 0  # 站立：零前进速度
            yaw_vel_ref = 0  # 零偏航角速度
        if self.mode == WalkModes.INPLACE:
            self._goal_speed_ref = 0  # 原地踏步：零前进速度
            yaw_vel_ref = self.mode_ref  # 使用采样的偏航角速度
        if self.mode == WalkModes.FORWARD:
            self._goal_speed_ref = self.mode_ref  # 前进：使用采样的前进速度
            yaw_vel_ref = 0  # 零偏航角速度

        # 构建奖励字典
        # 注意：这里假设腿部关节在数组的前12个位置
        reward = dict(
            # 脚步接触力奖励 - 确保脚步与地面的接触符合步态相位
            foot_frc_score=0.2 * rewards._calc_foot_frc_clock_reward(self, l_frc, r_frc),
            
            # 脚步速度奖励 - 减少摆动相的脚步滑动
            foot_vel_score=0.2 * rewards._calc_foot_vel_clock_reward(self, l_vel, r_vel),

            # 脚步间距奖励 - 保持合适的步宽，防止绊倒
            feet_separation=0.2 * rewards._calc_feet_separation_reward(self),
            
            # 前进方向奖励 - 鼓励保持直线前进
            heading=0.2 * rewards._calc_heading_reward(self),

            # 高度奖励 - 基于颈部高度的奖励
            # 期望颈部高度和骨盆高度都在合理范围内
            height_error=0.050 * neck_pos[2] + 0.3 * (
                self._client.get_object_xpos_by_name("PELVIS_S", 'OBJ_BODY')[2] - 0.85),

            # 前进速度奖励 - 鼓励达到目标速度6.0 m/s
            vel_reward=0.2 + 0.2 * -abs(self._client.get_body_vel("PELVIS_S")[0][0] - 6.0),

            # 速度惩罚 - 惩罚侧向移动和旋转速度
            velocity_penalty=0.1 * rewards._calc_orient_reward(self, 'PELVIS_S'),

            # 朝向奖励 - 保持身体和脚部正确朝向
            orient_cost=0.1 * (2 * rewards._calc_orient_reward(self, self._root_body_name) +
                               4 * rewards._calc_body_orient_reward(self, self._rfoot_body_name) +
                               4 * rewards._calc_body_orient_reward(self, self._lfoot_body_name)) / 3,

            # 手臂摆动协调性奖励 - 鼓励手臂与腿部协调摆动
            coordination_cost=0.05 * rewards._calc_arm_swing_coordination(self)
        )

        # 注释掉的奖励分量（可能被暂时禁用）：
        # root_accel=0.030 * rewards._calc_root_accel_reward(self),  # 根身体加速度奖励（平滑性）
        # com_vel_error=0.350 * rewards._calc_fwd_vel_reward(self),  # 前进速度误差奖励
        # yaw_vel_error=0.120 * rewards._calc_yaw_vel_reward(self, yaw_vel_ref),  # 偏航角速度误差奖励
        # upper_body_reward=0.050 * np.exp(-10 * np.linalg.norm(head_pos - root_pos)),  # 上半身稳定性奖励
        # posture_error=0.040 * np.exp(-np.linalg.norm(self._neutral_pose[:12] - current_pose[:12])),  # 姿态误差奖励
        # waist_cost=-0.3 * sum(abs(np.array(self._client.get_act_joint_positions()[12:15]) - np.array([0, 0.15, 0]))),

        # 注意：各奖励分量的权重和接近1.0，确保奖励值在合理范围
        return reward

    def step(self):
        """
        执行一步任务更新
        主要功能：
        1. 推进步态相位
        2. 随机切换行走模式
        3. 更新地形高度场（如果启用）
        """
        # 相位递增 - 模拟步态周期的推进
        self._phase += 1
        
        # 相位循环：达到周期后归零，形成周期性步态
        if self._phase >= self._period:
            self._phase = 0

        # 检查是否处于双支撑期（双脚同时着地）
        in_double_support = (self.right_clock[0](self._phase) == 1 and 
                            self.left_clock[0](self._phase) == 1)

        # 随机在INPLACE和STANDING模式间切换
        # 仅在双支撑期切换，确保切换的稳定性
        if np.random.randint(100) == 0 and in_double_support:
            if self.mode == WalkModes.INPLACE:
                self.mode = WalkModes.STANDING  # 原地踏步 -> 站立
            elif self.mode == WalkModes.STANDING:
                self.mode = WalkModes.INPLACE  # 站立 -> 原地踏步
            self.mode_ref = self.mode.sample_ref()  # 重新采样参考值

        # 随机在INPLACE和FORWARD模式间切换
        # 频率较低（1/200），避免频繁切换影响学习
        if np.random.randint(200) == 0 and self.mode != WalkModes.STANDING:
            if self.mode == WalkModes.FORWARD:
                self.mode = WalkModes.INPLACE  # 前进 -> 原地踏步
            elif self.mode == WalkModes.INPLACE:
                self.mode = WalkModes.FORWARD  # 原地踏步 -> 前进
            self.mode_ref = self.mode.sample_ref()  # 重新采样参考值

        # 操作高度场（用于地形适应训练）
        if self.manip_hfield:
            # 随机改变地形高度
            if np.random.randint(200) == 0 and self.mode != WalkModes.STANDING:
                self._client.model.geom("hfield").pos[:] = [
                    np.random.uniform(-0.5, 0.5),   # x方向随机偏移
                    np.random.uniform(-0.5, 0.5),   # y方向随机偏移
                    np.random.uniform(-0.015, -0.035)  # z方向随机高度（地面起伏）
                ]
        return

    def done(self):
        """
        检查任务是否应该终止
        
        终止条件：
        1. 机器人高度过低（摔倒）
        2. 机器人高度过高（异常跳跃）
        3. 发生自碰撞（肢体互相穿透）
        
        Returns:
            bool: True表示任务应该终止，False表示继续
        """
        # 检查自碰撞
        contact_flag = self._client.check_self_collisions()
        
        # 获取关节位置状态
        qpos = self._client.get_qpos()

        # 定义终止条件字典
        terminate_conditions = {
            "qpos[2]_ll": (qpos[2] < 0.4),  # 高度过低（摔倒） - z坐标小于0.4m
            "qpos[2]_ul": (qpos[2] > 1.4),  # 高度过高（异常跳跃） - z坐标大于1.4m
            # "contact_flag": contact_flag,  # 发生自碰撞（暂时禁用）
        }

        # 如果任一终止条件为True，则任务结束
        done = True in terminate_conditions.values()
        return done

    def reset(self, iter_count=0):
        """
        重置任务状态（每个episode开始时调用）
        
        功能：
        1. 重置行走模式
        2. 重新创建步态相位函数
        3. 随机初始化相位
        
        Args:
            iter_count: 当前训练迭代次数（可用于自适应调整）
        """
        # 固定为前进模式，目标速度6.0 m/s
        # 这样可以专注于学习高速行走
        self.mode = WalkModes.FORWARD
        self.mode_ref = 6.0  # 固定高速目标

        # 创建步态相位时钟函数
        # 这些函数定义了一个完整的步态周期
        self.right_clock, self.left_clock = rewards.create_phase_reward(
            self._swing_duration,   # 摆动相持续时间
            self._stance_duration,  # 支撑相持续时间
            0.1,                    # 相位偏移（左右脚步行相位差）
            "grounded",             # 时钟类型（基于地面接触）
            1 / self._control_dt    # 控制频率（转换为每周期步数）
        )

        # 计算完整步态周期的控制步数
        # 一个完整周期包括左摆动和右摆动两个阶段
        self._period = np.floor(2 * self._total_duration * (1 / self._control_dt))
        
        # 在初始化时随机化相位
        # 这样每个episode可以从步态周期的不同位置开始
        self._phase = np.random.randint(0, self._period)