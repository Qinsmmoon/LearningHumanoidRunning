import os  
import numpy as np  
import mujoco  
import mujoco_viewer  

DEFAULT_SIZE = 500  

class MujocoEnv():
    """所有MuJoCo环境的基类（父类）。
    这个类提供了MuJoCo仿真的基本功能，具体的机器人环境应该继承这个类。
    """

    def __init__(self, model_path, sim_dt, control_dt):
        """初始化MuJoCo环境
        
        参数:
            model_path: MuJoCo模型XML文件的完整路径
            sim_dt: 物理仿真时间步长（物理引擎每次计算的时间间隔）
            control_dt: 控制周期（算法给机器人发送指令的时间间隔）
        """
        # 检查模型路径是否为绝对路径（以"/"开头）
        if model_path.startswith("/"):
            fullpath = model_path  # 如果是绝对路径，直接使用
        else:
            # 如果不是绝对路径，抛出异常
            raise Exception("请提供机器人描述包的完整路径。")
        
        # 检查模型文件是否存在
        if not os.path.exists(fullpath):
            raise IOError("文件 %s 不存在" % fullpath)

        # 加载MuJoCo模型并创建对应的数据对象
        self.model = mujoco.MjModel.from_xml_path(fullpath)  # 从XML文件加载模型（包含机器人的物理属性、关节、几何形状等）
        self.data = mujoco.MjData(self.model)  # 创建与模型对应的数据对象（包含位置、速度、力等状态信息）
        self.viewer = None  # 初始化查看器为None，后面按需创建

        # 计算帧跳过次数（控制周期/仿真步长），并设置仿真时间步长
        self.frame_skip = (control_dt/sim_dt)  # 你的每个控制指令，物理引擎要计算多少次
        self.model.opt.timestep = sim_dt  # 设置物理仿真的时间步长（越小越精确，但计算越慢）

        # 保存初始关节位置和速度，用于重置环境时恢复
        self.init_qpos = self.data.qpos.ravel().copy()  # ravel()将多维数组展平为一维，copy()创建副本
        self.init_qvel = self.data.qvel.ravel().copy()  # 保存初始速度

    # 需要子类实现的方法（抽象方法）：
    # ----------------------------

    def reset_model(self):
        """
        重置机器人的自由度状态（关节位置qpos和速度qvel）。
        这个方法必须在每个子类中具体实现。
        例如：将机器人放回初始姿态，或添加随机扰动。
        """
        raise NotImplementedError  # 如果子类没有实现这个方法，调用时会报错

    def viewer_setup(self):
        """
        当查看器初始化时会调用这个方法。
        可以根据需要调整相机位置、视角等。
        这是一个可选实现的方法（子类可以重写它）。
        """
        self.viewer.cam.trackbodyid = 1  # 追踪第1个身体（让相机跟随这个身体）
        self.viewer.cam.distance = self.model.stat.extent * 2.0  # 设置相机距离（模型范围的2倍）
        self.viewer.cam.lookat[2] = 1.5  # 设置相机注视点的Z坐标（高度）
        self.viewer.cam.lookat[0] = 2.0  # 设置相机注视点的X坐标
        self.viewer.cam.elevation = -20  # 设置相机仰角（-20度，从上往下看）
        self.viewer.vopt.geomgroup[0] = 1  # 显示第0组几何体（通常是主要可见物体）
        self.viewer._render_every_frame = True  # 设置为每帧都渲染（即使仿真暂停）

    def viewer_is_paused(self):
        """检查查看器是否处于暂停状态"""
        return self.viewer._paused  # 返回查看器的暂停状态

    # 核心功能方法：
    # -----------------------------

    def reset(self):
        """重置整个环境到初始状态"""
        mujoco.mj_resetData(self.model, self.data)  # 调用MuJoCo函数重置所有数据到零状态
        ob = self.reset_model()  # 调用子类实现的reset_model来设置具体的初始状态
        return ob  # 返回初始观测值

    def set_state(self, qpos, qvel):
        """手动设置机器人的关节状态
        
        参数:
            qpos: 关节位置数组
            qvel: 关节速度数组
        """
        # 断言检查：确保输入的位置和速度维度与模型匹配
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = qpos  # 设置所有关节位置
        self.data.qvel[:] = qvel  # 设置所有关节速度
        mujoco.mj_forward(self.model, self.data)  # 前向动力学计算（更新加速度、传感器等）

    @property
    def dt(self):
        """属性方法，返回实际的控制周期
        
        控制周期 = 仿真步长 × 帧跳过次数
        这代表你的算法每多久执行一次
        """
        return self.model.opt.timestep * self.frame_skip

    def render(self):
        """渲染当前画面（显示可视化窗口）"""
        if self.viewer is None:  # 如果查看器还未创建
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)  # 创建新的查看器
            self.viewer_setup()  # 调用查看器设置方法（调整相机等）
        self.viewer.render()  # 渲染一帧画面

    def uploadGPU(self, hfieldid=None, meshid=None, texid=None):
        """将资源上传到GPU内存，用于可视化
        
        参数（都是可选的）:
            hfieldid: 高度场ID
            meshid: 网格ID
            texid: 纹理ID
        """
        # 如果提供了高度场ID，上传高度场到GPU
        if hfieldid is not None:
            mujoco.mjr_uploadHField(self.model, self.viewer.ctx, hfieldid)
        # 如果提供了网格ID，上传网格到GPU
        if meshid is not None:
            mujoco.mjr_uploadMesh(self.model, self.viewer.ctx, meshid)
        # 如果提供了纹理ID，上传纹理到GPU
        if texid is not None:
            mujoco.mjr_uploadTexture(self.model, self.viewer.ctx, texid)

    def close(self):
        """关闭查看器，释放资源"""
        if self.viewer is not None:  # 如果查看器存在
            self.viewer.close()  # 关闭查看器窗口
            self.viewer = None  # 将查看器设为None