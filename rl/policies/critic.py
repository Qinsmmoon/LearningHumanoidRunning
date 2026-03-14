import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.policies.base import Net, normc_fn 


# 评论家（值函数）网络基类，包含奖励和状态归一化功能
class Critic(Net):
    """
    评论家网络基类
    继承自Net基类，包含奖励归一化的基础功能
    奖励归一化可以稳定训练过程，防止梯度爆炸/消失
    """
    
    def __init__(self):
        """初始化评论家基类"""
        super(Critic, self).__init__()

        # Welford算法参数，用于在线计算奖励的均值和方差
        # Welford算法是一种数值稳定的在线方差计算方法
        self.welford_reward_mean = 0.0       # 奖励均值，初始为0
        self.welford_reward_mean_diff = 1.0  # 奖励方差相关量（M2），初始为1避免除零
        self.welford_reward_n = 1            # 样本计数，初始为1避免除零

    def forward(self):
        """
        前向传播函数，需要在子类中实现
        抛出NotImplementedError强制子类实现该方法
        """
        raise NotImplementedError

    def normalize_reward(self, r, update=True):
        """
        使用Welford算法归一化奖励
        
        Welford算法可以在线计算均值和方差，适用于流式数据
        相比传统方法，它具有数值稳定性好、计算效率高的优点
        
        算法原理：
        1. 更新计数：n += 1
        2. 计算与旧均值的差：delta = r - mean
        3. 更新均值：mean += delta / n
        4. 计算与新均值的差：delta2 = r - new_mean
        5. 更新M2：M2 += delta * delta2
        6. 方差 = M2 / n，标准差 = sqrt(M2 / n)
        
        参数:
            r: 输入的奖励值，可以是标量或批次数据
            update: 是否更新统计量，False时只返回归一化结果
        """
        if update:
            # 处理不同维度的输入
            if len(r.size()) == 1:  
                # 一维张量（单个奖励值）
                # 保存旧均值用于计算
                r_old = self.welford_reward_mean
                
                # 更新均值：使用累积移动平均公式
                self.welford_reward_mean += (r - r_old) / self.welford_reward_n
                
                # 更新M2（方差相关量）：使用Welford算法
                self.welford_reward_mean_diff += (r - r_old) * (r - r_old)
                
                # 增加样本计数
                self.welford_reward_n += 1
                
            elif len(r.size()) == 2:  
                # 二维张量（奖励批次），维度为[batch_size, 1]
                # 遍历批次中的每个奖励值
                for r_n in r:
                    # 对每个样本分别更新统计量
                    r_old = self.welford_reward_mean
                    self.welford_reward_mean += (r_n - r_old) / self.welford_reward_n
                    self.welford_reward_mean_diff += (r_n - r_old) * (r_n - r_old)
                    self.welford_reward_n += 1
            else:
                # 不支持的输入维度
                raise NotImplementedError("只支持1维或2维输入")

        # 返回标准化后的奖励：(r - mean) / std
        # 使用当前统计量进行Z-Score标准化
        # 标准差 = sqrt(M2 / n)，添加微小常数防止除零
        return (r - self.welford_reward_mean) / (torch.sqrt(self.welford_reward_mean_diff / self.welford_reward_n) + 1e-8)


class FF_V(Critic):
    """
    前馈状态值函数网络 - 估计V(s)
    使用多层前馈神经网络近似状态值函数
    输入：状态s，输出：状态值V(s)
    """
    
    def __init__(self, state_dim, layers=(256, 256), env_name='NOT SET', 
                 nonlinearity=torch.nn.functional.relu,
                 normc_init=True, obs_std=None, obs_mean=None):
        """
        初始化前馈值函数网络
        
        参数:
            state_dim: 状态空间的维度
            layers: 隐藏层的维度元组，如(256, 256)表示两个256维的隐藏层
            env_name: 环境名称，用于标识
            nonlinearity: 激活函数，默认使用ReLU
            normc_init: 是否使用列归一化初始化
            obs_std: 观测值的标准差，用于状态归一化
            obs_mean: 观测值的均值，用于状态归一化
        """
        # 调用父类Critic的初始化
        super(FF_V, self).__init__()

        # 构建多层前馈网络
        self.critic_layers = nn.ModuleList()
        
        # 输入层：从状态维度到第一个隐藏层维度
        self.critic_layers += [nn.Linear(state_dim, layers[0])]
        
        # 隐藏层：连接各个隐藏层
        for i in range(len(layers) - 1):
            self.critic_layers += [nn.Linear(layers[i], layers[i + 1])]
        
        # 输出层：从最后一个隐藏层到1维输出（状态值）
        self.network_out = nn.Linear(layers[-1], 1)

        # 保存环境名称，便于日志和调试
        self.env_name = env_name

        # 激活函数，默认使用ReLU
        self.nonlinearity = nonlinearity

        # 观察值归一化参数
        # 用于在评估时将输入状态归一化到标准范围
        self.obs_std = obs_std      # 观测标准差
        self.obs_mean = obs_mean    # 观测均值

        # PPO论文实验中使用的权重初始化方案
        # normc_init初始化可以使网络输出具有单位方差
        self.normc_init = normc_init

        # 初始化网络参数
        self.init_parameters()
        
        # 设置为训练模式（影响BatchNorm和Dropout的行为）
        self.train()

    def init_parameters(self):
        """
        初始化网络参数
        如果启用normc_init，对所有权重矩阵进行列归一化初始化
        这种初始化方法可以使每一层的输出具有相近的尺度
        """
        if self.normc_init:
            print("使用列归一化初始化网络参数")
            # 递归地对所有子模块应用normc_fn初始化函数
            self.apply(normc_fn)

    def forward(self, inputs):
        """
        前向传播，估计状态值V(s)
        
        参数:
            inputs: 输入状态张量，维度为[batch_size, state_dim]
            
        返回:
            value: 状态值估计，维度为[batch_size, 1]
        """
        # 在非训练模式下进行观察值归一化（评估模式）
        # 训练时使用当前batch的统计量，评估时使用全局统计量
        if self.training == False:
            # 使用预计算的均值和标准差进行归一化
            # 这有助于提高模型在测试时的泛化能力
            inputs = (inputs - self.obs_mean) / self.obs_std

        x = inputs
        # 前向传播通过所有隐藏层
        for l in self.critic_layers:
            # 每个隐藏层先进行线性变换，然后应用激活函数
            x = self.nonlinearity(l(x))
        
        # 输出层得到最终的状态值
        value = self.network_out(x)

        return value

    def act(self, inputs):
        """
        act方法（已弃用）
        为了兼容性保留，直接调用forward
        """
        return self(inputs)


# 类型别名
GaussianMLP_Critic = FF_V