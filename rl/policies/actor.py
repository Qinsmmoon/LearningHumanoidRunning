import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import sqrt

from rl.policies.base import Net  # 基础网络类

# 对数标准差的范围限制
LOG_STD_HI = -1.5  # 对数标准差上限
LOG_STD_LO = -20  # 对数标准差下限


class Actor(Net):
    """演员（策略）网络基类"""

    def __init__(self):
        super(Actor, self).__init__()

    def forward(self):
        raise NotImplementedError

    def get_action(self):
        raise NotImplementedError


class Gaussian_FF_Actor(Actor):
    """高斯前馈策略网络 - 输出动作的概率分布
    
    该网络输出高斯分布（正态分布）的均值和标准差，用于表示随机策略。
    """

    def __init__(self, state_dim, action_dim, layers=(256, 256), env_name=None, 
                 nonlinearity=torch.nn.functional.relu, fixed_std=None, bounded=False, normc_init=True):
        """初始化高斯前馈策略网络
        
        参数:
            state_dim (int): 状态空间的维度
            action_dim (int): 动作空间的维度
            layers (tuple): 隐藏层的维度，默认为(256, 256)表示两个256维的隐藏层
            env_name (str, optional): 环境名称
            nonlinearity (callable): 激活函数，默认为ReLU
            fixed_std (float, optional): 如果提供固定值，则使用固定的标准差；否则学习标准差
            bounded (bool): 是否使用tanh将动作限制在[-1, 1]范围内
            normc_init (bool): 是否使用PPO论文中的权重初始化方法（归一化初始化）
        """
        super(Gaussian_FF_Actor, self).__init__()

        # ==================== 网络结构构建 ====================
        # 使用ModuleList来存储所有的隐藏层，这样PyTorch能正确识别需要训练的参数
        self.actor_layers = nn.ModuleList()
        
        # 添加输入层：从状态维度映射到第一个隐藏层的维度
        self.actor_layers += [nn.Linear(state_dim, layers[0])]
        
        # 添加中间隐藏层：连接相邻的隐藏层
        for i in range(len(layers) - 1):
            self.actor_layers += [nn.Linear(layers[i], layers[i + 1])]
        
        # 均值输出层：将最后一个隐藏层的输出映射到动作空间维度
        # 这个层输出的是高斯分布的均值向量
        self.means = nn.Linear(layers[-1], action_dim)

        # ==================== 标准差处理 ====================
        # 标准差决定了策略的探索程度（动作的离散程度）
        if fixed_std is None:
            # 可学习标准差：网络学习对数标准差（log std），因为标准差必须为正数
            # 使用对数形式可以避免约束问题，且exp()后自动为正
            self.log_stds = nn.Linear(layers[-1], action_dim)  # 对数标准差输出层
            self.learn_std = True  # 标志位：标准差是可学习的
        else:
            # 固定标准差：使用给定的常数值，不参与训练
            # 这样可以减少计算量，但可能降低策略的灵活性
            self.fixed_std = fixed_std  # 固定的标准差
            self.learn_std = False


        self.action = None                # 存储最新采样的动作，方便后续访问
        self.action_dim = action_dim      # 保存动作维度
        self.env_name = env_name          # 保存环境名称（某些环境可能需要特殊处理）
        self.nonlinearity = nonlinearity  # 保存激活函数


        self.obs_std = 1.0    # 观察值的标准差，用于归一化
        self.obs_mean = 0.0   # 观察值的均值，用于归一化

        # 是否使用PPO论文中的权重初始化方案
        # normc_init是OpenAI Baselines中常用的一种初始化方法
        self.normc_init = normc_init

        # 是否对均值输出使用tanh限制
        # 如果环境动作空间有界（如[-1, 1]），可以启用这个选项
        self.bounded = bounded

        # 初始化网络参数
        self.init_parameters()

    def init_parameters(self):
        """初始化网络参数
        
        根据normc_init标志选择不同的初始化策略。
        normc_init是PPO论文作者推荐的方法，可以稳定训练过程。
        """
        if self.normc_init:
            # 对整个网络应用归一化初始化函数
            # normc_fn函数会将权重矩阵按列进行归一化，使得初始输出分布更稳定
            self.apply(normc_fn)
            
            # 特别对均值输出层进行特殊处理：将权重缩小100倍
            # 这样做的目的是使初始策略接近均匀分布（输出接近0），避免初始策略过于确定
            # 在连续动作空间中，初始输出接近0有助于稳定探索
            self.means.weight.data.mul_(0.01)

    def _get_dist_params(self, state):
        """获取高斯分布的参数：均值和标准差
        
        这是网络的核心计算逻辑，将输入状态转换为概率分布的参数。
        
        参数:
            state (tensor): 输入状态，形状为 [batch_size, state_dim]
            
        返回:
            tuple: (mean, sd) 均值和标准差
                   mean形状为 [batch_size, action_dim]
                   sd形状为 [batch_size, action_dim] 或标量（固定标准差时）
        """
  
        # 将输入状态归一化到零均值单位方差，有助于训练稳定性
        state = (state - self.obs_mean) / self.obs_std

        x = state
        # 依次通过所有隐藏层，每层后应用激活函数
        for layer in self.actor_layers:
            x = self.nonlinearity(layer(x))
        
        # 计算均值：隐藏层输出通过线性层得到动作的均值
        mean = self.means(x)

        # ==================== 可选：对均值输出进行范围限制 ====================
        # 如果环境动作空间有界，使用tanh将输出限制在[-1, 1]范围内
        # 这对于像Mujoco这样的连续控制环境非常重要
        if self.bounded:
            mean = torch.tanh(mean)  # tanh输出范围[-1, 1]

        # ==================== 计算标准差 ====================
        if self.learn_std:
            # 可学习标准差：使用复杂的变换确保标准差在合理范围内
            # 公式: sd = exp(-2 + 0.5 * tanh(log_std_output))
            # 这样的设计确保标准差范围在 [exp(-2.5), exp(-1.5)] ≈ [0.08, 0.22]
            # 1. tanh将输出限制在[-1, 1]之间
            # 2. 乘以0.5后范围变成[-0.5, 0.5]
            # 3. 加上-2后范围变成[-2.5, -1.5]
            # 4. exp后得到[0.08, 0.22]的范围
            # 这个范围是经验上对大多数连续控制问题有效的探索程度
            log_std_output = self.log_stds(x)  # 原始对数标准差输出
            sd = (-2 + 0.5 * torch.tanh(log_std_output)).exp()
        else:
            # 固定标准差：直接使用预设值
            # 可以是标量或与动作维度相同的向量
            sd = self.fixed_std

        return mean, sd

    def forward(self, state, deterministic=True, anneal=1.0):
        """前向传播：根据策略选择动作
        
        参数:
            state (tensor): 当前状态
            deterministic (bool): 
                True - 使用确定性策略（直接输出均值），用于评估/测试
                False - 使用随机策略（从分布中采样），用于训练/探索
            anneal (float): 探索退火系数，范围为(0, 1]
                用于在训练过程中逐渐减小标准差，降低探索程度
                值越小，探索越少，动作越确定
        
        返回:
            tensor: 选择的动作，形状为 [batch_size, action_dim]
        """
        # 获取高斯分布的参数
        mu, sd = self._get_dist_params(state)
        
        # 应用退火：随着训练进行，逐步减小标准差以降低探索
        # 这是常见的探索策略：早期多探索，后期少探索
        sd *= anneal

        if not deterministic:
            # ==================== 随机策略模式 ====================
            # 从高斯分布中采样得到动作
            # 这是标准的随机策略梯度方法使用的探索方式
            # Normal(mu, sd) 创建一个高斯分布对象，sample()从中采样
            self.action = torch.distributions.Normal(mu, sd).sample()
        else:
            # ==================== 确定性策略模式 ====================
            # 直接使用均值作为动作，不进行探索
            # 这用于评估策略性能或部署时使用
            self.action = mu

        return self.action

    def get_action(self):
        """获取最近一次前向传播选择的动作
        
        返回:
            tensor: 最近采样的动作
        """
        return self.action

    def distribution(self, inputs):
        """返回动作的概率分布对象
        
        这是PPO算法中需要的关键方法，用于计算新旧策略的概率比。
        
        参数:
            inputs (tensor): 输入状态
            
        返回:
            torch.distributions.Normal: 高斯分布对象
                包含均值和标准差，可以计算:
                - log_prob(action): 动作的对数概率
                - entropy(): 分布的熵（用于探索程度估计）
        """
        # 获取分布的参数
        mu, sd = self._get_dist_params(inputs)
        
        # 返回PyTorch内置的正态分布对象
        return torch.distributions.Normal(mu, sd)



# 初始化函数（来自PPO论文的初始化方案）
# 注意：这个函数名与参数名相同曾经导致了一个严重的bug
# 因为在Python中 "if <function_name>" 会评估为True...
def normc_fn(m):
    """归一化列初始化 - 确保每列的L2范数为1"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)  # 从标准正态分布初始化
        # 按列的L2范数归一化
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)  # 偏置初始化为0