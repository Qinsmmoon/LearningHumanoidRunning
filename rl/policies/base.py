import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import sqrt

# 对神经网络层进行特定的权重初始化
def normc_fn(m):
  classname = m.__class__.__name__    # 获取层的类名
  if classname.find('Linear') != -1:  # 如果是Linear层
      m.weight.data.normal_(0, 1)     # 1. 用标准正态分布初始化权重
      m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True)) # 2. 对权重进行归一化
      if m.bias is not None:          # 3. 如果有偏置项，初始化为0
          m.bias.data.fill_(0)

# The base class for an actor. Includes functions for normalizing state (optional)
class Net(nn.Module):
  def __init__(self):
      super(Net, self).__init__()
      # Welford算法参数，用于在线计算状态的均值和方差
      self.welford_state_mean = torch.zeros(1)  # 当前均值
      self.welford_state_mean_diff = torch.ones(1) # 用于计算方差的累积量
      self.welford_state_n = 1 # 已处理的样本数量

      self.env_name = None # 环境名称

  def forward(self):
      raise NotImplementedError

  def normalize_state(self, state, update=True):
      # 将输入转换为Tensor
      state = torch.Tensor(state)

      # 首次使用时初始化统计量
      if self.welford_state_n == 1:
          self.welford_state_mean = torch.zeros(state.size(-1))
          self.welford_state_mean_diff = torch.ones(state.size(-1))

      if update:
          if len(state.size()) == 1:  # 单个状态向量
              state_old = self.welford_state_mean
              # 更新均值: new_mean = old_mean + (x - old_mean)/n
              self.welford_state_mean += (state - state_old) / self.welford_state_n
              # 更新方差累积量
              self.welford_state_mean_diff += (state - state_old) * (state - state_old)
              self.welford_state_n += 1
              
          elif len(state.size()) == 2:  # 批量数据
              print("NORMALIZING 2D TENSOR (this should not be happening)")
              # 对批次中的每个样本进行更新
              for state_n in state:
                  state_old = self.welford_state_mean
                  self.welford_state_mean += (state_n - state_old) / self.welford_state_n
                  self.welford_state_mean_diff += (state_n - state_old) * (state_n - state_old)
                  self.welford_state_n += 1
                  
          elif len(state.size()) == 3:  # 序列数据批次
              print("NORMALIZING 3D TENSOR (this really should not be happening)")
              # 对序列中的每个样本进行更新
              for state_t in state:  # 遍历序列
                  for state_n in state_t:  # 遍历批次
                      state_old = self.welford_state_mean
                      self.welford_state_mean += (state_n - state_old) / self.welford_state_n
                      self.welford_state_mean_diff += (state_n - state_old) * (state_n - state_old)
                      self.welford_state_n += 1
      
      # 返回归一化后的状态: (state - mean) / std
      return (state - self.welford_state_mean) / sqrt(self.welford_state_mean_diff / self.welford_state_n)

  def copy_normalizer_stats(self, net):
      """从另一个网络复制归一化统计信息"""
      self.welford_state_mean = net.welford_state_mean
      self.welford_state_mean_diff = net.welford_state_mean_diff
      self.welford_state_n = net.welford_state_n

  def initialize_parameters(self):
      """初始化网络参数"""
      self.apply(normc_fn)