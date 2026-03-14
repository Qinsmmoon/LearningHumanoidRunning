from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import kl_divergence
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import ray
from rl.envs import WrapEnv


class PPOBuffer:
    """
    PPO经验回放缓冲区，用于存储轨迹数据并计算策略和值函数更新所需的回报这个容器在设计上没有针对内存分配速度进行优化，
    因为在策略梯度方法中这几乎从来不是性能瓶颈

    另一方面，经验缓冲区是策略梯度实现中经常出现差一错误和其他bug的地方，因此这段代码优先考虑清晰性和可读性，
    而不是（非常）轻微的速度优势（过早优化是万恶之源）
    """
    def __init__(self, gamma=0.99, lam=0.95, use_gae=False):
        """
        初始化PPO缓冲区
        
        参数:
            gamma: 折扣因子，用于计算未来奖励的现值
            lam: GAE(广义优势估计)参数，平衡方差和偏差
            use_gae: 是否使用GAE计算优势函数
        """
        # 存储轨迹数据
        self.states  = []   # 状态序列：记录每个时间步的状态
        self.actions = []   # 动作序列：记录每个时间步执行的动作
        self.rewards = []   # 奖励序列：记录每个时间步获得的即时奖励
        self.values  = []   # 值估计：Critic网络对每个状态的价值估计
        self.returns = []   # 回报序列：折扣累积奖励，用于训练Critic

        # 用于记录和日志的统计信息
        self.ep_returns = [] # 每个episode的总回报（未折扣的累积奖励）
        self.ep_lens    = [] # 每个episode的长度（时间步数）

        # 折扣因子和GAE参数
        self.gamma, self.lam = gamma, lam
        self.use_gae = use_gae

        # 指针和轨迹索引
        self.ptr = 0           # 当前缓冲区位置，指向下一个要存储的位置
        self.traj_idx = [0]    # 轨迹开始位置的索引列表，用于标记不同episode的分界

    def __len__(self):
        """返回缓冲区中存储的总时间步数"""
        return len(self.states)

    def storage_size(self):
        """返回缓冲区大小，与__len__相同"""
        return len(self.states)

    def store(self, state, action, reward, value):
        """
        将一个时间步的智能体-环境交互数据添加到缓冲区
        
        参数:
            state: 当前状态 [1, state_dim]
            action: 执行的动作 [1, action_dim]
            reward: 获得的奖励 [1]
            value: 状态值估计 [1]
        """
        # 移除批次维度，存储为1D张量
        self.states  += [state.squeeze(0)]   # [state_dim]
        self.actions += [action.squeeze(0)]  # [action_dim]
        self.rewards += [reward.squeeze(0)]  # 标量
        self.values  += [value.squeeze(0)]   # 标量

        self.ptr += 1  # 移动指针

    def finish_path(self, last_val=None):
        """完成一个轨迹的处理，计算每个时间步的折扣回报"""
        # 记录当前轨迹结束位置
        self.traj_idx += [self.ptr]
        
        # 获取当前轨迹的奖励序列
        rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1]]

        returns = []

        if self.use_gae:
            # 使用GAE计算优势函数和回报
            # TODO: 实现GAE逻辑
            pass
        else:
            # 使用标准的折扣回报计算
            # 使用最后一个状态的值函数估计作为bootstrap值（用于非终止状态）
            R = last_val.squeeze(0).copy()  # 避免修改原始数据
            
            # 反向计算折扣回报（从最后一个时间步向前计算）
            # R_t = r_t + γ * R_{t+1}
            for reward in reversed(rewards):
                R = self.gamma * R + reward  # 贝尔曼方程
                returns.insert(0, R)  # 在开头插入，保持时间顺序
                # 注意：在列表开头插入是O(k^2)操作，对于长序列可能成为瓶颈

        self.returns += returns

        # 记录episode统计信息
        self.ep_returns += [np.sum(rewards)]  # 总回报（未折扣）
        self.ep_lens    += [len(rewards)]     # 轨迹长度

    def get(self):
        """获取所有缓冲区数据"""
        return(
            self.states,
            self.actions,
            self.returns,
            self.values
        )


class PPO:
    """PPO算法主类，实现近端策略优化算法"""
    def __init__(self, args, save_path):
        """
        初始化PPO训练器
        
        参数:
            args: 包含所有超参数的字典
            save_path: 模型保存路径
        """
        # 核心超参数设置
        self.gamma          = args['gamma']           # 折扣因子
        self.lam            = args['lam']             # GAE参数
        self.lr             = args['lr']              # 学习率
        self.eps            = args['eps']             # Adam优化器的epsilon，防止除零
        self.ent_coeff      = args['entropy_coeff']   # 熵系数，鼓励探索
        self.clip           = args['clip']            # PPO裁剪参数
        self.minibatch_size = args['minibatch_size']  # 小批次大小
        self.epochs         = args['epochs']          # 每个批次数据的训练轮数
        self.max_traj_len   = args['max_traj_len']    # 最大轨迹长度（截断长度）
        self.use_gae        = args['use_gae']         # 是否使用GAE
        self.n_proc         = args['num_procs']       # 并行环境数量
        self.grad_clip      = args['max_grad_norm']   # 梯度裁剪阈值
        self.mirror_coeff   = args['mirror_coeff']    # 镜像对称损失系数
        self.eval_freq      = args['eval_freq']       # 评估频率


        # 批次大小 = 并行环境数 × 每个环境的轨迹长度
        self.batch_size = self.n_proc * self.max_traj_len

        # 值函数损失系数（PPO论文中的c1）
        self.vf_coeff = 0.5

        # 训练统计
        self.total_steps = 0      # 总采样步数
        self.highest_reward = -1  # 最高评估奖励
        self.limit_cores = 0      # 是否限制CPU核心数

        # 训练迭代计数器
        self.iteration_count = 0

        # 保存路径和日志文件
        self.save_path = save_path
        self.eval_fn = os.path.join(self.save_path, 'eval.txt')
        with open(self.eval_fn, 'w') as out:
            out.write("test_ep_returns,test_ep_lens\n")

        self.train_fn = os.path.join(self.save_path, 'train.txt')
        with open(self.train_fn, 'w') as out:
            out.write("ep_returns,ep_lens\n")

    def save(self, policy, critic, suffix=""):
        """保存策略和值函数模型"""
        try:
            os.makedirs(self.save_path)
        except OSError:
            pass
        filetype = ".pt" # pytorch模型
        torch.save(policy, os.path.join(self.save_path, "actor" + suffix + filetype))
        torch.save(critic, os.path.join(self.save_path, "critic" + suffix + filetype))

    @ray.remote
    @torch.no_grad()
    def sample(self, env_fn, policy, critic, max_steps, max_traj_len, deterministic=False, anneal=1.0, term_thresh=0):
        """
        采样max_steps个总时间步，如果轨迹超过max_traj_len个时间步则截断
        
        @ray.remote: 将此函数标记为Ray远程任务，可在不同进程中并行执行
        @torch.no_grad(): 禁用梯度计算，节省内存和计算
        
        参数:
            env_fn: 环境创建函数
            policy: 策略网络（Actor）
            critic: 值函数网络（Critic）
            max_steps: 最大采样步数
            max_traj_len: 最大轨迹长度
            deterministic: 是否使用确定性策略（评估时用）
            anneal: 退火系数
            term_thresh: 提前终止阈值
            
        返回:
            memory: 填充好的PPO缓冲区
        """
        # 限制PyTorch使用单核心，避免与Ray worker冲突
        # Ray每个worker单独进程，如果PyTorch使用多线程会导致CPU资源竞争
        torch.set_num_threads(1)

        # 包装环境，添加额外功能（如monitoring）
        env = WrapEnv(env_fn)  
        env.robot.iteration_count = self.iteration_count  # 传递迭代计数

        memory = PPOBuffer(self.gamma, self.lam)
        memory_full = False

        while not memory_full:
            # 重置环境，开始新episode
            state = torch.Tensor(env.reset())
            done = False
            traj_len = 0

            # 运行一个完整的episode或直到达到最大步数
            while not done and traj_len < max_traj_len:
                # 选择动作 - 策略网络输出动作（包含探索噪声）
                action = policy(state, deterministic=deterministic, anneal=anneal)
                
                # 估计状态值 - 用于后续优势函数计算
                value = critic(state)

                # 执行动作
                next_state, reward, done, _ = env.step(action.numpy())

                # 存储经验
                memory.store(state.numpy(), action.numpy(), reward, value.numpy())
                
                # 检查是否达到最大步数
                memory_full = (len(memory) == max_steps)

                # 更新状态
                state = torch.Tensor(next_state)
                traj_len += 1

                if memory_full:
                    break

            # 处理轨迹结束，计算回报
            # 对最后一个状态进行值估计（用于bootstrap）
            value = critic(state)
            
            # 如果轨迹没有自然结束（被截断），使用值函数进行bootstrap
            # 如果自然结束，last_val应为0
            memory.finish_path(last_val=(not done) * value.numpy())

        return memory

    def sample_parallel(self, env_fn, policy, critic, min_steps, max_traj_len, deterministic=False, anneal=1.0, term_thresh=0):
        """
        并行采样多个环境的经验
        
        使用Ray将采样任务分布到多个CPU核心上，大幅提高数据收集效率
        
        参数:
            min_steps: 最小总采样步数，会被均匀分配到各worker
        """
        worker = self.sample
        # 计算每个worker需要采样的步数
        steps_per_worker = min_steps // self.n_proc
        args = (self, env_fn, policy, critic, steps_per_worker, max_traj_len, deterministic, anneal, term_thresh)

        # 创建工作进程池，每个进程收集steps_per_worker步数据
        workers = [worker.remote(*args) for _ in range(self.n_proc)]
        
        # 等待所有worker完成并获取结果
        result = ray.get(workers)

        # 合并所有worker的缓冲区
        def merge(buffers):
            """
            合并多个PPO缓冲区
            
            参数:
                buffers: 多个PPOBuffer对象列表
            返回:
                合并后的PPOBuffer
            """
            merged = PPOBuffer(self.gamma, self.lam)
            for buf in buffers:
                offset = len(merged)  # 当前合并后的大小，用于调整轨迹索引
                
                # 合并数据列表
                merged.states  += buf.states
                merged.actions += buf.actions
                merged.rewards += buf.rewards
                merged.values  += buf.values
                merged.returns += buf.returns

                # 合并统计信息
                merged.ep_returns += buf.ep_returns
                merged.ep_lens    += buf.ep_lens

                # 调整轨迹索引：每个新缓冲区的索引需要加上当前总长度
                merged.traj_idx += [offset + i for i in buf.traj_idx[1:]]
                
                # 更新指针
                merged.ptr += buf.ptr

            return merged

        total_buf = merge(result)

        return total_buf

    def update_policy(self, obs_batch, action_batch, return_batch, advantage_batch, mask, mirror_observation=None, mirror_action=None):
        """
        更新策略网络和值函数网络
        
        这是PPO算法的核心，实现裁剪的代理目标函数
        
        参数:
            obs_batch: 观测批次 [batch_size, obs_dim]
            action_batch: 动作批次 [batch_size, action_dim]
            return_batch: 回报批次 [batch_size, 1] - 用于训练critic的目标
            advantage_batch: 优势函数批次 [batch_size, 1] - 用于训练actor
            mask: 填充掩码，用于处理变长序列 [batch_size, 1] 或 [seq_len, batch_size, 1]
            mirror_observation: 镜像观测函数（用于对称系统）
            mirror_action: 镜像动作函数
            
        返回:
            各种损失和统计信息的元组
        """
        policy = self.policy
        critic = self.critic
        old_policy = self.old_policy

        # 计算当前值函数估计
        values = critic(obs_batch)  # [batch_size, 1]
        
        # 计算当前策略的动作分布和对数概率
        pdf = policy.distribution(obs_batch)  # 动作分布（高斯分布）
        log_probs = pdf.log_prob(action_batch).sum(-1, keepdim=True)  # [batch_size, 1]

        # 计算旧策略的对数概率（用于重要性采样）
        old_pdf = old_policy.distribution(obs_batch)
        old_log_probs = old_pdf.log_prob(action_batch).sum(-1, keepdim=True)

        # 重要性采样比率: π_new(a|s) / π_old(a|s)
        # 通过对数概率差取指数得到
        ratio = (log_probs - old_log_probs).exp()  # [batch_size, 1]

        # 裁剪的替代损失 - PPO核心公式
        # 未裁剪的损失: r(θ) * A
        cpi_loss = ratio * advantage_batch * mask  # CPI: Conservative Policy Iteration
        
        # 裁剪后的损失: clip(r(θ), 1-ε, 1+ε) * A
        clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch * mask
        
        # PPO目标函数: min(r(θ)A, clip(r(θ),1-ε,1+ε)A)
        # 取最小值并取负号（因为我们要最小化损失）
        actor_loss = -torch.min(cpi_loss, clip_loss).mean()

        # 计算裁剪比例 - 用于监控有多少比例的策略更新被裁剪
        clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip).float()).item()

        # 值函数损失 - 使用MSE估计值与目标回报的差异
        # 这是PPO论文中的L^VF项
        critic_loss = self.vf_coeff * F.mse_loss(return_batch, values)

        # 熵惩罚 - 鼓励策略保持探索性
        # 熵越大，动作分布越均匀，探索性越强
        entropy_penalty = -(pdf.entropy() * mask).mean()

        # 镜像对称损失（用于人形机器人等对称系统）
        # 例如：左右对称的动作应该产生对称的效果
        if mirror_observation is not None and mirror_action is not None:
            # 当前状态下的确定性动作
            deterministic_actions = policy(obs_batch)
            
            # 镜像状态下的动作
            mir_obs = mirror_observation(obs_batch)
            mirror_actions = policy(mir_obs)
            
            # 对镜像动作进行镜像变换
            mirror_actions = mirror_action(mirror_actions)
            
            # 镜像一致性损失：原始状态的动作应与镜像状态镜像后的动作一致
            mirror_loss = (deterministic_actions - mirror_actions).pow(2).mean()
        else:
            mirror_loss = torch.Tensor([0])

        # 计算近似的反向KL散度，用于早停
        # 参考Schulman博客: http://joschu.net/blog/kl-approx.html
        # KL散度衡量新旧策略的差异，过大表示策略变化太剧烈
        with torch.no_grad():
            log_ratio = log_probs - old_log_probs
            # 近似KL散度: E[(r-1) - log r]，其中r = π_new/π_old
            approx_kl_div = torch.mean((ratio - 1) - log_ratio)

        return (
            actor_loss,
            entropy_penalty,
            critic_loss,
            approx_kl_div,
            mirror_loss,
            clip_fraction,
        )

    def train(self,
              env_fn,
              policy,
              critic,
              n_itr,
              anneal_rate=1.0):
        """
        主训练循环
        
        参数:
            env_fn: 环境创建函数
            policy: 策略网络（Actor）
            critic: 值函数网络（Critic）
            n_itr: 训练迭代次数
            anneal_rate: 探索退火率
        """

        # 初始化旧策略（用于重要性采样）深拷贝确保旧策略参数不参与梯度更新
        self.old_policy = deepcopy(policy)
        self.policy = policy
        self.critic = critic

        # 优化器
        self.actor_optimizer = optim.Adam(policy.parameters(), lr=self.lr, eps=self.eps)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=self.lr, eps=self.eps)

        train_start_time = time.time()

        # 镜像函数（如果环境支持）- 用于对称性正则化
        obs_mirr, act_mirr = None, None
        if hasattr(env_fn(), 'mirror_observation'):
            obs_mirr = env_fn().mirror_clock_observation

        if hasattr(env_fn(), 'mirror_action'):
            act_mirr = env_fn().mirror_action

        # 课程学习参数
        curr_anneal = 1.0    # 当前探索退火系数，1.0表示最大探索
        curr_thresh = 0      # 当前终止阈值
        start_itr = 0        # 开始迭代（用于课程学习）
        ep_counter = 0       # episode计数器
        do_term = False      # 是否启用提前终止

        # 评估统计
        test_ep_lens = []
        test_ep_returns = []

        # 主训练循环
        for itr in range(n_itr):
            print("********** Iteration {} ************".format(itr))

            # 设置迭代计数
            self.iteration_count = itr

            # ========== 采样阶段 ==========
            sample_start_time = time.time()
            
            # 自适应退火：当性能足够好时减少探索
            # 如果最高奖励超过最大轨迹长度的2/3，且当前退火系数>0.5，则减少探索
            if self.highest_reward > (2/3)*self.max_traj_len and curr_anneal > 0.5:
                curr_anneal *= anneal_rate  # 乘以退火率（通常<1）
                
            # 自适应提前终止阈值 - 用于早期终止表现差的轨迹
            if do_term and curr_thresh < 0.35:
                curr_thresh = .1 * 1.0006**(itr-start_itr)

            # 并行采样经验
            batch = self.sample_parallel(env_fn, self.policy, self.critic, 
                                         self.batch_size, self.max_traj_len, 
                                         anneal=curr_anneal, term_thresh=curr_thresh)
            
            # 将numpy数组转换为torch张量
            observations, actions, returns, values = map(torch.Tensor, batch.get())

            num_samples = batch.storage_size()
            elapsed = time.time() - sample_start_time
            print("Sampling took {:.2f}s for {} steps.".format(elapsed, num_samples))

            # ========== 优势函数计算和标准化 ==========
            # 优势函数 = 实际回报 - 值函数估计
            advantages = returns - values
            
            # 标准化优势函数 - 有助于稳定训练
            # 减去均值，除以标准差
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

            # 设置小批次大小
            minibatch_size = self.minibatch_size or num_samples
            self.total_steps += num_samples

            # ========== 策略更新阶段 ==========
            # 将当前策略复制到旧策略
            self.old_policy.load_state_dict(policy.state_dict())

            optimizer_start_time = time.time()
            
            # 多个epoch训练同一个batch的数据
            for epoch in range(self.epochs):
                actor_losses = []
                entropies = []
                critic_losses = []
                kls = []
                mirror_losses = []
                clip_fractions = []

                # 创建小批次采样器
                # 前馈网络：随机采样（打破时间相关性）
                random_indices = SubsetRandomSampler(range(num_samples))
                sampler = BatchSampler(random_indices, minibatch_size, drop_last=True)

                # 小批次训练
                for indices in sampler:

                    # 前馈网络：直接索引
                    obs_batch       = observations[indices]
                    action_batch    = actions[indices]
                    return_batch    = returns[indices]
                    advantage_batch = advantages[indices]
                    mask            = 1  # 无填充，全有效

                    # 计算损失 - 调用update_policy方法
                    scalars = self.update_policy(obs_batch, action_batch, return_batch, 
                                                advantage_batch, mask, 
                                                mirror_observation=obs_mirr, 
                                                mirror_action=act_mirr)
                    actor_loss, entropy_penalty, critic_loss, approx_kl_div, mirror_loss, clip_fraction = scalars

                    # 记录损失
                    actor_losses.append(actor_loss.item())
                    entropies.append(entropy_penalty.item())
                    critic_losses.append(critic_loss.item())
                    kls.append(approx_kl_div.item())
                    mirror_losses.append(mirror_loss.item())
                    clip_fractions.append(clip_fraction)

                    # ===== 策略网络更新 =====
                    self.actor_optimizer.zero_grad()
                    
                    # 总Actor损失 = PPO损失 + 镜像损失 + 熵损失
                    total_actor_loss = (actor_loss + 
                                       self.mirror_coeff * mirror_loss + 
                                       self.ent_coeff * entropy_penalty)
                    total_actor_loss.backward()

                    # 梯度裁剪 - 防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip)
                    self.actor_optimizer.step()

                    # ===== 值函数网络更新 =====
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), self.grad_clip)
                    self.critic_optimizer.step()

            elapsed = time.time() - optimizer_start_time
            print("Optimizer took: {:.2f}s".format(elapsed))

            # ========== 课程学习逻辑 ==========
            # 当平均episode长度足够长时开始计数
            if np.mean(batch.ep_lens) >= self.max_traj_len * 0.75:
                ep_counter += 1
                
            # 如果连续50个episode长度达标，启用提前终止
            if not do_term and ep_counter > 50:
                do_term = True
                start_itr = itr

            # ========== 打印训练统计 ==========
            sys.stdout.write("-" * 37 + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Return (batch)', "%8.5g" % np.mean(batch.ep_returns)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Mean Eplen', "%8.5g" % np.mean(batch.ep_lens)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Actor loss', "%8.3g" % np.mean(actor_losses)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Critic loss', "%8.3g" % np.mean(critic_losses)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Mirror loss', "%8.3g" % np.mean(mirror_losses)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Mean KL Div', "%8.3g" % np.mean(kls)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Mean Entropy', "%8.3g" % np.mean(entropies)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Clip Fraction', "%8.3g" % np.mean(clip_fractions)) + "\n")
            sys.stdout.write("-" * 37 + "\n")
            sys.stdout.flush()

            # 打印时间统计
            elapsed = time.time() - train_start_time
            print("Total time elapsed: {:.2f}s. Total steps: {} (fps={:.2f})".format(
                elapsed, self.total_steps, self.total_steps/elapsed))

            # 保存训练指标
            with open(self.train_fn, 'a') as out:
                out.write("{},{}\n".format(np.mean(batch.ep_returns), np.mean(batch.ep_lens)))

            # ========== 定期评估 ==========
            if (itr+1) % self.eval_freq == 0:
                # 评估阶段 - 使用确定性策略（无探索）
                evaluate_start = time.time()
                test = self.sample_parallel(env_fn, self.policy, self.critic, 
                                           self.batch_size, self.max_traj_len, 
                                           deterministic=True)  # deterministic=True表示无探索
                eval_time = time.time() - evaluate_start
                print("evaluate time elapsed: {:.2f} s".format(eval_time))

                avg_eval_reward = np.mean(test.ep_returns)
                print("====EVALUATE EPISODE====  (Return = {})".format(avg_eval_reward))

                # 保存评估指标
                with open(self.eval_fn, 'a') as out:
                    out.write("{},{}\n".format(np.mean(test.ep_returns), np.mean(test.ep_lens)))
                test_ep_lens.append(np.mean(test.ep_lens))
                test_ep_returns.append(np.mean(test.ep_returns))

                # 绘制评估曲线
                plt.clf()
                xlabel = [i*self.eval_freq for i in range(len(test_ep_lens))]
                plt.plot(xlabel, test_ep_lens, color='blue', marker='o', label='Ep lens')
                plt.plot(xlabel, test_ep_returns, color='green', marker='o', label='Returns')
                plt.xticks(np.arange(0, itr+1, step=self.eval_freq))
                plt.xlabel('Iterations')
                plt.ylabel('Returns/Episode lengths')
                plt.legend()
                plt.grid()
                plt.savefig(os.path.join(self.save_path, 'eval.svg'), bbox_inches='tight')

                # 保存策略 - 带迭代编号
                self.save(policy, critic, "_" + repr(itr))

                # 如果是最佳模型，保存为默认名称（覆盖之前的最佳模型）
                if self.highest_reward < avg_eval_reward:
                    self.highest_reward = avg_eval_reward
                    self.save(policy, critic)  # 保存为actor.pt和critic.pt