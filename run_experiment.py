import os
import sys
import argparse
import ray
from functools import partial

import numpy as np
import torch
import pickle

from envs.jvrc import JvrcArmEnv
from rl.algos.ppo import PPO
from rl.policies.actor import Gaussian_FF_Actor
from rl.policies.critic import FF_V
from rl.envs.normalize import get_normalization_params
from rl.envs.wrappers import SymmetricEnv

def import_env(env_name_str):
    """
    根据环境名称字符串导入对应的机器人环境类
    
    Args:
        env_name_str: 环境名称字符串
        
    Returns:
        Env: 对应的环境类
    """
    if env_name_str=='jvrc_walk':
        from envs.jvrc import JvrcWalkEnv as Env
    elif env_name_str=='jvrc_step':
        from envs.jvrc import JvrcStepEnv as Env
    elif env_name_str=='jvrc_run':
        from envs.jvrc import JvrcRunEnv as Env
    elif env_name_str=='jvrc_arm':
        from envs.jvrc import JvrcArmEnv as Env
    elif env_name_str=='jvrc_run_arm':
        from envs.jvrc import JvrcRunArmEnv as Env
    else:
        raise Exception("Check env name!")
    return Env

def run_experiment(args):
    """
    运行训练实验的主函数
    
    Args:
        args: 解析后的命令行参数
    """
    # 导入对应的环境类
    Env = import_env(args.env)

    # 创建环境创建函数，用于并行化环境
    env_fn = partial(Env)  # partial创建了一个可调用对象，调用时返回Env实例
    if not args.no_mirror:
        try:
            print("Wrapping in SymmetricEnv.")
            # 使用对称环境包装器，用于处理机器人的对称性（如左右对称）
            env_fn = partial(SymmetricEnv, env_fn,
                             mirrored_obs=env_fn().robot.mirrored_obs,    # 镜像观察空间索引
                             mirrored_act=env_fn().robot.mirrored_acts,   # 镜像动作空间索引
                             clock_inds=env_fn().robot.clock_inds)        # 时钟信号索引
        except AttributeError as e:
            print("Warning! Cannot use SymmetricEnv.", e)
    
    # 获取观察空间和动作空间的维度
    obs_dim = env_fn().observation_space.shape[0]
    action_dim = env_fn().action_space.shape[0]

    # 设置并行计算环境
    os.environ['OMP_NUM_THREADS'] = '1'  # 限制OpenMP线程数，避免资源竞争
    if not ray.is_initialized():
        ray.init(num_cpus=args.num_procs)  # 初始化Ray分布式计算框架

    # 设置随机种子，确保实验可重复
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 加载或创建策略网络和价值网络
    if args.continued:
        # 从已有模型继续训练
        path_to_actor = ""
        path_to_pkl = ""
        if os.path.isfile(args.continued) and args.continued.endswith(".pt"):
            path_to_actor = args.continued
        if os.path.isdir(args.continued):
            path_to_actor = os.path.join(args.continued, "actor.pt")
        # 根据actor路径构建critic路径
        path_to_critic = path_to_actor.split('actor')[0]+'critic'+path_to_actor.split('actor')[1]
        policy = torch.load(path_to_actor)    # 加载策略网络
        critic = torch.load(path_to_critic)   # 加载价值网络
    else:
        # 创建新的策略网络和价值网络
        policy = Gaussian_FF_Actor(obs_dim, action_dim, fixed_std=np.exp(args.std_dev), bounded=False)
        critic = FF_V(obs_dim)
        
        # 计算观察空间的归一化参数（均值和标准差）
        with torch.no_grad():
            policy.obs_mean, policy.obs_std = map(torch.Tensor,
                                                  get_normalization_params(iter=args.input_norm_steps,
                                                                           noise_std=1,
                                                                           policy=policy,
                                                                           env_fn=env_fn,
                                                                           procs=args.num_procs))
        # 将相同的归一化参数用于价值网络
        critic.obs_mean = policy.obs_mean
        critic.obs_std = policy.obs_std
    
    # 设置为训练模式
    policy.train()
    critic.train()

    # 保存超参数配置
    os.makedirs(args.logdir, exist_ok=True)
    pkl_path = os.path.join(args.logdir, "experiment.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(args, f)

    # 创建PPO算法实例
    algo = PPO(args=vars(args), save_path=args.logdir)
    
    # 开始训练
    algo.train(env_fn, policy, critic, args.n_itr, anneal_rate=args.anneal)

if __name__ == "__main__":
    # 主程序入口：解析命令行参数并运行实验

    parser = argparse.ArgumentParser()

    # 检查命令格式
    if sys.argv[1] != 'train':
        raise Exception("Invalid usage.")

    # 移除'train'参数，只保留实际配置参数
    sys.argv.remove(sys.argv[1])
    
    # 定义所有可配置的命令行参数
    parser.add_argument("--env", required=True, type=str)                     # 环境名称，必须指定
    parser.add_argument("--seed", default=0, type=int)                        # 随机种子
    parser.add_argument("--logdir", type=str, default="./logs_dir/")          # 日志保存路径
    parser.add_argument("--input_norm_steps", type=int, default=100000)       # 计算输入归一化的步数
    parser.add_argument("--n_itr", type=int, default=2000, help="Number of iterations of the learning algorithm")  # 训练迭代次数
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate") # Adam优化器学习率
    parser.add_argument("--eps", type=float, default=1e-5, help="Adam epsilon (for numerical stability)") # Adam优化器epsilon参数
    parser.add_argument("--lam", type=float, default=0.95, help="Generalized advantage estimate discount") # GAE参数λ
    parser.add_argument("--gamma", type=float, default=0.99, help="MDP discount") # 折扣因子γ
    parser.add_argument("--anneal", default=1.0, action='store_true', help="anneal rate for stddev") # 是否退火标准差
    parser.add_argument("--std_dev", type=int, default=-1.5, help="exponent of exploration std_dev") # 探索标准差的对数
    parser.add_argument("--entropy_coeff", type=float, default=0.0, help="Coefficient for entropy regularization") # 熵正则化系数
    parser.add_argument("--clip", type=float, default=0.2, help="Clipping parameter for PPO surrogate loss") # PPO裁剪参数
    parser.add_argument("--minibatch_size", type=int, default=64, help="Batch size for PPO updates") # 小批量大小
    parser.add_argument("--epochs", type=int, default=3, help="Number of optimization epochs per PPO update") # 每次更新优化的轮数
    parser.add_argument("--use_gae", type=bool, default=True,help="Whether or not to calculate returns using Generalized Advantage Estimation") # 是否使用GAE
    parser.add_argument("--num_procs", type=int, default=12, help="Number of threads to train on") # 并行进程数
    parser.add_argument("--max_grad_norm", type=float, default=0.05, help="Value to clip gradients at.") # 梯度裁剪阈值
    parser.add_argument("--max_traj_len", type=int, default=400, help="Max episode horizon") # 每个episode的最大步数
    parser.add_argument("--no_mirror", required=False, action="store_true", help="to use SymmetricEnv") # 是否不使用对称环境
    parser.add_argument("--mirror_coeff", required=False, default=0.4, type=float, help="weight for mirror loss") # 镜像损失的权重
    parser.add_argument("--eval_freq", required=False, default=100, type=int, help="Frequency of performing evaluation") # 评估频率
    parser.add_argument("--continued", required=False, default=None, type=str, help="path to pretrained weights") # 预训练权重路径
    
    # 解析参数并运行实验
    args = parser.parse_args()
    run_experiment(args)