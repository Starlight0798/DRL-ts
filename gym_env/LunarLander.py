import argparse
import os
import sys
import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import DiscreteSACPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.utils.space_info import SpaceInfo
from loguru import logger as loguru_logger

# setup root path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model import PSCN, MLP
# from utils.handler import raise_warning
# raise_warning()

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="LunarLander-v2")
    parser.add_argument("--seed", type=int, default=4213)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--auto-alpha", action="store_true", default=False)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=50000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--rew-norm", type=int, default=False)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    parser.add_argument("--save-buffer-name", type=str, default=None)
    return parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()
        self.model = torch.nn.Sequential(
            PSCN(np.prod(state_shape), 256),
            MLP([256, 256, 64], last_act=True)
        )
        self.output_dim = 64
        self.device = device

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


@loguru_logger.catch()
def run_discrete_sac(args: argparse.Namespace = get_args()) -> None:
    env = gym.make(args.task)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    train_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])
    
    # get observation and action space info
    space_info = SpaceInfo.from_env(env)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape

    loguru_logger.info(f"Observations shape: {args.state_shape}")
    loguru_logger.info(f"Actions shape: {args.action_shape}")
    
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # define model
    net_a, net_c1, net_c2 = [Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        device=args.device,
    ) for _ in range(3)]
    actor = Actor(net_a, args.action_shape, device=args.device, softmax_output=False)
    loguru_logger.info(f'Actor structure: \n' + str(actor))
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1 = Critic(net_c1, last_size=args.action_shape, device=args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, last_size=args.action_shape, device=args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
    loguru_logger.info(f'Critic structure: \n' + str(critic1))

    # define policy
    if args.auto_alpha:
        target_entropy = 0.98 * np.log(np.prod(args.action_shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy: DiscreteSACPolicy = DiscreteSACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
    ).to(args.device)
    
    # log
    args.algo_name = "discrete_sac"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed))
    log_path = os.path.join(args.logdir, log_name)
        
    # load a previous policy
    if args.resume_path:
        load_path = os.path.join(args.logdir, args.task, args.algo_name, str(args.seed), args.resume_path)
        policy.load_state_dict(torch.load(load_path, map_location=args.device))
        loguru_logger.info(f"Loaded agent from: {load_path}")
        
    # replay buffer
    buffer = VectorReplayBuffer(
        total_size=args.buffer_size, 
        buffer_num=args.training_num
    )
    
    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))
        loguru_logger.info(f"Saved best policy to {log_path}")

    def stop_fn(mean_rewards: float) -> bool:
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        if "Pong" in args.task:
            return mean_rewards >= 20
        return False

    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> str:
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        torch.save({"model": policy.state_dict()}, ckpt_path)
        loguru_logger.info(f"Saved checkpoint to {ckpt_path}")
        return ckpt_path

    # watch agent's performance
    def watch() -> None:
        loguru_logger.info("Setup test envs ...")
        test_envs.seed(args.seed)
        if args.save_buffer_name:
            loguru_logger.info(f"Generate buffer with size {args.buffer_size}")
            buffer = VectorReplayBuffer(
                total_size=args.buffer_size, 
                buffer_num=args.test_num
            )
            collector = Collector(policy, test_envs, buffer, exploration_noise=True)
            result = collector.collect(n_step=args.buffer_size)
            loguru_logger.info(f"Save buffer into {args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            loguru_logger.info("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(n_episode=args.test_num, render=args.render)
        result.pprint_asdict()

    if args.watch:
        watch()
        sys.exit(0)

    # test train_collector and start filling replay buffer
    train_collector.reset()
    train_collector.collect(n_step=args.batch_size * args.training_num)
    
    # logger
    logger_factory = LoggerFactoryDefault()
    logger_factory.logger_type = "tensorboard"
    logger = logger_factory.create_logger(
        log_dir=log_path,
        experiment_name=log_name,
        run_id=args.resume_id,
        config_dict=vars(args),
    )
    
    # trainer
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=1 / args.step_per_collect,
        test_in_train=False,
        resume_from_log=args.resume_id is not None,
        save_checkpoint_fn=save_checkpoint_fn,
    ).run()

    loguru_logger.info(result)
    watch()


if __name__ == "__main__":
    run_discrete_sac(get_args())
