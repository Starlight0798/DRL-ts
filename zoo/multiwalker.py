import argparse
import os, sys
from functools import partial
import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from pettingzoo.sisl import multiwalker_v9
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.data.stats import InfoStats
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import SACPolicy
from tianshou.policy import BasePolicy, MultiAgentPolicyManager
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from loguru import logger as loguru_logger

# setup root path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model import *
from utils.handler import print2log, raise_warning
print2log()
# raise_warning()

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--n-walkers", type=int, default=3)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--start-timesteps", type=int, default=16000)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=50000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)    
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--watch", default=False, action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def get_env(args: argparse.Namespace = get_args(), **kwargs) -> PettingZooEnv:
    return PettingZooEnv(multiwalker_v9.env(n_walkers=args.n_walkers, **kwargs))
    
            
class Net(torch.nn.Module):
    def __init__(self, state_shape, action_shape, device, concat):
        super().__init__()
        input_dim = np.prod(state_shape)
        if concat:
            input_dim += np.prod(action_shape)
        self.model = torch.nn.Sequential(
            PSCN(input_dim, 512),
            MLP([512, 512, 128], last_act=True)
        )
        self.output_dim = 128
        self.device = device

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


def get_agents(
    args: argparse.Namespace = get_args(),
    agents: list[BasePolicy] | None = None,
) -> tuple[BasePolicy, list[torch.optim.Optimizer] | None, list]:
    env = get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    
    args.state_shape = observation_space.shape or int(observation_space.n)
    args.action_shape = env.action_space.shape or int(env.action_space.n)
    args.max_action = env.action_space.high[0]
    
    loguru_logger.info(f"Observations shape: {args.state_shape}")
    loguru_logger.info(f"Actions shape: {args.action_shape}")
    loguru_logger.info(f"Action range: {np.min(env.action_space.low)}, {np.max(env.action_space.high)}")
    
    if agents is None:
        agents = []
        for i in range(args.n_walkers):
            # model
            net_a = Net(
                state_shape=args.state_shape, 
                action_shape=args.action_shape,
                device=args.device,
                concat=False
            )
            actor = ActorProb(
                net_a,
                args.action_shape,
                device=args.device,
                unbounded=True,
                conditioned_sigma=True,
            ).to(args.device)
            if i == 0:
                loguru_logger.info(f'Actor structure: \n' + str(actor))
            actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
            net_c1, net_c2 = [Net(
                state_shape=args.state_shape, 
                action_shape=args.action_shape,
                device=args.device,
                concat=True
            ) for _ in range(2)]
            critic1 = Critic(net_c1, device=args.device).to(args.device)
            critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
            critic2 = Critic(net_c2, device=args.device).to(args.device)
            critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
            if i == 0:
                loguru_logger.info(f'Critic structure: \n' + str(critic1))
            
            agent: SACPolicy = SACPolicy(
                actor=actor,
                actor_optim=actor_optim,
                critic=critic1,
                critic_optim=critic1_optim,
                critic2=critic2,
                critic2_optim=critic2_optim,
                tau=args.tau,
                gamma=args.gamma,
                alpha=args.alpha,
                estimation_step=args.n_step,
                action_space=env.action_space,
            )
                
            agents.append(agent)
            
    policy = MultiAgentPolicyManager(
        policies=agents,
        env=env,
        action_scaling=True,
        action_bound_method="clip",
    )
    return policy, env.agents
            
 
@loguru_logger.catch()
def train_agent(
    args: argparse.Namespace = get_args(),
    agents: list[BasePolicy] | None = None,
) -> tuple[InfoStats, BasePolicy]:
    if args.watch:
        args.training_num = 1
        args.test_num = 1
        
    # log
    args.task = "multiwalker"
    args.algo_name = "sac"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed))
    log_path = os.path.join(args.logdir, log_name)
    loguru_logger.add(os.path.join(log_path, "model.log"), rotation="1 MB", retention="10 days", level="DEBUG")
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)
        
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
    
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy, agents = get_agents(args, agents=agents)

    # collector
    buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    
    def save_best_fn(policy: BasePolicy) -> None:
        pass

    def stop_fn(mean_rewards: float) -> bool:
        return False

    def reward_metric(rews: np.ndarray) -> np.ndarray:
        return rews[:, 0]
        
    train_collector.reset()
    train_collector.collect(n_step=args.start_timesteps, random=True)    
    
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
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=1 / args.step_per_collect,
        resume_from_log=args.resume,
        reward_metric=reward_metric,
    ).run()
    
    return result, policy


def watch(args: argparse.Namespace = get_args(), policy: BasePolicy | None = None) -> None:
    env = DummyVectorEnv([partial(get_env, render_mode="human")])
    if not policy:
        loguru_logger.warning(
            "watching random agents, as loading pre-trained policies is currently not supported",
        )
        policy, agents = get_agents(args)
    collector = Collector(policy, env)
    collector_result = collector.collect(n_episode=1, render=args.render)
    loguru_logger.info(f"Watch result:\n {collector_result.pprints_asdict()}")


if __name__ == "__main__":
    with loguru_logger.catch():
        args = get_args()
        if args.watch:
            watch()
        else:
            train_agent()
