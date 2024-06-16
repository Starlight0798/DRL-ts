import argparse
import os, sys
from copy import deepcopy
from functools import partial
import gymnasium as gym
import numpy as np
import torch
from pettingzoo.classic import tictactoe_v3
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.data.stats import InfoStats
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from loguru import logger as loguru_logger

# setup root path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils.handler import raise_warning
# raise_warning()


def get_env(render_mode: str | None = None) -> PettingZooEnv:
    return PettingZooEnv(tictactoe_v3.env(render_mode=render_mode))


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="a smaller gamma favors earlier win",
    )
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--step-per-epoch", type=int, default=10000)
    parser.add_argument("--step-per-collect", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128])
    parser.add_argument("--training-num", type=int, default=50)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.05)
    parser.add_argument(
        "--self-play",
        default=True,
        action="store_true",
        help="train & watch the agent with self-play or not",
    )
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="no training, watch the play of pre-trained models",
    )
    parser.add_argument(
        "--agent-id",
        type=int,
        default=0,
        help="the learned agent plays as the agent_id-th player. Choices are 0 and 1.",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default="./log/tic_tac_toe/dqn/policy.pth",
        help="the path of agent pth file for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--opponent-path",
        type=str,
        default="",
        help="the path of opponent agent pth file for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def get_agents(
    args: argparse.Namespace = get_args(),
    agent_learn: BasePolicy | None = None,
    agent_opponent: BasePolicy | None = None,
    optim: torch.optim.Optimizer | None = None,
) -> tuple[BasePolicy, torch.optim.Optimizer | None, list]:
    # setup env
    env = get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    args.state_shape = observation_space.shape or int(observation_space.n)
    args.action_shape = env.action_space.shape or int(env.action_space.n)
    loguru_logger.info(f"Observations shape: {args.state_shape}")
    loguru_logger.info(f"Actions shape: {args.action_shape}")
    
    # setup agent
    if agent_learn is None:
        net = Net(
            state_shape=args.state_shape,
            action_shape=args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        ).to(args.device)
        loguru_logger.info(f'Net structure: \n' + str(net))
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        agent_learn = DQNPolicy(
            model=net,
            optim=optim,
            action_space=env.action_space,
            estimation_step=args.n_step,
            discount_factor=args.gamma,
            target_update_freq=args.target_update_freq,
        )
        # It doesn't matter if model fails to load
        with loguru_logger.catch():
            if args.resume_path:
                agent_learn.load_state_dict(torch.load(args.resume_path))
                loguru_logger.info(f"Loaded agent from {args.resume_path}")
            
    # setup opponent
    if agent_opponent is None:
        if args.opponent_path:
            agent_opponent = deepcopy(agent_learn)
            agent_opponent.load_state_dict(torch.load(args.opponent_path))
            loguru_logger.info(f"Loaded opponent from {args.opponent_path}")
        elif args.self_play:
            agent_opponent = agent_learn
            loguru_logger.info("Using self policy as opponent")
        else:
            agent_opponent = RandomPolicy(action_space=env.action_space)
            loguru_logger.info("Using random policy as opponent")

    if args.agent_id == 0:
        agents = [agent_learn, agent_opponent]
    else:
        agents = [agent_opponent, agent_learn]
    policy = MultiAgentPolicyManager(policies=agents, env=env)
    return policy, optim, env.agents


def train_agent(
    args: argparse.Namespace = get_args(),
    agent_learn: BasePolicy | None = None,
    agent_opponent: BasePolicy | None = None,
    optim: torch.optim.Optimizer | None = None,
) -> tuple[InfoStats, BasePolicy]:
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
    
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # policy
    policy, optim, agents = get_agents(
        args,
        agent_learn=agent_learn,
        agent_opponent=agent_opponent,
        optim=optim,
    )

    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    
    # test with random policy
    test_policy, _, _ = get_agents(
        agent_learn=policy.policies[agents[args.agent_id]],
        agent_opponent=RandomPolicy(action_space=get_env().action_space),  
    )
    test_collector = Collector(test_policy, test_envs, exploration_noise=True)
    
    # policy.set_eps(1)
    train_collector.reset()
    train_collector.collect(n_step=args.batch_size * args.training_num)
    
    # log
    log_path = os.path.join(args.logdir, "tic_tac_toe", "dqn")
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy: BasePolicy) -> None:
        if hasattr(args, "model_save_path"):
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(args.logdir, "tic_tac_toe", "dqn", "policy.pth")
        agent_learn = policy.policies[agents[args.agent_id]]
        torch.save(agent_learn.state_dict(), model_save_path)
        loguru_logger.info(f"Saved policy to {model_save_path}")
        

    def train_fn(epoch: int, env_step: int) -> None:
        policy.policies[agents[args.agent_id]].set_eps(args.eps_train)

    def test_fn(epoch: int, env_step: int | None) -> None:
        policy.policies[agents[args.agent_id]].set_eps(args.eps_test)

    def reward_metric(rews: np.ndarray) -> np.ndarray:
        return rews[:, args.agent_id]

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
        train_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best_fn,
        update_per_step=1 / args.step_per_collect,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
    ).run()

    return result, policy.policies[agents[args.agent_id]]


def watch(
    args: argparse.Namespace = get_args(),
    agent_learn: BasePolicy | None = None,
    agent_opponent: BasePolicy | None = None,
) -> None:
    env = DummyVectorEnv([partial(get_env, render_mode="human")])
    policy, optim, agents = get_agents(args, agent_learn=agent_learn, agent_opponent=agent_opponent)
    policy.policies[agents[args.agent_id]].set_eps(args.eps_test)
    
    role = 'X' if args.agent_id == 0 else 'O'
    loguru_logger.info(f"Learned agent plays as {role}")
    
    collector = Collector(policy, env, exploration_noise=True)
    collector.reset()
    result = collector.collect(n_episode=1, render=args.render)
    result.pprint_asdict()
    
    
if __name__ == '__main__':
    with loguru_logger.catch():
        args = get_args()
        if args.watch:
            watch()
        else:
            result, agent = train_agent(args)
