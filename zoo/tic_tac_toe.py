import argparse
import os, sys
from copy import deepcopy
from functools import partial
import gymnasium as gym
import numpy as np
import torch
from pettingzoo.classic import tictactoe_v3
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.utils.net.common import NetBase
from tianshou.utils.net.discrete import NoisyLinear
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.data.stats import InfoStats
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, RainbowPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from loguru import logger as loguru_logger

# setup root path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model import PSCN, MLP
# from utils.handler import raise_warning
# raise_warning()

class Net(NetBase):
    def __init__(self, state_shape, action_shape, num_atoms, noisy_std, device):
        super(Net, self).__init__()
        def linear(in_dim, out_dim) -> NoisyLinear:
            return NoisyLinear(in_dim, out_dim, noisy_std=noisy_std)
        
        self.action_num = int(np.prod(action_shape))
        self.num_atoms = num_atoms
        self.device = device
        self.fc_head = PSCN(np.prod(state_shape), 128, linear=linear)
        self.Q = MLP([128, 128, self.action_num * self.num_atoms], linear=linear)
        self.V = MLP([128, 128, self.num_atoms], linear=linear)
        self.output_dim = self.action_num * self.num_atoms

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        batch = obs.shape[0]
        obs = obs.view(batch, -1)
        x = self.fc_head(obs)
        q = self.Q(x).view(-1, self.action_num, self.num_atoms)
        v = self.V(x).view(-1, 1, self.num_atoms)
        logits = q - q.mean(dim=1, keepdim=True) + v
        probs = logits.softmax(dim=2)
        return probs, state


def get_env(render_mode: str | None = None) -> PettingZooEnv:
    return PettingZooEnv(tictactoe_v3.env(render_mode=render_mode))


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--num-atoms", type=int, default=51)
    parser.add_argument("--v-min", type=float, default=-10.0)
    parser.add_argument("--v-max", type=float, default=10.0)
    parser.add_argument("--noisy-std", type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--step-per-epoch", type=int, default=10000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
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
    
    # log
    args.task = 'tic_tac_toe'
    args.algo_name = 'rainbow'
    log_name = os.path.join(args.task, args.algo_name, str(args.seed))
    log_path = os.path.join(args.logdir, log_name)
    
    args.state_shape = observation_space.shape or int(observation_space.n)
    args.action_shape = env.action_space.shape or int(env.action_space.n)
    
    loguru_logger.add(os.path.join(log_path, "model.log"), rotation="1 MB", retention="10 days", level="DEBUG")
    loguru_logger.info(f"Observations shape: {args.state_shape}")
    loguru_logger.info(f"Actions shape: {args.action_shape}")
    
    # setup agent
    if agent_learn is None:
        net = Net(
            args.state_shape,
            args.action_shape,
            args.num_atoms,
            args.noisy_std,
            args.device,
        ).to(args.device)
        loguru_logger.info(f'Net structure: \n' + str(net))
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        
        # The algo for policy
        agent_learn: RainbowPolicy[RainbowTrainingStats] = RainbowPolicy(
            model=net,
            optim=optim,
            discount_factor=args.gamma,
            action_space=env.action_space,
            num_atoms=args.num_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            estimation_step=args.n_step,
            target_update_freq=args.target_update_freq,
        ).to(args.device)
        
        # It doesn't matter if pre-trained model fails to load
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
    
    # buffer
    buffer: VectorReplayBuffer = VectorReplayBuffer(
        total_size=args.buffer_size,
        buffer_num=args.training_num,
    )

    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    
    # test with random policy
    test_policy, _, _ = get_agents(
        agent_learn=policy.policies[agents[args.agent_id]],
        agent_opponent=RandomPolicy(action_space=get_env().action_space),  
    )
    test_collector = Collector(test_policy, test_envs, exploration_noise=True)
    
    # policy.set_eps(1)
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

    def save_best_fn(policy: BasePolicy) -> None:
        agent_learn = policy.policies[agents[args.agent_id]]
        model_save_path = os.path.join(log_path, "policy.pth")
        torch.save(agent_learn.state_dict(), model_save_path)
        loguru_logger.info(f"Saved policy to {model_save_path}")
        

    def train_fn(epoch: int, env_step: int) -> None:
        policy.policies[agents[args.agent_id]].set_eps(args.eps_train)


    def test_fn(epoch: int, env_step: int | None) -> None:
        policy.policies[agents[args.agent_id]].set_eps(args.eps_test)


    def reward_metric(rews: np.ndarray) -> np.ndarray:
        return rews[:, args.agent_id]
    
    
    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> str:
        agent_learn = policy.policies[agents[args.agent_id]]
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        torch.save(agent_learn.state_dict(), ckpt_path)
        loguru_logger.info(f"Epoch: {epoch}, EnvStep: {env_step}, GradientStep: {gradient_step}, Saved checkpoint to {ckpt_path}")
        return ckpt_path

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
        save_checkpoint_fn=save_checkpoint_fn,
        update_per_step=1 / args.step_per_collect,
        logger=logger,
        test_in_train=False,
        resume_from_log=args.resume_id is not None,
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
    result = collector.collect(n_episode=args.test_num, render=args.render)
    result.pprint_asdict()
    
    
if __name__ == '__main__':
    with loguru_logger.catch():
        args = get_args()
        if args.watch:
            watch()
        else:
            result, agent = train_agent(args)
