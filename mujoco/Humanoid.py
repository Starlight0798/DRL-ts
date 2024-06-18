import argparse
import os, sys
import numpy as np
import torch
from mujoco_env import make_mujoco_env
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import SACPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.continuous import ActorProb, Critic
from loguru import logger as loguru_logger

# setup root path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model import PSCN, MLP
# from utils.handler import raise_warning
# raise_warning()

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Humanoid-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=200000)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=False, action="store_true")
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--start-timesteps", type=int, default=32000)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=100000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default="1")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    return parser.parse_args()
    
    
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
    


@loguru_logger.catch()
def run_sac(args: argparse.Namespace = get_args()) -> None:
    if args.watch:
        args.training_num = 1
        args.test_num = 1
        
    env, train_envs, test_envs, watch_env = make_mujoco_env(
        args.task,
        args.seed,
        args.training_num,
        args.test_num,
        obs_norm=False,
    )
    
    # log
    args.algo_name = "sac"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed))
    log_path = os.path.join(args.logdir, log_name)
    
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]

    loguru_logger.add(os.path.join(log_path, "model.log"), rotation="1 MB", retention="10 days", level="DEBUG")
    loguru_logger.info(f"Observations shape: {args.state_shape}")
    loguru_logger.info(f"Actions shape: {args.action_shape}")
    loguru_logger.info(f"Action range: {np.min(env.action_space.low)}, {np.max(env.action_space.high)}")
    
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
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
    loguru_logger.info(f'Critic structure: \n' + str(critic1))

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy: SACPolicy = SACPolicy(
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

    # load a previous policy
    if args.resume_path:
        load_path = os.path.join(args.logdir, args.task, args.algo_name, str(args.seed), args.resume_path)
        policy.load_state_dict(torch.load(load_path, map_location=args.device))
        loguru_logger.info(f"Loaded agent from: {load_path}")

    # collector
    buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))
        loguru_logger.info(f"Saved best policy to {log_path}")
    
        
    def watch() -> None:
        loguru_logger.info("Setup watch envs ...")
        watch_collector = Collector(policy, watch_env, exploration_noise=True)
        watch_collector.reset()
        loguru_logger.info("Watching agent ...")
        result = watch_collector.collect(n_episode=1, render=args.render)
        result.pprint_asdict()


    if args.watch:
        watch()
        sys.exit(0)
        
    train_collector.reset()
    train_collector.collect(n_step=args.start_timesteps, random=True)    
    
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
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=1 / args.step_per_collect,
        resume_from_log=args.resume_id is not None,
        test_in_train=False,
    ).run()
    loguru_logger.info(result)


if __name__ == "__main__":
    run_sac()
