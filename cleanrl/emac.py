import os
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from .nn import Actor, Critic

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    critic_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk
def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = env.action_space.high

    def forward(self, state):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.max_action * torch.tanh(x)






class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

MEM_DTYPE = t.float16


class MemBuffer:
    """Buffer that stores (s, a) representations with MC Q-values. """

    def __init__(self, state_dim, action_dim, capacity, k, mem_dim,
                 device="cuda:1"):
        self.max_size = capacity
        self.ptr = 0
        self.size = 0

        self.size = 0
        self.k = k

        self.sa_cuda = t.zeros(capacity, mem_dim, dtype=MEM_DTYPE).to(device)
        self.q = np.zeros((capacity, 1))
        self.device = device

        self.mapping_cpu = np.random.randn(state_dim + action_dim, mem_dim)
        self.mapping = t.from_numpy(self.mapping_cpu).to(self.device, dtype=MEM_DTYPE)

    def store(self, state, action, q):
        sa = np.concatenate([state, action], axis=0).reshape(1, -1)
        sa = np.dot(sa, self.mapping_cpu)

        self.sa_cuda[self.ptr] = t.from_numpy(sa)
        self.q[self.ptr] = q

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def _calc_l2_dist(self, v1, v2):
        v2.unsqueeze_(1)
        l2 = t.pow(t.sum(t.pow(t.abs(v1 - v2), 2), dim=-1), 0.5)
        return l2

    def retrieve_cuda(self, states, actions, k=None):
        if k is None:
            k = self.k

        sa = t.cat([states, actions], dim=1).to(self.device, dtype=MEM_DTYPE)
        sa = t.mm(sa, self.mapping)

        # TODO: Bug here, I take only first self.size elements
        # TODO 2: I don't recall what the first TODO is about now...
        dists_all = self._calc_l2_dist(self.sa_cuda[:self.size], sa)
        soft = t.nn.Softmin(dim=1)
        dists, inds = t.topk(dists_all, k, dim=1, largest=False)

        weights = soft(dists)

        inds = inds.cpu().numpy()
        weights = weights.cpu().numpy()

        qs = self.q[inds]
        weights = np.expand_dims(weights, 2)
        qs = np.multiply(qs, weights) 
        qs = np.sum(qs, axis=1)

        return qs

    def save(self, file_name):
        np.save(f"{file_name}.npy", {
            "sa": self.sa,
            "q": self.q,
        })
        print("Memory module saved.")
class EpisodicReplayBuffer(object):
    """Buffer that saves transitions for incoming epsiodes. """

    def __init__(self, state_dim, action_dim, mem,
                 max_size=int(1e6), device="cuda", prioritized=False, beta=0.0, 
                 start_timesteps=0, **kwargs):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.mem = mem
        self.expl_noise = kwargs["expl_noise"]

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.q = np.zeros((max_size, 1))
        self.p = np.ones(max_size)
        self.beta = beta

        self.ep_state = []
        self.ep_action = []
        self.ep_next_state = []
        self.ep_reward = []

        self.ep_length = 1000
        self.prioritized = prioritized
        self.start_timesteps = start_timesteps

        self.device = device

    def _add_replay_buffer(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add(self, state, action, next_state, reward, terminations, truncations, env, policy, step): #나중에 맞게 고치기
        self.ep_state.append(state)
        self.ep_action.append(action)
        self.ep_next_state.append(next_state)
        self.ep_reward.append(reward)

        if done_limit:
            dones = [0] * (len(self.ep_state) - 1) + [1]

            # Calculate Q-values
            if not done_env:
                for i_add_step in range(1000):
                    # TODO: Range of (-1, 1) is for HalfCheetah, Walker, Hopper only
                    action_dim = env.action_space.shape[0]
                    action = (
                            policy.select_action(np.array(state))
                            + np.random.normal(0, self.expl_noise, size=action_dim)
                    ).clip(-1, 1)
                    _, r, d, _ = env.step(action)
                    self.ep_reward.append(r)

                    if d:
                        print("Extended only for ", i_add_step)
                        break

            qs = []
            reward_np = np.asarray(self.ep_reward)

            n = len(self.ep_reward)
            for i in range(min(1000, len(self.ep_reward))):
                slide = min(n-i, 1000)
                gamma = np.power(np.ones(slide) * 0.99, np.arange(slide))

                q = np.sum(reward_np[i:i+slide] * gamma)
                qs.append(q)

            # Add to memory
            for s, a, q in zip(self.ep_state, self.ep_action, qs):
                self.mem.store(s, a, q)

            for s, a, ns, r, d in zip(self.ep_state, self.ep_action,
                                      self.ep_next_state, self.ep_reward,
                                      dones):
                self._add_replay_buffer(s, a, ns, r, d)

            self.ep_state.clear()
            self.ep_action.clear()
            self.ep_next_state.clear()
            self.ep_reward.clear()

            if self.prioritized and step >= self.start_timesteps:
                self._recalc_priorities()

    def _recalc_priorities(self):
        self.p[:self.size] = self.mem.q[:self.size].flatten()
        min_mc = np.min(self.p[:self.size])
        if min_mc < 0:
            self.p[:self.size] += np.abs(min_mc)
        
        self.p[:self.size] **= self.beta
        d_s = np.sum(self.p[:self.size]) 
        self.p[:self.size] /= d_s

    def sample(self, batch_size, step=None):
        if step is None or (step < self.start_timesteps or (not self.prioritized)):
            p = None
        else:
            p = self.p[:self.size]

        ind = np.random.choice(self.size, batch_size, p=p)
        return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def save(self, file_name):
        np.save(f"{file_name}.npy", {
            "state": self.state,
            "action": self.action,
            "next_state": self.next_state,
            "reward": self.reward,
            "not_done": self.not_done
        })






if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    actor_target = copy.deepcopy(actor)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    critic = Critic(state_dim, action_dim).to(device)
    critic_target = copy.deepcopy(critic)
    critic_optimizer = optim.Adam(list(critic.parameters()), lr=args.q_lr)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    episode_Buffer=EpisodicReplayBuffer(state_dim, action_dim, mem,
                                             device=device,
                                             prioritized=self.c["prioritized"],
                                             beta=self.c["beta"],
                                             start_timesteps=self.c["start_timesteps"],
                                             expl_noise=self.c["expl_noise"])
    start_time = time.time()
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions=actor(obs)
            actions = actions.detach().cpu().numpy()
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        episode_buffer.add(obs, actions, next_obs, rewards, done_env, done_limit, env, policy, t)
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break
        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            # Compute the target Q value
            target_Q = critic_target(data.next_observations, actor_target(data.next_observations))
            target_Q = data.rewards.flatten()  + (data.dones.flatten() * args.gamma * target_Q).detach()

            mem_q = replay_buffer.mem.retrieve_cuda(state, action)
            mem_q = torch.from_numpy(mem_q).float().to(self.device)

            
class EMAC:
    """Episodic Memory Actor-Critic. """

    def __init__(self, state_dim, action_dim, max_action, discount=0.99, alpha=0.0,
            tau=0.005, device="cuda", log_dir="tb"):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        self.device = device
        self.q = 0

        self.step = 0
        self.tb_logger = SummaryWriter(log_dir)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size, self.step)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        mem_q = replay_buffer.mem.retrieve_cuda(state, action)
        mem_q = torch.from_numpy(mem_q).float().to(self.device)

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        q_loss = F.mse_loss(current_Q, target_Q)
        q_loss_mem = F.mse_loss(current_Q, mem_q)
        critic_loss = (1 - self.alpha) * q_loss + self.alpha * q_loss_mem

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Logging
        if self.step % 250 == 0:
            q = np.mean(current_Q.detach().cpu().numpy())
            self.tb_logger.add_scalar("algo/q", q, self.step)
            q_mem = np.mean(mem_q.cpu().numpy())
            self.tb_logger.add_scalar("algo/q_mem", q_mem, self.step)
            q_loss = q_loss.detach().cpu().item()
            self.tb_logger.add_scalar("algo/q_cur_loss", q_loss, self.step)
            q_mem_loss = q_loss_mem.detach().cpu().item()
            self.tb_logger.add_scalar("algo/q_mem_loss", q_mem_loss, self.step)
            q_total_loss = q_loss + q_mem_loss
            self.tb_logger.add_scalar("algo/critic_loss", q_total_loss, self.step)
            pi_loss = actor_loss.detach().cpu().item()
            self.tb_logger.add_scalar("algo/pi_loss", pi_loss, self.step)
        self.step += 1

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
