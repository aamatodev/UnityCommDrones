import random
import time

import numpy as np
import torch
from mlagents_envs.environment import UnityEnvironment
from torch import optim, nn, cuda
from mlagents_envs.envs.unity_aec_env import UnityAECEnv
import torch.nn.functional as F

from cleanrl_utils.buffers import ReplayBuffer


# ALGO LOGIC: initialize agent here:
class QNetworkUnity(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space(env.possible_agents[0]).shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space(env.possible_agents[0]).n),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":

    # Algorithm specific arguments
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    save_model: bool = False

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    """ ENV SETUP """

    env = UnityEnvironment(file_name=None)

    env = UnityAECEnv(env)

    env.reset()

    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = env.observation_space(env.possible_agents[0]).shape

    """ LEARNER SETUP """

    # set policies and optimizers for each agent
    policies = {"policies": [], "optimizers": [], "target_network": []}
    replay_buffers = {"replay_buffer": []}

    for idx in range(num_agents):
        # Setting up the policies

        policies["policies"].append(QNetworkUnity(env).to(device))
        policies["optimizers"].append(optim.Adam(policies["policies"][idx].parameters(), lr=learning_rate))
        policies["target_network"].append(QNetworkUnity(env).to(device))
        policies["target_network"][idx].load_state_dict(policies["policies"][idx].state_dict())

        # Setting up the replay buffers

        rb = ReplayBuffer(
            buffer_size=buffer_size,
            observation_space=env.observation_space(env.possible_agents[0]),
            action_space=env.action_space(env.possible_agents[0]),
            device=device,
            n_envs=1
        )

        replay_buffers["replay_buffer"].append(rb)

    start_time = time.time()

    # Getting the env ready
    env.reset()
    obs, _, _, _ = env.last(observe=True)
    # follow the DQN ALG
    for global_step in range(total_timesteps):
        print(global_step)
        epsilon = linear_schedule(start_e, end_e, exploration_fraction * total_timesteps, global_step)

        # If random is less than epsilon sample the action space. Otherwise, use the policy.
        if random.random() < epsilon:
            action = env.action_space(env.possible_agents[0]).sample()
        else:
            q_value = policies["policies"][int(env.agent_selection.split("?")[2].split("=")[1])](torch.tensor(obs).to(device))
            action = torch.argmax(q_value, dim=1).cpu().numpy()

        # Step the environment with the selected action
        env.step(action=action)

        # Get env status after step
        next_obs, reward, done, info = env.last(observe=True)


        if done:
            next_obs = {"observation": next_obs}
            env.reset()


        # Save data to the replay Buffer
        replay_buffers["replay_buffer"][int(env.agent_selection.split("?")[2].split("=")[1])].add(
            obs["observation"],
            next_obs["observation"],
            np.array([action]),
            np.array([reward]),
            np.array([done])
        )

        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > learning_starts:
            if global_step % train_frequency == 0:
                data = (replay_buffers["replay_buffer"][int(env.agent_selection.split("?")[2].split("=")[1])]
                        .sample(batch_size))
                with (torch.no_grad()):
                    target_max, _ = policies["target_network"][int(env.agent_selection.split("?")[2].split("=")[1])](
                        data.next_observations
                    ).max(dim=1)

                    td_target = data.rewards.flatten() + gamma * target_max * (1 - data.dones.flatten())

                old_val = policies["policies"][int(env.agent_selection.split("?")[2].split("=")[1])](
                    data.observations
                ).gather(1, data.actions).squeeze()

                loss = F.mse_loss(td_target, old_val)

                # optimize the model
                policies["optimizers"][int(env.agent_selection.split("?")[2].split("=")[1])].zero_grad()
                loss.backward()
                policies["optimizers"][int(env.agent_selection.split("?")[2].split("=")[1])].step()

            # update target network
            if global_step % target_network_frequency == 0:
                for target_network_param, q_network_param in zip(policies["target_network"][int(env.agent_selection.split("?")[2].split("=")[1])].
                                                                         parameters(),
                                                                 policies["policies"][int(env.agent_selection.split("?")[2].split("=")[1])].
                                                                         parameters()):
                    target_network_param.data.copy_(
                        tau * q_network_param.data + (1.0 - tau) * target_network_param.data
                    )

