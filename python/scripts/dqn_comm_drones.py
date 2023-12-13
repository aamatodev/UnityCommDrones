import time

import numpy as np
import torch
from pandas._typing import Shape
from torch import nn, optim

from cleanrl_utils.buffers import ReplayBuffer
from python.env.unity_env import DronesUnityParallelEnv
from python.utils.args import Args
from python.utils.utlis import Utils


class QNetwork(nn.Module):
    def __init__(self, observation_space, action_space: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(observation_space).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_space),
        )

    def forward(self, x):
        return self.network(x)


class DronesDQN:

    def __init__(self, env: DronesUnityParallelEnv, num_agents=3, device=None, args: Args = None, log_to_wandb=True):
        self.env = env
        self.num_agents = num_agents
        self.device = device
        self.args = args
        self.log_to_wandb = log_to_wandb

    def train(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        """ LEARNER SETUP """

        # set policies and optimizers for each agent
        policies = {"policies": [], "optimizers": [], "target_network": []}
        replay_buffers = {"replay_buffer": []}
        wandb_records = {
            "loss": [],
            "old_loss": [],
            "q_values": [],
            "epsilon": [],
            "rewards": []
        }

        for idx in range(self.num_agents):
            policies["policies"].append(
                QNetwork(self.env.get_observation_specs()[0].shape,
                         self.env.get_action_space().n)
                .to(device=self.device)
            )

            policies["optimizers"].append(
                optim.Adam(policies["policies"][idx].parameters(), lr=self.args.learning_rate)
            )

            policies["target_network"].append(
                QNetwork(self.env.get_observation_specs()[0].shape,
                         self.env.get_action_space().n).to(device=self.device)
            )

            replay_buffers["replay_buffer"].append(
                ReplayBuffer(
                    buffer_size=self.args.buffer_size,
                    observation_space=self.env.get_observation_space(),
                    action_space=self.env.get_action_space(),
                    device=self.device,
                    n_envs=1
                )
            )

        self.env.reset()

        for global_steps in range(self.args.total_timesteps):

            epsilon = Utils.linear_schedule(self.args.start_e, self.args.end_e,
                                            int(self.args.exploration_fraction * self.args.total_timesteps),
                                            global_steps)
            actions = []
            for agent_id in range(self.num_agents):
                if np.random.random() < epsilon:
                    action = np.array(
                        [self.env.get_action_space().sample() for _ in range(self.env.get_num_of_agents())])
                else:
                    q_values = policies["policies"][0](torch.Tensor(self.env.get_observation_space()).to(self.device))
                    action = torch.argmax(q_values, dim=1).cpu().numpy()
                actions.append([action])

            # TRY NOT TO MODIFY: execute the game and log data.
            obs, rewards, dones, infos = self.env.step(actions)

            for idx in range(self.num_agents):
                replay_buffers["replay_buffer"][idx].add(obs=self.env.get_observation_space()[idx],
                                                         next_obs=obs[idx], action=actions[idx],
                                                         reward=rewards[idx], done=dones[idx])

                if replay_buffers["replay_buffer"][idx].is_full():
                    # Training the network
                    obs, next_obs, actions, rewards, dones = replay_buffers["replay_buffer"][idx].sample(
                        self.args.batch_size)

                    q_values = policies["policies"][idx](obs).gather(1, actions.unsqueeze(1)).squeeze(1)
                    next_q_values = policies["target_network"][idx](next_obs).max(1)[0]
                    expected_q_values = rewards + (self.args.gamma * next_q_values * (1 - dones))

                    loss = nn.MSELoss()(q_values, expected_q_values)

                    policies["optimizers"][idx].zero_grad()
                    loss.backward()
                    policies["optimizers"][idx].step()
