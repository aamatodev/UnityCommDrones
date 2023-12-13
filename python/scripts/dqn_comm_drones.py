import time

import numpy as np
import torch
from pandas._typing import Shape
from torch import nn, optim

from cleanrl_utils.buffers import ReplayBuffer
from python.env.unity_env import DronesUnityParallelEnv
from python.utils.args import Args
from python.utils.logger import Logger
from python.utils.utlis import Utils

import torch.nn.functional as F


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

        if self.log_to_wandb:
            self.logger = Logger(args)

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

        obs, rewards, dones, infos = self.env.reset()

        for global_steps in range(self.args.total_timesteps):

            epsilon = Utils.linear_schedule(self.args.start_e, self.args.end_e,
                                            int(self.args.exploration_fraction * self.args.total_timesteps),
                                            global_steps)
            actions = []
            for agent_id in range(self.num_agents):
                if np.random.random() < epsilon:
                    action = self.env.get_action_space().sample()
                else:
                    q_values = policies["policies"][agent_id](torch.Tensor(obs[0][agent_id]).to(self.device))
                    action = torch.argmax(q_values).cpu().numpy()
                actions.append([action])

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, dones, infos = self.env.step(actions)

            for idx in range(self.num_agents):
                replay_buffers["replay_buffer"][idx].add(obs=obs[0][idx],
                                                         next_obs=next_obs[0][idx], action=actions[idx],
                                                         reward=rewards[idx], done=dones[idx])
            obs = next_obs

            # ALGO LOGIC: training.
            if global_steps > self.args.learning_starts:

                if global_steps % self.args.train_frequency == 0:

                    for idx in range(self.num_agents):
                        data = replay_buffers["replay_buffer"][idx].sample(self.args.batch_size)
                        with (torch.no_grad()):
                            target_max, _ = policies["target_network"][idx](
                                data.next_observations
                            ).max(dim=1)

                            td_target = data.rewards.flatten() + self.args.gamma * target_max * (
                                        1 - data.dones.flatten())

                        old_val = policies["policies"][idx](
                            data.observations
                        ).gather(1, data.actions).squeeze()

                        loss = F.mse_loss(td_target, old_val)

                        if self.log_to_wandb:
                            if global_steps % 100 == 0:
                                self.logger.log(f"charts/agent-{idx}/loss", loss, global_steps)
                                self.logger.log(f"charts/agent-{idx}/epsilon", epsilon, global_steps)
                                self.logger.log(f"charts/agent-{idx}/rewards", rewards[idx], global_steps)

                        # optimize the model
                        policies["optimizers"][idx].zero_grad()
                        loss.backward()
                        policies["optimizers"][idx].step()

                        # update target network
                        if global_steps % self.args.target_network_frequency == 0:
                            for target_network_param, q_network_param in zip(
                                    policies["target_network"][idx].
                                            parameters(),
                                    policies["policies"][idx].
                                            parameters()):
                                target_network_param.data.copy_(
                                    self.args.tau * q_network_param.data + (
                                                1.0 - self.args.tau) * target_network_param.data
                                )

        # save the model
        if self.args.save_model:
            for a in range(self.env.get_num_of_agents()):
                model_path = f"runs/{self.args.run_name}/{self.args.exp_name}-{a}.cleanrl_model"
                torch.save(policies["policies"][a].state_dict(), model_path)
                print(f"model saved to {model_path}")
