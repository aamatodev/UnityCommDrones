import os
import random
import time


import numpy as np
import torch
import tyro
from mlagents_envs.environment import UnityEnvironment
from torch import optim, nn, cuda
from mlagents_envs.envs.unity_aec_env import UnityAECEnv
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer
import wandb

from python.utils.args import Args


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

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    args = tyro.cli(Args)

    """ WANDB SETUP"""

    wandb.init(
        project="commAgents",
        sync_tensorboard=True,
        name=args.run_name,
        monitor_gym=True,
        save_code=True,
    )
    """ WRITER SETUP"""

    writer = SummaryWriter(f"runs/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

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
    agents_data = {"loss": [], "old_loss": [], "q_values": [], "epsilon": [], "rewards": []}

    for idx in range(num_agents):
        # Setting up the policies

        policies["policies"].append(QNetworkUnity(env).to(device))
        policies["optimizers"].append(optim.Adam(policies["policies"][idx].parameters(), lr=args.learning_rate))
        policies["target_network"].append(QNetworkUnity(env).to(device))
        policies["target_network"][idx].load_state_dict(policies["policies"][idx].state_dict())

        agents_data["loss"].append([0])
        agents_data["old_loss"].append([0])
        agents_data["q_values"].append([0])
        agents_data["epsilon"].append([0])
        agents_data["rewards"].append([0])


        # Setting up the replay buffers

        rb = ReplayBuffer(
            buffer_size=args.buffer_size,
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
    for global_step in range(args.total_timesteps):
        print(global_step)
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps,
                                  global_step)

        # If random is less than epsilon sample the action space. Otherwise, use the policy.
        if random.random() < epsilon:
            action = env.action_space(env.possible_agents[0]).sample()
        else:
            q_value = policies["policies"][int(env.agent_selection.split("?")[2].split("=")[1])](
                torch.tensor(np.array(obs["observation"])).to(device))
            action = torch.argmax(q_value).cpu().numpy()

        # Step the environment with the selected action
        # if isinstance(action, np.ndarray):
        #     if action.size == 1:
        #         action = action[0]

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
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = (replay_buffers["replay_buffer"][int(env.agent_selection.split("?")[2].split("=")[1])]
                        .sample(args.batch_size))
                with (torch.no_grad()):
                    target_max, _ = policies["target_network"][int(env.agent_selection.split("?")[2].split("=")[1])](
                        data.next_observations
                    ).max(dim=1)

                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())

                old_val = policies["policies"][int(env.agent_selection.split("?")[2].split("=")[1])](
                    data.observations
                ).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                agents_data["loss"][int(env.agent_selection.split("?")[2].split("=")[1])][0] = loss
                agents_data["epsilon"][int(env.agent_selection.split("?")[2].split("=")[1])][0] = epsilon
                agents_data["rewards"][int(env.agent_selection.split("?")[2].split("=")[1])][0] = reward


                if global_step % 100 == 0:
                    for a in range(num_agents):
                        writer.add_scalar(f"charts/agent-{a}/loss",  agents_data["loss"][a][0], global_step)
                        writer.add_scalar(f"charts/agent-{a}/epsilon", agents_data["epsilon"][a][0], global_step)
                        writer.add_scalar(f"charts/agent-{a}/rewards", agents_data["rewards"][a][0], global_step)

                # optimize the model
                policies["optimizers"][int(env.agent_selection.split("?")[2].split("=")[1])].zero_grad()
                loss.backward()
                policies["optimizers"][int(env.agent_selection.split("?")[2].split("=")[1])].step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(
                        policies["target_network"][int(env.agent_selection.split("?")[2].split("=")[1])].
                                parameters(),
                        policies["policies"][int(env.agent_selection.split("?")[2].split("=")[1])].
                                parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    # save the model
    if args.save_model:
        for a in range(num_agents):
            model_path = f"runs/{args.run_name}/{args.exp_name}-{a}.cleanrl_model"
            torch.save(policies["policies"][a].state_dict(), model_path)
            print(f"model saved to {model_path}")
