"""

This is the runner for the communication drone environment. It is used to train the agents.

"""

import numpy as np
import tyro

from python.env.unity_env import DronesUnityParallelEnv
from python.scripts.dqn_comm_drones import DronesDQN
from python.utils.args import Args

if __name__ == "__main__":
    args = tyro.cli(Args)

    args.run_name = "dqn_comm_drones"
    args.exp_name = "dqn_comm_drones"
    args.wandb_project_name = "commAgents"

    ##
    # Create the environment -
    #   -   Port 5004 is used for communication with the unity editor
    #   -   Port 5005 is used for the build
    ##
    env = DronesUnityParallelEnv(file_name=None, seed=1, no_graphics=False, worker_id=0, base_port=5004)
    # Reset the environment
    env.reset()
    # Setting up the training
    dronesDQN = DronesDQN(env=env, num_agents=env.get_num_of_agents(), args=args)
    dronesDQN.train()


