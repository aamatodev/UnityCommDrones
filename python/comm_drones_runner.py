"""

This is the runner for the communication drone environment. It is used to train the agents.

"""
from python.env.unity_env import DronesUnityParallelEnv

if __name__ == "__main__":

    # Create the environment - Port 5004 is used for communication with the unity editor, while the 5005 is for the
    # build
    env = DronesUnityParallelEnv(file_name=None, seed=1, no_graphics=False, worker_id=0, base_port=5004)

