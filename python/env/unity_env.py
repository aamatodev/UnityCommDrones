import numpy as np
from gymnasium.spaces import Discrete, Box
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

from python.env.base_env import BaseEnv

"""
This class uses the UnityEnvironment class from mlagents_envs.environment to create a wrapper for the Unity environment. 
The objective is to create a class that can be used in the same way as the PettingZoo Parallel API environments.

This class has been customized for the communication drone Environment.
"""


class DronesUnityParallelEnv(BaseEnv):

    # Initialize the environment. Receive the same parameter as the UnityEnvironment class.
    def __init__(self, file_name=None, seed=1, no_graphics=False, worker_id=0, base_port=5005, **kwargs):
        # Create the UnityEnvironment object
        super().__init__()

        self.unityEnv = UnityEnvironment(file_name=file_name, seed=seed, no_graphics=no_graphics, worker_id=worker_id,
                                         base_port=base_port)
        # Reset the environment to fillup all the information
        self.unityEnv.reset()
        # Get the brain name
        self.behavior_names = list(self.unityEnv.behavior_specs.keys())

    """
    This method is used to step in the environment. 
    It receives an action for each agent and returns the observations, rewards, dones and infos
    """

    def step(self, actions):
        # Take a step in the environment

        # We assume we will have just one behavior. This may be wrong in the future
        behavior_name = self.behavior_names[0]

        ##
        # decision_steps contains all the information collected by the agents since the last simulation step:
        #   - reward: tuple containing the rewards for each agent
        #   - action_mask: tuple containing the action mask for each agent
        #
        # terminal_steps contains the data a batch of Agents collected when their episode terminated.
        #   - obs:          is a list of numpy arrays observations collected by the batch of agent -
        #   - reward:       is a float vector of length batch size
        #   - interrupted:  is an array of booleans of length batch size. True if the associated Agent was interrupted
        #                   since the last decision step.
        #   - agent_id     is an int vector of length batch size containing unique identifier for the Agent
        ##

        # decision_steps, terminal_steps = self.unityEnv.get_steps(behavior_name)

        actions = np.array(actions)
        actions = ActionTuple(discrete=actions)
        self.unityEnv.set_actions(behavior_name, actions)

        self.unityEnv.step()

        decision_steps, terminal_steps = self.unityEnv.get_steps(behavior_name)

        # This means that the episode is finished
        if len(terminal_steps.obs[0]) > 0:
            observations = terminal_steps.obs
            rewards = terminal_steps.reward
            dones = terminal_steps.interrupted
            infos = {}

        else:
            observations = decision_steps.obs
            rewards = decision_steps.reward
            dones = [False for _ in range(self.get_num_of_agents())]
            infos = {}

        return observations, rewards, dones, infos

    def get_last_step(self):

        # We assume we will have just one behavior. This may be wrong in the future
        behavior_name = self.behavior_names[0]

        decision_steps, terminal_steps = self.unityEnv.get_steps(behavior_name)

        # Get the observations, rewards, dones and infos
        observations = terminal_steps.obs
        rewards = terminal_steps.reward
        dones = terminal_steps.interrupted
        infos = {}

        return observations, rewards, dones, infos

    def reset(self):
        # Reset the environment
        self.unityEnv.reset()
        # We assume we will have just one behavior. This may be wrong in the future
        behavior_name = self.behavior_names[0]

        decision_steps, terminal_steps = self.unityEnv.get_steps(behavior_name)

        # Get the observations, rewards, dones and infos
        observations = decision_steps.obs
        rewards = decision_steps.reward
        dones = terminal_steps.interrupted
        infos = {}

        return observations, rewards, dones, infos


    def close(self):
        self.unityEnv.close()

    """
    This method is used to retrieve the action for each agent. We consider that each agent has the same action sapce. 
    Therefore, we just return the action space of the first agent.
    """

    def get_action_space(self):
        # We assume we will have just one behavior. This may be wrong in the future
        behavior_name = self.behavior_names[0]

        # Get the action space
        action_space = self.unityEnv.behavior_specs[behavior_name].action_spec

        # We assume that we will have a discrete action space with a single branch
        assert action_space.is_discrete(), "The action space is not discrete"
        action_space = Discrete(action_space.discrete_branches[0])

        return action_space

    """
    This method is used to retrieve the observation space for each agent. We consider that each agent has the same
    observation space. Therefore, we just return the observation space of the first agent.
    """

    def get_observation_specs(self):
        # We assume we will have just one behavior. This may be wrong in the future
        behavior_name = self.behavior_names[0]

        # Get the observation space
        observation_space = self.unityEnv.behavior_specs[behavior_name].observation_specs

        return observation_space

    def get_action_mask(self, agent_id):

        # We assume we will have just one behavior. This may be wrong in the future
        behavior_name = self.behavior_names[0]
        decision_steps, terminal_steps = self.unityEnv.get_steps(behavior_name)
        action_mask = decision_steps.action_mask[0][agent_id]

        return action_mask

    """ This methods allow to retrieve the number of agents in the environment. """

    def get_num_of_agents(self):
        # We assume we will have just one behavior. This may be wrong in the future
        behavior_name = self.behavior_names[0]
        num_of_agents = max(
            len(self.unityEnv.env_state[behavior_name][0]),
            len(self.unityEnv.env_state[behavior_name][1])
        )

        return num_of_agents

    """ This method allows to know if a specified action is allowed by the action mask or not"""

    def is_action_allowed(self, agent_id, action):
        action_mask = self.get_action_mask(agent_id)

        if 0 <= action < len(action_mask):
            # Check if the corresponding value in the action mask is true
            return action_mask[action]
        else:
            # If the random action is out of bounds, it's not allowed
            return False

    """ This methods allows to retrieve an agent observation space"""

    def get_observation_space(self):
        # We assume we will have just one behavior. This may be wrong in the future
        behavior_name = self.behavior_names[0]

        # Get the observation space
        observation_space = self.unityEnv.behavior_specs[behavior_name].observation_specs

        observation_space = Box(
            low=-np.float32(np.inf),
            high=np.float32(np.inf),
            shape=observation_space[0].shape,
            dtype=np.float32,
            )

        return observation_space
