from  mlagents_envs.environment import UnityEnvironment


"""
This class uses the UnityEnvironment class from mlagents_envs.environment to create a wrapper for the Unity environment. 
The objective is to create a class that can be used in the same way as the PettingZoo environments.
"""

class UnityEnv:
    def __init__(self, parameter1, parameter2):
        self.parameter1 = parameter1
        self.parameter2 = parameter2

    def method1(self):
        # Method logic here
        pass

    def method2(self):
        # Method logic here
        pass