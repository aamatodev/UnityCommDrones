from mlagents_envs.envs.unity_aec_env import UnityAECEnv

"""
Provides methods to allows unity to work with cleanRL
"""


class CleanRLUnityUnityWrapper(UnityAECEnv):

    def get_step(self, behavior_name):
        return self._env.get_steps(behavior_name=behavior_name)
