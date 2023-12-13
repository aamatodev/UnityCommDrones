from python.env.unity_env import DronesUnityParallelEnv

if __name__ == "__main__":

    # Create the environment - Port 5004 is used for communication with the unity editor, while the 5005 is for the
    # build
    env = DronesUnityParallelEnv(file_name=None, seed=1, no_graphics=False, worker_id=0, base_port=5004)

    # Reset the environment
    env.reset()
    actions = []

    for i in range(100):
        actions = []
        if env.get_num_of_agents() > 0:
            for agent_id in range(env.get_num_of_agents()):
                action = env.get_action_space().sample()
                actions.append([action])

            # Take a step in the environment
            env.step(actions)
            print("Step: ", i)
