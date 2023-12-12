class BaseEnv:
    def __init__(self):
        # Define any other attributes or variables needed for your environment initialization
        pass

    def reset(self):
        # Reset the environment to its initial state
        # Return the initial observation
        pass

    def step(self, action):
        # Take a step in the environment given an action
        # Return the next observation, reward, whether the episode is done, and additional information
        pass

    def render(self, mode='human'):
        # Render the current state of the environment (optional)
        pass

    def close(self):
        # Clean up resources or close the environment (optional)
        pass
