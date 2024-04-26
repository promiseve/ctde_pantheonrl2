import gym
import numpy as np

class EpymarlCompatWrapper(gym.Wrapper):
    def __init__(self, env, num_agents):
        super().__init__(env)
        self.num_agents = num_agents
        self.observation_space = gym.spaces.Tuple([env.observation_space for _ in range(num_agents)])
        self.action_space = gym.spaces.Tuple([env.action_space for _ in range(num_agents)])

    def reset(self):
        obs = self.env.reset()
        return self._reshape_obs(obs)

    def step(self, actions):
        # Assuming actions are provided as a list or tuple of actions for all agents
        joint_action = self._combine_actions(actions)
        obs, reward, done, info = self.env.step(joint_action)
        return self._reshape_obs(obs), reward, done, info

    def _combine_actions(self, actions):
        # Combine the actions from each agent into the format expected by the environment
        # This could be as simple as a concatenation or more complex if the environment requires
        joint_action = np.array(actions)
        return joint_action

    def _reshape_obs(self, obs):
        # Reshape the observation to be a tuple of observations, one for each agent
        return tuple(obs for _ in range(self.num_agents))
