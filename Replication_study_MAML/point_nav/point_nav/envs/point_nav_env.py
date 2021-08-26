# Mainly taken from https://github.com/cbfinn/maml_rl/blob/master/maml_examples/point_env_rand2goal.py
# just removed the rllab dependencies for easier use

import gym
import numpy as np
import random

INIT_STATE_RANGE = 1

class PointNavEnv(gym.Env):

    def __init__(self):
        self._goal = None
        self._state = np.zeros((2))

    @property
    def observation_space(self):
        return np.array([-np.inf,np.inf,2])

    @property
    def action_space(self):
        return np.array([-0.1,0.1,2])

    def reset(self, task_goal=None, random_init_state=False):
        goal = task_goal
        if goal is not None:
            self._goal = goal
        elif self._goal is None:
            # Only set a new goal if this env hasn't had one defined before.
            goals = [np.array([-0.5,0]), np.array([0.5,0])]
            self._goal = goals[np.random.randint(2)]

        #TODO: init self._state randomly
        if random_init_state:
            self._state = (random.uniform(-INIT_STATE_RANGE, +INIT_STATE_RANGE),random.uniform(-INIT_STATE_RANGE, +INIT_STATE_RANGE))
        else:
            self._state = np.zeros((2))

        observation = np.copy(self._state)
        return observation

    def step(self, action):
        action=np.clip(action,-0.1,0.1)
        self._state[0] = self._state[0] + action[0]
        self._state[1] = self._state[1] + action[1]
        x = self._state[0]
        y = self._state[1]
        x -= self._goal[0]
        y -= self._goal[1]
        reward = -np.sqrt(x ** 2 + y ** 2)
        done = ((np.abs(x) < 0.01) and (np.abs(y) < 0.01))
        next_observation = np.copy(self._state)
        return next_observation, reward, done

    def render(self):
        print('current state:', self._state)
