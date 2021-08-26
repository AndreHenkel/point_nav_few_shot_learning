from collections import deque
import random
import numpy as np
import torch
import ptan

#this is no longer needed, as the "memory" in terms of importance sampling wasn't implemented.
# And for the short duration the EpisodeBatch class is used.

##decide weather to use gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device="cpu"#hotfix, due to error: RuntimeError: Expected object of device type cuda but got device type cpu for argument #1 'self' in call to _th_addmm

GAMMA = 0.99

class Episode(object):
    def __init__(self, task, policy):
        self._policy = policy
        self._task = task
        self._states = []
        self._actions = []
        self._rewards = []
        self._dones = [] #the last one is always done, no?
        self._q_vals = None


    def add_sar(self, state, action, reward):
        self._states.append(state.astype(np.float32))
        self._actions.append(action.astype(np.float32))
        self._rewards.append(reward.astype(np.float32))

    @property
    def task(self):
        return self._task

    @property
    def states(self):
        #return self._states
        return torch.FloatTensor(self._states).to(device)

    @property
    def actions(self):
        #return self._actions
        return torch.FloatTensor(self._actions).to(device)

    @property
    def rewards(self):
        return self._rewards
        #return torch.FloatTensor(self._rewards).to(device)

    def __len__(self):
        return len(self._rewards)

    @property
    def q_vals(self):
        if self._q_vals is None:
            self._calc_q_value()
        return self._q_vals

    def _calc_q_value(self):
        rewards_np = np.array(self.rewards, dtype=np.float32)
        last_vals_v = self._policy(self.states)[2]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0] #TODO: is that necessary if only one trajectory is considered?
        rewards_np += GAMMA * last_vals_np

        ref_vals_v = torch.FloatTensor(rewards_np).to(device)
        self._q_vals = ref_vals_v
        del self._policy #since we don't need the policy anymore



class ReplayMemory(object):
    """

    """
    def __init__(self, size=10000):
        self._episodes = []


    def add_episode(self, episode):
        self._episodes.append(episode)

    def sample(self):
        """
            According to the MAML paper appendix A.2. Reinforcement Learning:
            "In the 2D navigation, we used a meta batchsize of 20;[...]"
            Sampling the x unique State-Action-Reward pairs and return them as FloatTensors
        """
        return self._episodes

    def get_all(self):
        """
            Returns all states,actions rewards in the buffer
        """
        # s = torch.FloatTensor(self._states).to(device)
        # a = torch.FloatTensor(self._actions).to(device)
        # r = torch.FloatTensor(self._rewards).to(device)
        return self._episodes

    def get_rewards(self):
        rew = 0
        for e in self._episodes:
            rew += sum(e.rewards)
        return rew
