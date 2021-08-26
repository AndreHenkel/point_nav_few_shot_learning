import numpy as np
import random
from collections import namedtuple, deque

from drl.model import Network
from drl.replaymemory import Episode

import torch
import torch.nn.functional as F
import torch.optim as optim

import copy

from drl.episode import BatchEpisodes

GAMMA = 0.99 #discount taken from official maml trpo_point repo

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, env, state_size, action_size, baseline, batch_size = 20):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.env = env
        self.baseline = baseline

        self.batch_size = batch_size


    def sample(self, policy, task, max_steps=100, inner_updates=1, learning_rate=0.1, gamma=0.95, gae_lambda=1.0, device='cpu'):
        """
        This function creates "batch-size"-times trajectories, with which the policy is trained.
        Afterwards this policy samples "batch-size"-times trajectories with the newly modified(trained) policy on the given task.

        returns: training_episodes, validation_episodes
        """
        task_training_samples = self.create_batchepisode(policy,task, max_steps=max_steps, gamma=gamma,gae_lambda=gae_lambda,device=device)
        task_training_samples_first = task_training_samples
        policy.update_inner_loop_policy(task_training_samples, step_size = learning_rate)
        #multiple inner_updates, would take the same data from the original policy to iteratore through right now
        for step in range(inner_updates-1):
            task_training_samples = self.create_batchepisode(policy,task, max_steps=max_steps, gamma=gamma,gae_lambda=gae_lambda,device=device)
            policy.update_inner_loop_policy(task_training_samples, step_size = learning_rate)

        task_validation_samples = self.create_batchepisode(policy, task, max_steps=max_steps, gamma=gamma,gae_lambda=gae_lambda,device=device)

        return task_training_samples_first, task_validation_samples

    def create_batchepisode(self, policy, task, max_steps=100, gamma=0.99, gae_lambda=1.0,device='cpu'):
        episode_batch = BatchEpisodes(self.batch_size,gamma=gamma, device=device)
        # bo_id = None
        for batch_id in range(self.batch_size):
            done = False
            episode = Episode(task,policy)
            #reset env
            state = self.env.reset(task_goal=task,random_init_state=False) #reset with the new task
            next_state = None
            action = None
            for i in range(max_steps):
                action = policy.get_action(state)
                #print("action: ",action)
                next_state, reward, done = self.env.step(action)
                episode_batch.append(state,action,reward,batch_id)

                # check if done
                if done:
                    break

                # update the state for the next iteration
                state=next_state
        self.baseline.fit(episode_batch)
        episode_batch.compute_advantages(self.baseline, gae_lambda,normalize=True)
        return episode_batch

    def sample_trajectory(self, policy, task, max_steps = 100, action_with_dist=True): #the max steps come from the Horizon being mentioned in the MAML paper
        """
        Sample trajectory in the environment, with the given task and the current policy (network)
        return: np array of states, actions and already normalized and discounted rewards
        TODO: If done, don't add this sars (ars) to the trajectory
        """
        done = False
        states = []
        actions = []
        rewards = []
        #reset env
        state = self.env.reset(task_goal=task,random_init_state=False) #reset with the new task
        next_state = None
        for i in range(max_steps):
            action = policy.get_action(state,dist=action_with_dist) #TODO: is scaled?
            #print("Action: ", action)
            next_state, reward, done = self.env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # check if done
            if done:
                break

            # update the state for the next iteration
            state=next_state

        return states,actions,rewards
