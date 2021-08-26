import gym
import point_nav

import numpy as np
import torch
import random

from drl.maml import show_net
from drl.policy import Policy
from drl.baseline import LinearFeatureBaseline
from drl.agent import Agent


STATE_SIZE = 2
ACTION_SIZE = 2
HORIZON = 100
TASK_SET_AMOUNT = 1000
MAX_STEPS = 100
TASK_RANGE = 0.5

FIRST_STEP_SIZE = 0.001
STEP_SIZE = 0.005
INNER_UPDATES = 3
STATE_DICT = "basic_train_comparison_deep_lfb_own_adam_actual_lr_0.001_horizontal_task_HL_64_64(250)"
FC1_UNITS = 64
FC2_UNITS = 64

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_task = np.array([random.uniform(-TASK_RANGE, +TASK_RANGE),random.uniform(-TASK_RANGE, +TASK_RANGE)])
test_task = np.array([0.432, -0.0345]) #override

env = gym.make("PointNav-v0")
obs = env.reset()

#init agent and neural network
baseline = LinearFeatureBaseline(STATE_SIZE,reg_coeff=1e-5, device=device)
agent = Agent(env, STATE_SIZE, ACTION_SIZE, baseline, batch_size = 20)
# model = Network(STATE_SIZE, ACTION_SIZE, seed=4892, fc1_units=100, fc2_units=100)
policy = Policy(STATE_SIZE, ACTION_SIZE, fc1=FC1_UNITS, fc2=FC2_UNITS, init_std=1.0, min_std=1e-6, device=device)
policy.load_state_dict(torch.load(STATE_DICT))
policy.to(device)

print("test_task: ",test_task)
show_net(policy,env,agent,test_task, max_steps=MAX_STEPS,first_step_size=FIRST_STEP_SIZE, step_size=STEP_SIZE, inner_updates=INNER_UPDATES)
