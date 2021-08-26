import gym
import point_nav
import numpy as np
import torch

from drl.baseline import LinearFeatureBaseline
import random

from drl.agent import Agent
from drl.maml import display_trajectory, show_net, evaluate_net
from drl.policy import Policy


FC1_UNITS = 16
FC2_UNITS = 16

STATE_SIZE = 2
ACTION_SIZE = 2

TASK_RANGE = 0.5
MAX_STEPS = 100
INNER_UPDATES_PER_TASK = 1
INNER_UPDATE_LEARNING_RATE = 0.01
EVALUATION_LR = 0.1
SAMPLE_TASK_BATCH_SIZE = 20
TASK_SET_AMOUNT = 10000
SIGMA_INIT_STD = 0.1
SIGMA_MAX = 1.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ",device)

env = gym.make("PointNav-v0")
obs = env.reset()

#init agent and neural network
baseline = LinearFeatureBaseline(STATE_SIZE,reg_coeff=1e-5, device=device)
agent = Agent(env, STATE_SIZE, ACTION_SIZE, baseline, batch_size = SAMPLE_TASK_BATCH_SIZE)
# model = Network(STATE_SIZE, ACTION_SIZE, seed=4892, fc1_units=100, fc2_units=100)
policy = Policy(STATE_SIZE, ACTION_SIZE, fc1=FC1_UNITS, fc2=FC2_UNITS, init_std=SIGMA_INIT_STD, min_std=1e-6, max_std=SIGMA_MAX, optim_lr=INNER_UPDATE_LEARNING_RATE, device=device)
policy.to(device)

#test task to check performance
#defining 500 tasks(mentioned in MAML, not yet sure though if appropriate amount) in the range of TASK_Range, two dimensional positive and negative going
tasks = [np.array([random.uniform(-TASK_RANGE, +TASK_RANGE),random.uniform(-TASK_RANGE, +TASK_RANGE)]) for i in range(TASK_SET_AMOUNT)]

#test task to check performance
test_task = np.array([random.uniform(-TASK_RANGE, +TASK_RANGE),random.uniform(-TASK_RANGE, +TASK_RANGE)])
test_task = np.array([0.432, -0.0345]) #override
print("test_task: ", test_task)

it = 0
while True:
    task_batch = [random.choice(tasks) for i in range(SAMPLE_TASK_BATCH_SIZE)]
    for task in task_batch:
        task_training_samples = agent.create_batchepisode(policy,task, max_steps=MAX_STEPS, gamma=0.99,gae_lambda=1.0,device=device)
        policy.update_inner_loop_policy(task_training_samples, step_size = INNER_UPDATE_LEARNING_RATE)

    if it%10 == 0:
        evaluate_net(policy,agent, test_task, lr=EVALUATION_LR)
        print("iterations: ",it)
        if it%20 == 0 and it !=0:
            torch.save(policy.state_dict(), 'results/basic_train_comparison_SGD_0.01_SIGMA_0.1_HL_16_16(%d)' %it)

    print("iterations: ",it)
    it = it + 1
