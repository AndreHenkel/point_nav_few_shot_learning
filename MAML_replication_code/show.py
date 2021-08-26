import gym
import point_nav

import numpy as np
import torch
import random
import numpy as np
import copy

from drl.maml import show_net, display_both_trajectories
from drl.policy import Policy
from drl.baseline import LinearFeatureBaseline
from drl.agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

STATE_SIZE = 2
ACTION_SIZE = 2
MAX_STEPS = 100
TASK_RANGE = 0.5

STATE_DICT = 'test_SGD_SURRL_0.01_tanh_mu_init_scale_0.01_topped_0.1_params_forward_HL_100_100(280)' #actually 0.1 scale and 1.0 topping #scale = [0.0279, 0.0273]
#STATE_DICT =  'test_SGD_SURRL_0.1_tanh_mu_init_scale_1.0_topped_2.0_params_forward_HL_64_64(350)'
#STATE_DICT = 'test_2d_nav_config_SGD_lr_0.01_HL_100_100(200)'
#STATE_DICT = 'test_SGD_SURRL_0.01_tanh_mu_init_scale_0.1_topped_1.0_params_forward_HL_16_16(250)' #scale = [0.0350, 0.0376]
#STATE_DICT = 'test_SGD_SURRL_0.01_tanh_mu_init_scale_1.0_topped_2.0_params_forward_HL_100_100(450)'

#TASK =  [-0.28236289, +0.02733627]
TASK = [random.uniform(-TASK_RANGE, +TASK_RANGE),random.uniform(-TASK_RANGE, +TASK_RANGE)]
#TASK = [-1.3, +1.456]

#STEP_SIZES = [0.5,0.05,0.25,0.125,0.06,0.05,0.05,0.05,0.05,0.05,0.05,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
#STEP_SIZES = [0.1,0.05,0.05,0.05] #according to  the paper
#STEP_SIZES = [0.25,0.25,0.25,0.0,0.25] #as in the github repo
#own tests
STEP_SIZES = [1.0,0.25,0.25,0.25]#,0.5,0.1,0.05]
STEP_SIZES = [0.1,0.05,0.05,0.05]
STEP_SIZES = [0.01,0.002,0.002,0.002]
#STEP_SIZES = [0.03]#, 0.03,0.005,0.005]
#STEP_SIZES = [0.0100,0.0020,0.0020,0.0020]

FC1_UNITS = 100
FC2_UNITS = 100
BATCH_SIZE = 40  #as described in the paper, use 40 for evaluation
SIGMA_INIT_STD = 0.1
SIGMA_MAX_STD = 1.0

ACTION_WITH_DIST = True

env = gym.make("PointNav-v0")

#init agent and baseline
baseline = LinearFeatureBaseline(STATE_SIZE,reg_coeff=1e-5, device=device)
agent = Agent(env, STATE_SIZE, ACTION_SIZE, baseline, batch_size = BATCH_SIZE)

policy = Policy(STATE_SIZE, ACTION_SIZE, fc1=FC1_UNITS, fc2=FC2_UNITS, init_std=SIGMA_INIT_STD, min_std=1e-6, max_std=SIGMA_MAX_STD,device=device)
policy.load_state_dict(torch.load('results/'+STATE_DICT))
policy.to(device)
evaluate_policy_net = copy.deepcopy(policy)

# print("test_task: ",test_task)
# show_net(policy,env,agent,test_task, max_steps=MAX_STEPS,first_step_size=FIRST_STEP_SIZE, step_size=STEP_SIZE, inner_updates=INNER_UPDATES)

#basic

print("Using task: ",TASK)

states_before,_,rewards_before = agent.sample_trajectory(evaluate_policy_net, TASK,action_with_dist=ACTION_WITH_DIST)
print("rewards before: ",sum(rewards_before))
#train
states_after = []
for i_sz,step_size in enumerate(STEP_SIZES):
    train_batch = agent.create_batchepisode(evaluate_policy_net,TASK,device=device) #on policy, so sample with new policy the new train_batches
    evaluate_policy_net.update_inner_loop_policy(train_batch, step_size=step_size, first_order=False)
    #test afterwards
    print("step_size: ",step_size)
    states_after,_,rewards_after = agent.sample_trajectory(evaluate_policy_net, TASK,action_with_dist=ACTION_WITH_DIST)
    print("steps: ",len(states_after))
    print("rewards after: ",sum(rewards_after))

titel = "Task_{}_{}_name_{}.".format(TASK[0], TASK[1], STATE_DICT)
titel = "maml_sgd_0.01_HL_{}_{}".format(FC1_UNITS,FC2_UNITS)
display_both_trajectories(np.array(states_before), np.array(states_after),TASK,title=titel,figtext=STEP_SIZES)
