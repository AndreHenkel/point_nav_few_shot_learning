"""
    This file incorporates the maml algorithm.
    -Defining tasks
    -holding and handling networks
    -updating network parameters
"""

import gym
import point_nav
import numpy as np
import random
import statistics
import torch
import copy

import matplotlib.pyplot as plt

#from drl.model import Network
from drl.agent import Agent
#from drl.replaymemory import ReplayMemory
from drl.baseline import LinearFeatureBaseline
from drl.policy import Policy, weighted_mean, weighted_normalize

from torch.distributions import Categorical, Independent, Normal
from torch.nn.utils.convert_parameters import _check_param_device

from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.kl import kl_divergence

STATE_SIZE = 2
ACTION_SIZE = 2
HORIZON = 100
TASK_SET_AMOUNT = 1000 #500x20 #TODO: just sample everytime anew, don't worry with actually saving /precreating them
MAX_STEPS = 100

FC1_UNITS = 100
FC2_UNITS = 100

TASK_RANGE = 0.5 #assuming it is squared and in positiv and negative directions

SAMPLE_NETWORK_BATCH_SIZE = 20
SAMPLE_TASK_BATCH_SIZE = 20
SAMPLE_TRAJECTORY_AMOUNT = 20 # as stated in the MAML - paper: Use 20 trajectories for each task to update
INNER_UPDATES_PER_TASK = 1
INNER_UPDATE_LEARNING_RATE = 0.01
#SURROGATE_ADAPTATION_LR = 0.1
SURROGATE_ADAPTATION_LR = INNER_UPDATE_LEARNING_RATE
EVALUATION_LR = 0.01 #step size for adaptation testing // performance

SIGMA_INIT_STD = 0.1
MAX_INIT_STD = 1.0


##decide weather to use gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ",device)
#device="cpu"#hotfix, due to error: RuntimeError: Expected object of device type cuda but got device type cpu for argument #1 'self' in call to _th_addmm
# => problem was, that the fully connected layer in the policy weren't put to the device


def display_trajectory(states, goal,title="Point_Nav"):
    plt.title(title)
    #plt.plot(states)
    #print(states)
    for i in range(len(states)):
        c = i/len(states)
        plt.plot(states[:,0],states[:,1],color='grey')
    plt.scatter(0,0,c='red',zorder=1)#start
    plt.scatter(goal[0],goal[1],c='green',zorder=1)#goal in other color
    plt.show()

def display_both_trajectories(states_before,states_after, goal,title="Point_Nav",figtext=""):
    plt.title(title)
    plt.plot(states_before[:,0],states_before[:,1],label="before update",color='grey')
    plt.plot(states_after[:,0],states_after[:,1],label="after update", color='yellow')

    plt.scatter(0,0,c='red')#start
    plt.scatter(goal[0],goal[1],c='green')#goal in other color
    plt.legend(loc="lower left")
    plt.figtext(0,0,figtext)
    plt.show()

def evaluate_net(policy,agent, test_task, lr=0.1, inner_updates = 1):
    #sample with test_task
    evaluate_policy_net = copy.deepcopy(policy)
    train,valid = agent.sample(evaluate_policy_net,test_task,inner_updates=inner_updates, learning_rate=lr,device=device)
    #print total rewards before and after
    del evaluate_policy_net
    # display_trajectory(train.observations[:,0,:].cpu(),test_task,"evaluate_train")
    # display_trajectory(valid.observations[:,0,:].cpu(),test_task,"evaluate_valid")

    print("Rewards before: ", sum(train.rewards.sum(dim=0)))
    print("Rewards after: ", sum(valid.rewards.sum(dim=0)))

def show_net(policy,env,agent,test_task, max_steps=MAX_STEPS,first_step_size=0.1, step_size=0.05,inner_updates=1):

    #basic
    evaluate_policy_net = copy.deepcopy(policy)
    states_before,_,rewards_before = agent.sample_trajectory(evaluate_policy_net, test_task)
    states_arr_before = np.array(states)
    print("Before rewards: ", sum(rewards_before))
    print("steps: ",len(rewards_before))

    #train
    train_batch = agent.create_batchepisode(evaluate_policy_net,test_task,device=device)
    evaluate_policy_net.update_inner_loop_policy(train_batch, step_size=first_step_size, first_order=False)
    for i in range(inner_updates-1):
        train_batch = agent.create_batchepisode(evaluate_policy_net,test_task,device=device)
        evaluate_policy_net.update_inner_loop_policy(train_batch, step_size=step_size, first_order=False)

    #test afterwards
    states_after,_,rewards_after = agent.sample_trajectory(evaluate_policy_net, test_task)

    del evaluate_policy_net
    print("After rewards: ", sum(rewards_after))
    print("steps: ",len(rewards_after))
    states_arr_after = np.array(states_after)
    display_both_trajectories(states_arr_before, states_arr_after,test_task,title="before and after")


def test_net(model, env, agent, task, max_steps=MAX_STEPS):
    """
        In order to evaluate the model on its meta performance, we have to train a buffered network clone
        on a given test_task, which is not included in the train_task_set.
        MAML: first gradient update, alpha=0.1, further steps use alpha=0.05
        with a meta batch size of 20
    """

    rep_mem = ReplayMemory()
    for i in range(20):
        #b_states, b_actions, b_discounted_rewards = agent.get_trajectory(model, task, max_steps=100)
        rep_mem.add_episode(agent.get_trajectory(model, task, max_steps=MAX_STEPS))

    evaluation_buffer_model = model.get_deep_copy()
    #doing it just once
    evaluation_buffer_model.evaluation_update_policy(rep_mem.sample(), lr=0.1)

    #if k shot learning > 1 then use more update steps with step size of 0.05, as mentioned in the maml paper
    K = 4
    for _ in range(K):
        evaluation_buffer_model.evaluation_update_policy(rep_mem.sample(), lr=0.1)

    #now testing the performance of the first gradient update
    done = False
    episode = Episode(task)
    #reset env
    state = env.reset(task_goal=task,random_init_state=False) #reset with the new task
    next_state = None
    reward = 0
    action = None
    for i in range(max_steps):
        #print(state, action, done)
        action = evaluation_buffer_model.get_action_abs(state) #already sc
        next_state, reward, done = env.step(action)

        # append
        episode.add_episode(state,action,reward)

        # check if done
        if done:
            break

        # update the state for the next iteration
        state=next_state

    print("Steps: ", len(episode))

#taken from https://discuss.pytorch.org/t/compute-the-hessian-matrix-of-a-network/15270/2
# eval Hessian matrix

"""
follow functions are taken from: https://github.com/tristandeleu/pytorch-maml-rl/blob/243214b17da2ebfa152bba784778884b46a7e349/maml_rl/metalearners/maml_trpo.py#L90
"""

def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.clone().detach()
    r = b.clone().detach()
    x = torch.zeros_like(b).float()
    rdotr = torch.dot(r, r)

    for i in range(cg_iters):
        z = f_Ax(p).detach()
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr.item() < residual_tol:
            break

    return x.detach()

def vector_to_parameters(vector, parameters):
    param_device = None

    pointer = 0
    for param in parameters:
        param_device = _check_param_device(param, param_device)

        num_param = param.numel()
        param.data.copy_(vector[pointer:pointer + num_param]
                         .view_as(param).data)
        pointer += num_param


def hessian_vector_product(policy, kl, damping=1e-2):
    grads = torch.autograd.grad(kl,
                                policy.parameters(),
                                create_graph=True)
    flat_grad_kl = parameters_to_vector(grads)

    def _product(vector, retain_graph=True): #vector input is the step_dir from conjugate gradient
        grad_kl_v = torch.dot(flat_grad_kl, vector)
        grad2s = torch.autograd.grad(grad_kl_v, #taking 2nd derivative
                                     policy.parameters(),
                                     retain_graph=retain_graph)
        flat_grad2_kl = parameters_to_vector(grad2s) #transform to vector (just the formatting)

        return flat_grad2_kl + damping * vector #damp vector, in order to not overshoot with the numbers
    return _product

def reinforce_loss(policy, episodes):
    pi = policy(episodes.observations.view((-1, *episodes.observation_shape)))

    log_probs = pi.log_prob(episodes.actions.view((-1, *episodes.action_shape)))
    log_probs = log_probs.view(len(episodes), episodes.batch_size)

    losses = -weighted_mean(log_probs * episodes.advantages,
                            lengths=episodes.lengths)

    return losses.mean()

def adapt(policy, train_episodes, first_order=True, fast_lr=0.1):
        # Loop over the number of steps of adaptation, currently only one
        params = None
        inner_loss = reinforce_loss(policy,
                                    train_episodes)
        params = policy.update_params(inner_loss,
                                           step_size=fast_lr,
                                           first_order=first_order)
        return params

def surrogate_loss(policy, train_episodes, valid_episodes, old_pi=None, first_order=True, step_size=0.1):
    first_order = (old_pi is not None) or first_order
    #create params to track the gradients for the loss wrt the model
    params = adapt(policy, train_episodes,
                              first_order=first_order, fast_lr=step_size)
    with torch.set_grad_enabled(old_pi is None):
        policy.to(device)
        pi = policy(valid_episodes.observations,params=params) #use the adapted params, which were trained on the initial training data, to get policy dist


        #if no old_pi is given, reinterpret the distribution again, in order to have an old_pi availalbe for the calculations
        if old_pi is None:
            old_pi = Independent(Normal(loc=pi.base_dist.loc.detach(), scale=pi.base_dist.scale.detach()),
                                    pi.reinterpreted_batch_ndims)
        log_ratio = (pi.log_prob(valid_episodes.actions)
                     - old_pi.log_prob(valid_episodes.actions))
        ratio = torch.exp(log_ratio)

        losses = -weighted_mean(ratio * valid_episodes.advantages, #negative, because gradient optimizer assumes maximization
                                lengths=valid_episodes.lengths)
        kls = weighted_mean(kl_divergence(pi, old_pi),
                            lengths=valid_episodes.lengths)
    # also mentioned in TRPO paper, that kl MEAN is taken
    return losses.mean(), kls.mean(), old_pi



def MAML(state_dict = None):
    """
        The MAML algorithm from the paper
        TODO: *check if we have to use a bigger step size for the buffer models
    """
    #init environment
    env = gym.make("PointNav-v0")
    obs = env.reset()

    #init agent and neural network
    baseline = LinearFeatureBaseline(STATE_SIZE,reg_coeff=1e-5, device=device)
    agent = Agent(env, STATE_SIZE, ACTION_SIZE, baseline, batch_size = 20)
    # model = Network(STATE_SIZE, ACTION_SIZE, seed=4892, fc1_units=100, fc2_units=100)
    policy = Policy(STATE_SIZE, ACTION_SIZE, fc1=FC1_UNITS, fc2=FC2_UNITS, init_std=SIGMA_INIT_STD, min_std=1e-6, max_std=MAX_INIT_STD,device=device)
    if state_dict is not None:
        policy.load_state_dict(state_dict)
    policy.to(device)

    #defining N tasks(mentioned in MAML, not yet sure though if appropriate amount) in the range of TASK_Range, two dimensional positive and negative going
    tasks = [np.array([random.uniform(-TASK_RANGE, +TASK_RANGE),random.uniform(-TASK_RANGE, +TASK_RANGE)]) for i in range(TASK_SET_AMOUNT)]

    #test task to check performance
    test_task = np.array([random.uniform(-TASK_RANGE, +TASK_RANGE),random.uniform(-TASK_RANGE, +TASK_RANGE)])
    print("test_task: ", test_task)

    #from 2 github repos
    num_tasks = 20
    batch_size = 20
    max_kl=1.0e-2
    cg_iters=10
    cg_damping=1.0e-5
    ls_max_steps=15
    ls_backtrack_ratio=0.8
    iterations = 0
    while True: #only break if the performance is good, due to training
        task_batch = [random.choice(tasks) for i in range(SAMPLE_TASK_BATCH_SIZE)]
        task_training_data = [] #tuple consisting of the base trajectories and the trajectories produced after training and the trained model
        for task in task_batch:

            #create buffer model
            inner_task_policy = copy.deepcopy(policy)
            #inner_task_policy.cuda()
            #sample already updates inner loop policy and creates validation trajectories with it
            task_training_data.append(agent.sample(inner_task_policy, task,  max_steps=MAX_STEPS, inner_updates=INNER_UPDATES_PER_TASK, learning_rate=INNER_UPDATE_LEARNING_RATE, gamma=0.99, gae_lambda=1.0, device=device))
            del inner_task_policy
            """
             End inner-loop maml
            """

        """
         Now update general model
        """
        old_losses = []
        old_kls = []
        old_pis = []
        for i in range(batch_size): #batch_size
            losses, kls, pis = surrogate_loss(policy, task_training_data[i][0], task_training_data[i][1], old_pi=None, first_order=True, step_size=SURROGATE_ADAPTATION_LR)
            old_losses.append(losses)
            old_kls.append(kls)
            old_pis.append(pis)

        #used same policy for that, so that the loss can be backpropagated by the same parameters
        #loss of current policy over all tasks (normalized by task amount)
        old_loss = sum(old_losses)/num_tasks #20 tasks
        #print("old_loss: ", old_loss)
        #trpo tricks
        # surrogate loss over all tasks again, to create the loss with the gradients for the policy, which is then updated via TRPO using conjugate_gradient
        grads = torch.autograd.grad(old_loss,
                                    policy.parameters(),
                                    retain_graph=True)
        grads = parameters_to_vector(grads)
        # Compute the step direction with Conjugate Gradient
        old_kl = sum(old_kls) / num_tasks #num_tasks
        created_hessian_vector_product = hessian_vector_product(policy, old_kl,
                                                             damping=cg_damping)
        stepdir = conjugate_gradient(created_hessian_vector_product,
                                             grads,
                                             cg_iters=cg_iters)
        # Compute the Lagrange multiplier, as described in the TRPO-paper
        shs = 0.5 * torch.dot(stepdir,
                              created_hessian_vector_product(stepdir, retain_graph=False))
        lagrange_multiplier = torch.sqrt(shs / max_kl)
        step = stepdir / lagrange_multiplier
        #print("step: ", step)

        # Save the old parameters
        old_params = parameters_to_vector(policy.parameters())

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            #update, to check the new policy
            vector_to_parameters(old_params - step_size * step,
                                 policy.parameters())

            losses = []
            kls = []
            for i in range(batch_size): #batch_size
                loss, kl_div, _ = surrogate_loss(policy, task_training_data[i][0], task_training_data[i][1], old_pi=old_pis[i],first_order=True,step_size=SURROGATE_ADAPTATION_LR)
                losses.append(loss)
                kls.append(kl_div)

            improve = (sum(losses) / num_tasks) - old_loss
            kl = sum(kls) / num_tasks
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                print("improved by: ", improve.item())
                print("kl-constraint satisfied: ", kl.item())
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, policy.parameters()) #go back to old_policy


        #Don't test model in between (), use total loss as measurement)
        if iterations % 50 == 0:
            print("Iterations: ", iterations)
            evaluate_net(policy,agent, test_task,lr=EVALUATION_LR)
            torch.save(policy.state_dict(), 'results/test_HL_100_100(%d)' %iterations)
            if iterations >=50:
                print("Iterations: ", iterations)
                #show_net(policy,env,agent, test_task)

        print("iterations: ",iterations)
        iterations = iterations + 1

    """
    End infinite while-loop
    """
