import gym
import point_nav

import numpy as np
import torch
import random
import numpy as np
from numpy import array
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm #progress bar

from drl.maml import show_net, display_both_trajectories
from drl.policy import Policy
from drl.baseline import LinearFeatureBaseline
from drl.agent import Agent


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

STATE_SIZE = 2
ACTION_SIZE = 2
MAX_STEPS = 100
TASK_RANGE = 0.5

STATE_DICTS = ['test_SGD_SURRL_0.01_tanh_mu_init_scale_0.1_HL_100_100(60)',
                'test_SGD_SURRL_0.01_tanh_mu_init_scale_0.1_HL_100_100(100)',
                'test_SGD_SURRL_0.01_tanh_mu_init_scale_0.1_HL_100_100(140)',
                'basic_train_comparison_SGD_0.1_SIGMA_0.1_HL_100_100(50)']

STATE_DICTS = ['test_SGD_SURRL_0.01_tanh_mu_init_scale_0.01_topped_0.1_params_forward_HL_100_100(280)',
                'basic_train_comparison_SGD_0.01_SIGMA_0.1_HL_100_100(70)']

#STATE_DICTS = ['test_SGD_SURRL_0.01_tanh_mu_init_scale_0.01_topped_0.1_params_forward_HL_100_100(280)']

# STATE_DICTS = [ 'test_SGD_SURRL_0.01_tanh_mu_init_scale_0.01_topped_0.1_params_forward_HL_100_100(80)',
#                 'test_SGD_SURRL_0.01_tanh_mu_init_scale_0.01_topped_0.1_params_forward_HL_100_100(180)',
#                 'test_SGD_SURRL_0.01_tanh_mu_init_scale_0.01_topped_0.1_params_forward_HL_100_100(280)',
#                 'test_SGD_SURRL_0.01_tanh_mu_init_scale_0.01_topped_0.1_params_forward_HL_100_100(380)']

# STATE_DICTS = ['test_SGD_SURRL_0.01_tanh_mu_init_scale_0.1_topped_1.0_params_forward_HL_16_16(250)',
#                 'basic_train_comparison_SGD_0.01_SIGMA_0.1_HL_16_16(20)']#, #actually 0.1 sigma and 1.0 topped
                #'basic_train_comparison_SGD_0.01_SIGMA_0.1_HL_16_16(220)']
#FC_UNITS = [[16,16],[16,16]]
FC_UNITS = [[100,100],[100,100],[100,100],[100,100]]

#POLICY_NAMES = ['maml_100_100(80)', 'maml_100_100(180)','maml_100_100(280)', 'maml_100_100(380)']
POLICY_NAMES = ['maml_100_100(280)', 'pretrained_100_100(1200)']


def print_rews(logs,step_sizes="N/A", with_errorbar=True, title="maml learning",names=["maml recreation","normal pretrained","other"], colors=["blue","red","pink","black"]):
    """
    assuming [State_dict, task, steps]
    """
    fig, ax = plt.subplots(1,1)
    x = np.linspace(0, len(step_sizes[0]), num=len(step_sizes[0])+1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # to only display full integer numbers for the step amounts
    alpha = 0.7
    fig.suptitle(title)
    for i,l in enumerate(logs):
        means = np.mean(l,axis=0)
        ax.plot(x,means, colors[i], label=names[i], alpha=0.7)
        if with_errorbar:
            below = (means - np.min(l,axis=0))
            above = (np.max(l,axis=0) - means)
            #errorbar
            yerr = [below,above]
            #print("error per step: \n",yerr)
            ax.errorbar(x,means,yerr=yerr,ecolor=colors[i], fmt='o',alpha=alpha)

    #maml results as comparison
    ax.plot([-40.41, -11.68, -3.33, -3.23],"green",label="maml paper")

    ax.set_xlabel('Number of update steps')
    ax.set_ylabel('Total rewards per episode') #of consecutive updates
    plt.legend(loc="lower right")
    plt.figtext(0,0,"step_sizes: {}".format(step_sizes))
    plt.show()



TASK_AMOUNT = 20 #in githubrepo 6 from cbfinn
TASKS = [np.array([random.uniform(-TASK_RANGE, +TASK_RANGE),random.uniform(-TASK_RANGE, +TASK_RANGE)]) for i in range(TASK_AMOUNT)]
def rnd_sgn():
    return 1 if random.random() < 0.5 else -1

# OOR_TASKS = [np.array([rnd_sgn() * random.uniform(1.0, 2.0),rnd_sgn() * random.uniform(1.0, 2.0)]) for i in range(TASK_AMOUNT)]
# TASKS = OOR_TASKS

#TASKS = [array([-0.12336959, -0.48616786]), array([-0.27461975, -0.41841589]), array([ 0.47146714, -0.43439608]), array([-0.09703799, -0.38909384]), array([0.29890572, 0.4824892 ]), array([-0.15209714, -0.25235329])]
#Step_Sizes - Dict, to give every state_dict it's own step sizes array
STEP_SIZES_DICT = [ [0.0100,0.0020,0.0020,0.0020], #[0.03,0.015,0.015,0.015], works well with test_SGD_SURRL_0.01_tanh_mu_init_scale_0.1_HL_100_100(140)
                    [0.0100,0.0020,0.0020,0.0020],
                    [0.0100,0.0020,0.0020,0.0020],
                    [0.0100,0.0020,0.0020,0.0020]]

STEP_SIZES_DICT = [ [0.0100,0.0020,0.0020,0.0020],
                    [0.100,0.020,0.020,0.020]]


#STEP_SIZES_DICT *=5 #scaling
SIGMA_INIT_STD = 0.1
SIGMA_MAX_STD = 1.0


# FC1_UNITS = 16
# FC2_UNITS = 16
BATCH_SIZE = 40  #as described in the paper, use 40 for evaluation, called "samples" in paper

ACTION_WITH_DIST = True

LOGS = np.ones((len(STATE_DICTS),len(TASKS),len(STEP_SIZES_DICT[0])+1)) #+1 for the rewards of the metamodel without a gradient step

#init env
env = gym.make("PointNav-v0")
#init agent and baseline
baseline = LinearFeatureBaseline(STATE_SIZE,reg_coeff=1e-5, device=device)
agent = Agent(env, STATE_SIZE, ACTION_SIZE, baseline, batch_size = BATCH_SIZE)


# print("test_task: ",test_task)
# show_net(policy,env,agent,test_task, max_steps=MAX_STEPS,first_step_size=FIRST_STEP_SIZE, step_size=STEP_SIZE, inner_updates=INNER_UPDATES)

#basic
for i_sd,state_dict in enumerate(STATE_DICTS):
    print("Using state dict: ",state_dict)
    #for i_t,task in enumerate(TASKS):
    i_t = 0
    for task in tqdm(TASKS, desc='test tasks'): #progress bar
        #print("Using task: ",task)
        policy = Policy(STATE_SIZE, ACTION_SIZE, fc1=FC_UNITS[i_sd][0], fc2=FC_UNITS[i_sd][1], init_std=SIGMA_INIT_STD, min_std=1e-6,max_std=SIGMA_MAX_STD,device=device)
        policy.load_state_dict(torch.load('results/'+state_dict))
        policy.to(device)
        evaluate_policy_net = copy.deepcopy(policy)

        states_before,_,rewards_before = agent.sample_trajectory(evaluate_policy_net, task,action_with_dist=ACTION_WITH_DIST)
        LOGS[i_sd][i_t][0] =  sum(rewards_before) #log rewards of meta_model without gradient update
        #train
        states_after = []
        for i_sz,step_size in enumerate(STEP_SIZES_DICT[i_sd]):
            train_batch = agent.create_batchepisode(evaluate_policy_net, task,device=device) #use 40
            evaluate_policy_net.update_inner_loop_policy(train_batch, step_size=step_size, first_order=False)
            #test afterwards
            states_after,_,rewards_after = agent.sample_trajectory(evaluate_policy_net, task,action_with_dist=ACTION_WITH_DIST)
            LOGS[i_sd][i_t][i_sz+1] = sum(rewards_after)

        i_t = i_t + 1
        #display_both_trajectories(np.array(states_before), np.array(states_after),task,title="before and after")

print("STATE_DICTS: \n", STATE_DICTS)
print("Tasks: \n",TASKS)
print("STEP_SIZES_DICT: \n",STEP_SIZES_DICT)
print("logs: \n",LOGS)

print("average per step for each state_dict: \n",  np.mean(LOGS,axis=1))

print_rews(LOGS,step_sizes=STEP_SIZES_DICT, names=POLICY_NAMES )

"""
Some configs:
STEP_SIZES = [0.1,0.05,0.05,0.05,0.0]
#STEP_SIZES = [0.5,0.25,0.25,0.25]
#STEP_SIZES = [0.1,0.05,0.05,0.05]
STEP_SIZES = [0.02,0.005,0.005,0.005]

"""

"""

Nice sample :)
------
STATE_DICTS = ['test_SGD_SURRL_0.01_tanh_mu_init_scale_0.1_HL_100_100(140)',
                'basic_train_comparison_SGD_0.01_SIGMA_0.1_HL_100_100(70)']
------
Using task:  [-0.32098094 -0.38959011]
Using task:  [ 0.29748908 -0.44533778]
Using task:  [ 0.36038806 -0.09885351]
Using task:  [-0.29137576  0.46616117]
Using task:  [-0.31755594  0.03164448]
Using task:  [-0.34015445  0.32252856]
------
STEP_SIZES = [0.02,0.005,0.005,0.005]
------
average per step for each state_dict:
 [[-47.39990469 -21.03329463 -13.40638494 -10.01489813  -7.94698345]
 [-46.24693665 -38.63373229 -37.06780219 -35.17051173 -33.21283116]]

##############

STATE_DICTS:
 ['test_SGD_SURRL_0.01_tanh_mu_init_scale_0.1_HL_100_100(140)', 'basic_train_comparison_SGD_0.01_SIGMA_0.1_HL_100_100(70)']
Tasks:
 [array([-0.39978022, -0.3914635 ]), array([ 0.02493056, -0.28768951]), array([ 0.18946648, -0.03511493]), array([-0.38875429,  0.37672505]), array([-0.05720385, -0.07558795]), array([-0.17825366,  0.18822064])]
STEP_SIZES_DICT:
 [[0.03, 0.015, 0.015, 0.015], [0.1, 0.05, 0.05, 0.03]]
logs:
 [[[-60.41610674 -27.93017159 -14.08680907  -5.08007657  -0.83872426]
  [-30.70129033 -13.51595851  -8.96811224  -5.30895328  -6.1612293 ]
  [-15.53468546 -13.38671041  -3.13415424  -6.07510476  -3.27441353]
  [-56.1600084   -6.33474581  -0.85593651 -10.45307105  -7.79980735]
  [-13.81986504 -22.43505278  -6.26066083  -7.94669283  -4.86333751]
  [-27.95058744 -33.46243127  -0.14161228  -7.29629595 -12.0092781 ]]

 [[-58.83891622 -13.85997243  -0.8417123   -0.83872426  -0.83872426]
  [-34.44266892 -13.10518177 -10.94522907  -6.48917631  -6.48770261]
  [-22.42595333 -23.01000077  -7.74073401  -7.89170012  -7.10167682]
  [-49.30738565  -6.85146069 -16.91631244 -12.857399    -6.24305436]
  [-13.65668642  -8.61197046  -0.14554204  -8.06204466  -8.65589633]
  [-21.07422966 -27.2289994   -0.99000399  -6.54757121  -3.65356942]]]
average per step for each state_dict:
 [[-34.09709057 -19.51084506  -5.57454753  -7.02669907  -5.82446501]
 [-33.29097337 -15.44459758  -6.26325564  -7.11443593  -5.49677063]]


#############################
STATE_DICTS:
 ['test_SGD_SURRL_0.01_tanh_mu_init_scale_0.1_HL_100_100(140)', 'basic_train_comparison_SGD_0.01_SIGMA_0.1_HL_100_100(70)']
Tasks:
 [array([0.21465557, 0.14069711]), array([0.01658062, 0.35751142]), array([-0.03145228,  0.40233176]), array([ 0.24619862, -0.11079361]), array([-0.35816354, -0.40146931]), array([-0.17800531,  0.11373725])]
STEP_SIZES_DICT:
 [[0.034, 0.015, 0.01, 0.01], [0.1, 0.05, 0.03, 0.03]]
logs:
 [[[-20.99306023 -10.32000242  -5.41319521  -4.28791713  -5.25194715]
  [-34.00155132 -28.87058786  -4.8252632   -6.14821024  -2.64694165]
  [-39.10546952 -24.23812848 -15.2963661   -5.39401323  -6.26564382]
  [-24.03211711 -34.89499578 -15.01616673  -4.58243436  -1.75159279]
  [-58.14878837 -20.96032197 -15.80513884  -8.86363105  -0.779224  ]
  [-24.14326543 -28.96882362  -0.11097244  -5.89033105  -0.12625119]]

 [[-24.66089047 -34.49107269  -0.34481646  -2.0152855   -4.04583578]
  [-30.50770998 -23.33079698  -5.24745519  -4.34200792  -0.54477406]
  [-34.84842176 -21.16970461 -10.42225694  -3.06733813  -4.31808026]
  [-30.99003456  -8.27288096  -4.4572928   -5.37922631  -7.17037534]
  [-57.00774652 -15.66781448 -17.96786874  -9.23314867  -0.77966612]
  [-17.3146374   -2.9308353   -0.32351716  -3.45360257  -0.40483652]]]
average per step for each state_dict:
 [[-33.40404199 -24.70881002  -9.41118375  -5.86108951  -2.8036001 ]
 [-32.55490678 -17.64385084  -6.46053455  -4.58176818  -2.87726135]]

##################
STATE_DICTS:
 ['test_SGD_SURRL_0.01_tanh_mu_init_scale_0.1_HL_100_100(140)', 'basic_train_comparison_SGD_0.01_SIGMA_0.1_HL_100_100(70)']
Tasks:
 [array([-0.04238094, -0.39270178]), array([ 0.33899691, -0.06574738]), array([0.00807824, 0.28999727]), array([-0.01071798,  0.12601209]), array([-0.16902611, -0.00355174]), array([-0.04188643,  0.39570544])]
STEP_SIZES_DICT:
 [[0.03, 0.015, 0.01, 0.01], [0.1, 0.05, 0.03, 0.03]]
logs:
 [[[-42.03513007 -22.54576493 -10.76657893  -4.68121877  -0.59456592]
  [-30.70371949 -17.76367092  -4.57679704  -6.07458706  -0.4628333 ]
  [-27.37754889 -25.87364355  -5.11372366  -0.3209318   -0.4994345 ]
  [-12.06660337 -16.82209427  -8.56013441  -6.05550234  -0.08740326]
  [-21.40412539 -38.54768393  -9.3293493  -10.81991656  -5.13990783]
  [-38.6735203  -21.44931815 -12.19153933  -9.899359    -8.82714846]]

 [[-44.78000157  -6.71506507 -17.53905013  -6.85494565  -2.05090882]
  [-37.48154041  -5.53317176 -10.13933643  -3.42649052  -3.1486679 ]
  [-23.70702318 -27.99650539  -2.30947632  -0.60945775  -0.47161853]
  [ -7.2330998  -19.90990124  -0.74857965  -0.53197748  -0.66267013]
  [-16.43184899  -8.62958751 -10.31683923  -7.66683877  -1.63692968]
  [-34.24973154 -24.27780892 -20.41392221 -14.76704662 -13.07912478]]]
average per step for each state_dict:
 [[-28.71010792 -23.83369596  -8.42302044  -6.30858592  -2.60188221]
 [-27.31387425 -15.51033998 -10.24453399  -5.6427928   -3.50831997]]

####################
VERY nice:

STATE_DICTS:
 ['test_SGD_SURRL_0.01_tanh_mu_init_scale_0.1_HL_100_100(140)', 'basic_train_comparison_SGD_0.01_SIGMA_0.1_HL_100_100(70)']
Tasks:
 [array([0.30098707, 0.16970193]), array([-0.16468093, -0.4935084 ]), array([0.46777696, 0.2026589 ]), array([ 0.39864573, -0.27459401]), array([ 0.29745336, -0.47432266]), array([ 0.30055365, -0.3642191 ]), array([ 0.37027932, -0.21032377]), array([-0.36798001,  0.03323969]), array([-0.28372791,  0.39676123]), array([ 0.36891271, -0.21515113]), array([-0.20344162,  0.18334366]), array([0.36613029, 0.07072671]), array([-0.44536986,  0.45278537]), array([-0.39890194, -0.36648808]), array([0.32501298, 0.30276335]), array([-0.39196097,  0.48613604]), array([-4.55580834e-05,  1.19446301e-01]), array([0.05797236, 0.19157366]), array([0.24258822, 0.29619929]), array([0.31155152, 0.2629673 ])]
STEP_SIZES_DICT:
 [[0.035, 0.015, 0.01, 0.01], [0.1, 0.05, 0.03, 0.03]]
logs:
 [[[-2.98222474e+01 -1.39153045e+01 -9.66174878e+00 -3.37451468e+00
   -4.61627871e+00]
  [-5.53019611e+01 -1.72322204e+01 -1.23412819e+01 -7.46819710e+00
   -5.94118162e+00]
  [-4.62142939e+01 -2.48163971e+01 -1.09981083e+01 -7.00959471e+00
   -4.90195341e+00]
  [-4.60369658e+01 -9.17939638e+00 -8.38416971e+00 -6.60524576e-01
   -6.59124346e-01]
  [-5.54453882e+01 -9.38110537e+00 -6.51576700e+00 -5.40631163e+00
   -5.41355126e+00]
  [-4.61045947e+01 -1.52927430e+01 -1.07604903e+01 -7.13490620e+00
   -4.63443016e+00]
  [-3.98806718e+01 -1.39676882e+01 -5.88381632e-01 -4.47595434e+00
   -1.06886639e+00]
  [-4.12095329e+01 -2.94937576e+01 -1.10758261e+01 -5.27946285e+00
   -3.16074887e+00]
  [-5.00313189e+01 -2.12756763e+01 -1.66318360e+01 -7.17074347e+00
   -4.70373711e+00]
  [-4.00488644e+01 -1.89608613e+01 -6.74978211e+00 -4.63263983e+00
   -3.89847785e+00]
  [-2.97174868e+01 -3.90480245e+01 -4.44525644e+00 -1.39633273e-01
   -3.71970577e+00]
  [-3.26477102e+01 -1.64313360e+01 -8.46357645e+00 -4.26006545e+00
   -5.27648370e+00]
  [-6.54093093e+01 -4.93737102e+00 -1.27043817e+01 -3.13521879e+00
   -3.22942585e+00]
  [-5.86829293e+01 -1.70507195e+01 -8.47243739e+00 -7.84498553e-01
   -5.72303461e+00]
  [-3.99718828e+01 -1.00070901e+01 -6.79458217e+00 -5.36984337e+00
   -3.32321109e+00]
  [-6.39144730e+01 -8.79971578e+00 -3.82458837e+00 -1.13974580e+00
   -1.21039886e+00]
  [-1.10194859e+01 -3.88856097e+01 -1.27646806e+01 -2.49804504e+00
   -5.63307668e+00]
  [-1.73610170e+01 -1.27945089e+01 -1.53586438e+01 -4.71183750e+00
   -1.09670160e-01]
  [-3.41376619e+01 -2.12166689e+01 -8.02298911e+00 -3.80241568e+00
   -5.24456970e-01]
  [-3.62431448e+01 -1.16368166e+01 -4.39413755e-01 -3.89853956e+00
   -5.56539904e+00]]

 [[-3.37532714e+01 -1.85101107e+01 -7.06594871e+00 -2.93628062e+00
   -7.41746690e+00]
  [-5.67685273e+01 -1.66115847e+01 -6.44624638e+01 -3.73713750e+01
   -2.75439080e+01]
  [-5.06103877e+01 -1.24276612e+01 -1.76113808e+01 -5.73380876e+00
   -6.25526810e+00]
  [-5.29210974e+01 -6.41170792e+00 -1.37277571e+01 -2.57423884e+00
   -1.00696633e+01]
  [-6.14505253e+01 -1.47096150e+01 -4.70370525e+01 -9.62482622e+00
   -1.10309715e+01]
  [-5.24697284e+01 -3.10638785e+00 -9.69526313e+00 -1.63194507e+01
   -2.71281636e+00]
  [-4.68213613e+01 -6.74550264e+00 -1.52518293e+01 -8.90321729e+00
   -6.52190443e+00]
  [-3.53020406e+01 -9.28741870e+00 -4.57632392e+00 -4.49887922e+00
   -4.50239685e+00]
  [-4.34881640e+01 -1.67411234e+01 -1.44260858e+01 -5.65908897e+00
   -5.47208887e+00]
  [-4.69836644e+01 -7.10158260e+00 -3.46108383e+01 -8.91191991e+00
   -5.17570914e+00]
  [-2.27944026e+01 -1.97024318e+02 -7.37671126e+01 -5.15972332e+01
   -3.62907461e+01]
  [-3.82613157e+01 -1.20316369e+01 -1.61819982e+01 -1.45585664e+01
   -1.21025699e+01]
  [-5.85988940e+01 -1.50849278e+01 -3.13488866e+01 -5.41560807e+00
   -1.49809777e+01]
  [-5.69115319e+01 -1.18615196e+01 -1.01740401e+01 -5.32407374e+00
   -4.89051456e+00]
  [-4.22062996e+01 -1.55790551e+01 -1.22458721e+01 -5.11496154e+00
   -6.59795693e+00]
  [-5.72744422e+01 -5.92327520e+00 -1.87253591e+01 -7.21674914e+00
   -5.75547903e+00]
  [-6.76781969e+00 -3.27976110e+01 -1.57026489e+01 -5.51142075e+00
   -3.78071478e-01]
  [-1.56277243e+01 -2.92573362e+01 -2.99564134e+01 -1.46625745e+01
   -1.04420079e+01]
  [-3.54261822e+01 -2.01860539e+01 -6.64988759e+00 -6.94048398e+00
   -5.08676633e+00]
  [-3.88525744e+01 -2.77171174e+01 -1.29715893e+01 -8.63126281e+00
   -6.54363633e+00]]]
average per step for each state_dict:
 [[-41.960047   -17.71615055  -8.74989708  -4.11763462  -3.66566062]
 [-42.66449772 -23.95577728 -22.80943756 -11.37530098  -9.48854599]]

"""
