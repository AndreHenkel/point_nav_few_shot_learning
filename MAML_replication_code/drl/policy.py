"""
Code taken from: https://github.com/tristandeleu/pytorch-maml-rl/blob/master/maml_rl/policies/
and slightly adapted

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal
import math

import numpy as np

from collections import OrderedDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

"""
Taken from: https://github.com/tristandeleu/pytorch-maml-rl/blob/master/maml_rl/utils/torch_utils.py
"""
def weighted_mean(tensor, lengths=None):
    if lengths is None:
        return torch.mean(tensor)
    if tensor.dim() < 2:
        raise ValueError('Expected tensor with at least 2 dimensions '
                         '(trajectory_length x batch_size), got {0}D '
                         'tensor.'.format(tensor.dim()))
    for i, length in enumerate(lengths):
        tensor[length:, i].fill_(0.)

    extra_dims = (1,) * (tensor.dim() - 2)
    lengths = torch.as_tensor(lengths, dtype=torch.float32)

    out = torch.sum(tensor, dim=0)
    out.div_(lengths.view(-1, *extra_dims).to(device))

    return out

"""
Taken from: https://github.com/tristandeleu/pytorch-maml-rl/blob/master/maml_rl/utils/torch_utils.py
"""
def weighted_normalize(tensor, lengths=None, epsilon=1e-8):
    mean = weighted_mean(tensor, lengths=lengths)
    out = tensor - mean.mean()
    for i, length in enumerate(lengths):
        out[length:, i].fill_(0.)

    std = torch.sqrt(weighted_mean(out ** 2, lengths=lengths).mean())
    out.div_(std + epsilon)

    return out

class Policy(nn.Module):
    def __init__(self, state_size, action_size, fc1=100, fc2=100, init_std=1.0, min_std=1e-6, max_std=1.0, optim_lr=0.0001, device='cpu'):
        super(Policy, self).__init__()
        self.device=device
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, fc1).to(self.device)
        self.fc2 = nn.Linear(fc1, fc2).to(self.device)
        self.mu = nn.Linear(fc2, action_size).to(self.device)

        self.min_log_std = math.log(min_std)
        self.max_log_std = math.log(max_std)#don't go higher than this
        #self.sigma = nn.Parameter(torch.Tensor(action_size))
        self.register_parameter(name='sigma', param=nn.Parameter(torch.Tensor(action_size)))
        self.sigma.data.fill_(math.log(init_std))
        self.apply(weight_init)

        self.optim = torch.optim.Adam(self.parameters(),lr=optim_lr)

                # For compatibility with Torchmeta
        self.named_meta_parameters = self.named_parameters
        self.meta_parameters = self.parameters
        #self.low_scale = torch.tensor([0.1,0.1]).to(device)


    def update_inner_loop_policy(self, episode_batch, step_size=0.01, momentum=0.9,first_order=False):
        """Apply one step of gradient descent on the loss function `loss`, with
        step-size `step_size` and updates parameters
        """
        #optimizer = torch.optim.Adam(self.parameters(),lr=step_size)
        optimizer = torch.optim.SGD(self.parameters(),lr=step_size, momentum=momentum)

        optimizer.zero_grad()
        #calculate loss
        obs = episode_batch.observations.view((-1, *episode_batch.observation_shape))

        policy_dist = self.forward(obs.float().unsqueeze(0).to(self.device))

        log_probs = policy_dist.log_prob(episode_batch.actions.view((-1, *episode_batch.action_shape)).float().unsqueeze(0).to(self.device))
        log_probs = log_probs.view(len(episode_batch), episode_batch.batch_size)
        losses = -weighted_mean(log_probs.to(device) * episode_batch.advantages, lengths = episode_batch.lengths)

        mean_loss = losses.mean()
        #print("mean_loss: ",mean_loss)
        mean_loss.backward()
        optimizer.step()


    def get_action(self, state, dist=True):
        """
            Returns an action depending on the given state with the current policy.
            If dist is True, then use a sampled action from the distribution.
            If false, use the mean value used to create this distribution.
        """
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            policy_distribution = self.forward(state_tensor)
            if not dist:
                return policy_distribution.base_dist.loc.detach().data.cpu().numpy().flatten()
            action_tensor = policy_distribution.sample()
            action = action_tensor.data.cpu().numpy()
            #action /= 10 #dividing by then, since tanh outputs [-1,1]
            return action.flatten()

    def update_params(self, loss, step_size=0.1, first_order=False):
        """Apply one step of gradient descent on the loss function `loss`, with
        step-size `step_size`, and returns the updated parameters of the neural
        network.
        """

        grads = torch.autograd.grad(loss, self.parameters(),
                                    create_graph=not first_order)

        updated_params = OrderedDict()
        for (name, param), grad in zip(self.named_parameters(), grads):
            updated_params[name] = param - step_size * grad

        return updated_params

    # def forward(self,x,params): #just with using other parameters
    #     x = F.relu(F.linear(x,weight=params['layer1.weight'],bias=params['layer1.bias']))
    #     x = F.relu(F.linear(x,weight=params['layer2.weight'],bias=params['layer2.bias']))
    #     mu = torch.tanh(F.linear(x,weight=params['mu.weight'],bias=params['mu.bias']))
    #     scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std, max=self.max_log_std))
    #     return Independent(Normal(loc=mu, scale=scale), 1)

    def forward(self, x, params=None):
        """
        Params addition is needed, since for the meta update we need to separate the created parameters from the base model to improve it.
        Otherwise we could also deep copy it every time. Maybe TODO.
        """
        if params is None:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            mu = self.mu(x)
            mu = torch.tanh(mu)
            #print("sigma: ",self.sigma)
            # print("params: ",self.parameters)
            scale = torch.exp(torch.clamp(self.sigma, min=self.min_log_std, max=self.max_log_std))
            #print("scale: ",scale)
            #print("mu: ",mu)
            return Independent(Normal(loc=mu, scale=scale), 1)
        else:
            x = F.relu(F.linear(x,weight=params['fc1.weight'],bias=params['fc1.bias']))
            x = F.relu(F.linear(x,weight=params['fc2.weight'],bias=params['fc2.bias']))
            mu = torch.tanh(F.linear(x,weight=params['mu.weight'],bias=params['mu.bias']))
            scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std, max=self.max_log_std))
            return Independent(Normal(loc=mu, scale=scale), 1)
