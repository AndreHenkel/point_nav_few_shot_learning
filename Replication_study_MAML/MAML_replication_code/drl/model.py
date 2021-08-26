import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

#First attempt, but then withdrawn from it to try another implementation, which performed better
##decide weather to use gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device="cpu"#hotfix, due to error: RuntimeError: Expected object of device type cuda but got device type cpu for argument #1 'self' in call to _th_addmm

LEARNING_RATE = 0.01 #fast learning rate 0.1, but 2nd loss is really high in this setting
#REPORT: high lr let's the network change significantly
ENTROPY_BETA = 1e-4

class Network(nn.Module):
    """ According to MAML Paper: 5.3 Reinforcement Learning "[...] the model trained
        by MAML is a neural network policy with two hidden layers of size 100, with ReLU nonlinearities"

        Model structure of mu,var,value taken from https://www.youtube.com/watch?v=kWHSH2HgbNQ
    """

    def __init__(self, state_size, action_size, seed=4892, fc1_units=100, fc2_units=100, learning_rate=3e-4):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.state_size = state_size
        self.action_size = action_size

        #need to use ".cuda()" to use gpu with these layers
        self.fc1 = nn.Linear(state_size, fc1_units, bias=True).cuda()
        self.fc2 = nn.Linear(fc1_units, fc2_units, bias=True).cuda()

        self.mu = nn.Linear(fc2_units, action_size, bias=True).cuda() # the movement/actions
        self.var = nn.Linear(fc2_units,action_size, bias=True).cuda() # the variance
        self.value = nn.Linear(fc2_units, 1, bias=True).cuda() # value

        #TODO: Maybe use another optimizer, like SGD for this purpose
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.reset_parameters()


    def reset_parameters(self):
        """
            It is not explicitly stated how the neural networks were initialized by the authors of the paper.
            We are using xavier_uniform here.
            The used gain of 1.0 is the standard value
        """
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)

        nn.init.xavier_uniform_(self.mu.weight, gain=1.0)
        nn.init.xavier_uniform_(self.var.weight, gain=1.0)
        nn.init.xavier_uniform_(self.value.weight, gain=1.0)

    def forward(self, x):
        """
            Return mu,var,value as tensor with grad_fn for backprop
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = torch.tanh(self.mu(x))/10 #TODO: check if it's fine like that
        var = F.softplus(self.var(x))
        value = self.value(x)
        return mu, var, value

    def get_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mu_v, var_v, _ = self.forward(state_tensor)
        # transform data to numpy to work with it
        mu = mu_v.data.cpu().numpy()
        # from the variance take an action from the normal distribution
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        action = np.random.normal(mu,sigma)
        action = np.clip(action, -0.1,0.1) #TODO: check if the network should output already values betwenn -0.1 and 0.1
        return action.flatten()

    def get_action_abs(self,state):
        """
            The difference to get_action here is that this method takes the actual value and doesn't care about the variance
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mu_v, var_v, _ = self.forward(state_tensor)
        # transform data to numpy to work with it
        mu = mu_v.data.cpu().numpy()
        action = np.clip(mu, -0.1,0.1) #TODO: check if the network should output already values betwenn -0.1 and 0.1
        return action.flatten()

    def update_policy(self, episodes):
        """
        Taken from: https://github.com/colinskow/move37/blob/master/actor_critic/lib/common.py
        """
        losses = []


        for episode in episodes:
            self.optimizer.zero_grad() #clear gradients for next backprop
            # TODO: Not sure about the formatting here
            mu_v, var_v, value_v = self.forward(episode.states)
            loss_value_v = F.mse_loss(value_v.squeeze(-1), episode.q_vals)

            #advantage
            adv_v = episode.q_vals.unsqueeze(dim=-1) - value_v.detach()
            log_prob_v = adv_v * self._calc_logprob(mu_v, var_v, episode.actions)
            loss_policy_v = -log_prob_v.mean()
            entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*var_v) + 1)/2).mean()

            loss = loss_policy_v + entropy_loss_v + loss_value_v
            losses.append(loss)
            loss.backward()
            self.optimizer.step()

        total_loss = sum(losses) / len(losses)
        print("total_loss: ", total_loss)
        #total_loss.backward()


    def meta_update_policy(self,sar,maml_loss,lr=0.01):
        states, actions, discounted_rewards = sar
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9) #SGD as mentioned in maml paper
        #states = torch.from_numpy(states).float().unsqueeze(0).to(device)
        #actions = torch.from_numpy(actions).float().unsqueeze(0).to(device)
        #discounted_rewards = torch.from_numpy(discounted_rewards).float().unsqueeze(0).to(device)
        optimizer.zero_grad() #clear gradients for next backprop

        mu_v, var_v, value_v = self.forward(states)
        loss_value_v = F.mse_loss(value_v.squeeze(-1), discounted_rewards)

        #advantage
        adv_v = discounted_rewards.unsqueeze(dim=-1) - value_v.detach()
        log_prob_v = adv_v * self._calc_logprob(mu_v, var_v, actions)
        loss_policy_v = -log_prob_v.mean()
        entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*var_v) + 1)/2).mean()

        loss = loss_policy_v + entropy_loss_v + loss_value_v
        with torch.no_grad(): #this change shouldn't change any gradient, therefore no_grad
            loss.set_(torch.FloatTensor([maml_loss]).to(device)[0])
        #change value of loss with maml loss
        # loss[0] = maml_loss[0] ~ something like that. But keep the gradient graphs intact

        loss.backward()
        optimizer.step()

    def evaluation_update_policy(self,sar,lr=0.1):
        states, actions, discounted_rewards = sar
        ev_optimizer = optim.SGD(self.parameters(), lr=lr)
        #states = torch.from_numpy(states).float().unsqueeze(0).to(device)
        #actions = torch.from_numpy(actions).float().unsqueeze(0).to(device)
        #discounted_rewards = torch.from_numpy(discounted_rewards).float().unsqueeze(0).to(device)
        ev_optimizer.zero_grad() #clear gradients for next backprop

        mu_v, var_v, value_v = self.forward(states)
        loss_value_v = F.mse_loss(value_v.squeeze(-1), discounted_rewards)

        #advantage
        adv_v = discounted_rewards.unsqueeze(dim=-1) - value_v.detach()
        log_prob_v = adv_v * self._calc_logprob(mu_v, var_v, actions)
        loss_policy_v = -log_prob_v.mean()
        entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*var_v) + 1)/2).mean()

        loss = loss_policy_v + entropy_loss_v + loss_value_v
        loss.backward()
        ev_optimizer.step()

    # def update_policy_with_loss(self, loss):
    #     self.optimizer.zero_grad() #clear gradients for next backprop
    #     loss.backward()
    #     self.optimizer.step()

    def get_loss(self, states, actions, discounted_rewards):
        self.optimizer.zero_grad() #clear gradients for next backprop
        mu_v, var_v, value_v = self.forward(states)
        loss_value_v = F.mse_loss(value_v.squeeze(-1), discounted_rewards)

        #advantage
        adv_v = discounted_rewards.unsqueeze(dim=-1) - value_v.detach()
        log_prob_v = adv_v * self._calc_logprob(mu_v, var_v, actions)
        loss_policy_v = -log_prob_v.mean()
        entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*var_v) + 1)/2).mean()

        loss = loss_policy_v + entropy_loss_v + loss_value_v
        return loss


    def _calc_logprob(self,mu_v, var_v, actions_v):
        p1 = - ((mu_v - actions_v) ** 2) /(2*var_v.clamp(min=1e-3))
        p2 = -torch.log(torch.sqrt(2*math.pi * var_v))
        return p1 + p2

    def get_deep_copy(self):
        buffer_model = type(self)(self.state_size,self.action_size)
        buffer_model.load_state_dict(self.state_dict())#copy weights etc.
        return buffer_model
