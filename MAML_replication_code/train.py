

import torch
from drl.maml import MAML

#allows to use already trained state_dict to continue training on it
#STATE_DICT = torch.load("results/Some_state_dict_to_train_upon-Needs_same_hidden_sizes")
STATE_DICT = None #override
MAML(STATE_DICT)
