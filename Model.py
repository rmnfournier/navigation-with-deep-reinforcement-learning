## Modified from template provided by udacity
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed,fc1_units=128,fc2_units=64,dueling_agent=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            dueling_agent(bool): True if we want to use the dueling architecture
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1= nn.Linear(state_size, fc1_units)
        self.fc2=nn.Linear(fc1_units,fc2_units)

        self.advantage=nn.Linear(fc2_units,action_size)
        self.state_value=nn.Linear(fc2_units,1)
        self.dueling_agent=dueling_agent

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x=F.relu(self.fc1(state))
        x=F.relu(self.fc2(x))
        if self.dueling_agent:
                advantage=self.advantage(x)
                value=self.state_value(x)
                return advantage+value 
        else : 
               return self.advantage(x)     

        
