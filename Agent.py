import numpy as np
import random
from collections import namedtuple, deque

from Model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, double_agent=False,dueling_agent=False,prioritized_memory=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
	    double_agent(bool) : True if we want to use DDQN
            dueling_agent (bool): True if we want to use Dueling
            prioritized_memory(bool) : True if we want to use prioritized memory
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.double_agent=double_agent
        self.dueling_agent=dueling_agent
        self.prioritized_memory=prioritized_memory
        self.qnetwork_local = QNetwork(state_size, action_size, seed,dueling_agent=dueling_agent).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed,dueling_agent=dueling_agent).to(device)
        self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory : if we use prioritized memory, we need to compute the delta first
        if self.prioritized_memory:
                with torch.no_grad():
                        state_pyt=torch.from_numpy(np.vstack([state])).float().detach().to(device)
                        next_state_pyt=torch.from_numpy(np.vstack([next_state])).float().detach().to(device)
                        reward_pyt=torch.from_numpy(np.vstack([reward])).float().detach().to(device)
                        done_pyt = torch.from_numpy(np.vstack([done]).astype(np.uint8)).float().detach().to(device)
                        Q_target=self.get_q_target(next_state_pyt,reward_pyt,GAMMA,done_pyt,1)
                        action_pyt = torch.from_numpy(np.vstack([action])).long().detach().to(device)
                        Q_expected = self.qnetwork_local(state_pyt).gather(1, action_pyt)
                        delta=Q_target-Q_expected
                        self.memory.add(state, action, reward, next_state, done,delta.detach())
        else :
                self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(self.prioritized_memory)
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def weighted_mse_loss(self,Q_expected, Q_targets,deltas):
        """ Returns the weighted mean square error between Q_expected and Q_target
  
        Params
        ======
            Q_expected, Q_targets : target and current guesses
            deltas : weights
        """
        weight =( deltas/torch.sum(deltas)*BATCH_SIZE )** (-1)
        return torch.mean(weight * (Q_expected - Q_targets) ** 2)

    def get_q_target(self,next_states,rewards,gamma,dones):
        """ Returns the target expected Q value  
  
        Params
        ======
            next_states : list of states we arrived in
            rewards : rewards we got
            gamma : discounting factor
            dones : list of bool telling if the episode is done
        """
        # Get max predicted Q values (for next states) from target model
        if (not self.double_agent):
                Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        else :
                indices= torch.argmax(self.qnetwork_local(next_states).detach(),1)
                Q_targets_next = self.qnetwork_target(next_states).detach().gather(1,indices.unsqueeze(1))
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        return Q_targets

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones,deltas = experiences

        Q_targets = self.get_q_target(next_states,rewards,gamma,dones)

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss

        loss = self.weighted_mse_loss(Q_expected, Q_targets,deltas)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed,min_prob_value=0.01):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done","delta"])
        self.seed = random.seed(seed)
        self.min_prob_value = min_prob_value
        self.a=0.0
    def add(self, state, action, reward, next_state, done,delta=1.):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done,np.abs(delta))
        self.memory.append(e)
    
    def sample(self,prioritized=False):
        """Randomly sample a batch of experiences from memory."""
        if prioritized :
                p= np.array([(e.delta+self.min_prob_value)**self.a for e in self.memory])
                experiences = random.choices(self.memory, k=self.batch_size,weights=p/np.sum(p))
        else :
                experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        deltas = torch.from_numpy(np.vstack([e.delta for e in experiences if e is not None])).float().to(device)
        return (states, actions, rewards, next_states, dones,deltas)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
