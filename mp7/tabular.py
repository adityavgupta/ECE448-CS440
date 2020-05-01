import math

import gym
import numpy as np
import torch

import utils
from policies import QPolicy


class TabQPolicy(QPolicy):
    def __init__(self, env, buckets, actionsize, lr, gamma, model=None):
        """
        Inititalize the tabular q policy

        @param env: the gym environment
        @param buckets: specifies the discretization of the continuous state space for each dimension
        @param actionsize: dimension of the descrete action space.
        @param lr: learning rate for the model update 
        @param gamma: discount factor
        @param model (optional): Stores the Q-value for each state-action
            model = np.zeros(self.buckets + (actionsize,))
            
        """
        super().__init__(len(buckets), actionsize, lr, gamma)
        self.env = env
        self.buckets = buckets
        self.model = np.zeros(self.buckets + (actionsize,)) if model is None else model
        self.lr = lr
        self.gamma = gamma
        self.actionsize = actionsize
        self.N_table = dict()

    def discretize(self, obs):
        """
        Discretizes the continuous input observation

        @param obs: continuous observation
        @return: discretized observation  
        """
        upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
        lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def qvals(self, states):
        """
        Returns the q values for the states.

        @param state: the state
        
        @return qvals: the q values for the state for each action. 
        """
        qvals = np.zeros((1,3))
        d_state = self.discretize(states[0])
        qvals[0][0] = self.model[d_state + (0,)]
        qvals[0][1] = self.model[d_state + (1,)]
        qvals[0][2] = self.model[d_state + (2,)]
        return qvals

    def td_step(self, state, action, reward, next_state, done):
        """
        One step TD update to the model

        @param state: the current state
        @param action: the action
        @param reward: the reward of taking the action at the current state
        @param next_state: the next state after taking the action at the
            current state
        @param done: true if episode has terminated, false otherwise
        @return loss: total loss the at this time step
        """
        d_curr_state = self.discretize(state)
        q_vals = self.model[d_curr_state+ (action,)]

        #decayed learning rate
        self.N_table[q_vals] = self.N_table.get(q_vals, 0) + 1
        C = 0.01
        self.lr = C/(C+self.N_table[q_vals])

        d_next_state = self.discretize(next_state)
        
        if (done == True) and (next_state[0] == self.env.goal_position):
            reward = 1.0
            target = reward
        else:
            target = reward+ self.gamma*max(self.model[d_next_state+ (0,)], self.model[d_next_state+ (1,)], self.model[d_next_state + (2,)])

        self.model[d_curr_state + (action,)] = q_vals + self.lr*(target - q_vals)
        loss = (q_vals - target)**2
        return loss
    def save(self, outpath):
        """
        saves the model at the specified outpath
        """
        torch.save(self.model, outpath)


if __name__ == '__main__':
    args = utils.hyperparameters()

    env = gym.make('MountainCar-v0')

    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n
    policy = TabQPolicy(env, buckets=(10,10), actionsize=actionsize, lr=args.lr, gamma=args.gamma)

    utils.qlearn(env, policy, args)

    torch.save(policy.model, 'models/tabular.npy')
