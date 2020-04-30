import argparse

import gym
import numpy as np

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='path to model')
parser.add_argument('--episodes', type=int, default=100, help='number of episodes')
parser.add_argument('--epsilon', type=float, default=0.01, help='exploration temperature')

args = parser.parse_args()

# Environment (a Markov Decision Process model)
env, statesize, actionsize = gym.make('MountainCar-v0'), 2, 3

# Q Model
model = utils.loadmodel(args.model, env, statesize, actionsize)
print("Model: {}".format(model))

# Rollout
_, rewards = utils.rollout(env, model, args.episodes, args.epsilon, render=True)

# Report
#Evaluate total rewards for MountainCar environment
score = np.array([np.array(rewards) > -200.0]).sum()
print('Score: ' + str(score) + '/' + str(args.episodes))
