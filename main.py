###########################
#
#   @author mckvg
#   @author olin322
#
###########################

import CONSTANTS
from envCube import envCube
import gym
from gym import spaces
from gym import spaces
from gym.spaces.space import Space
from gym.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, DDPG, HerReplayBuffer, DQN
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback


env = envCube()
check_env(env)

eposides = 1
for ep in range(eposides):
    obs = env.reset()
    done = False
    rewards = 0
    while not done:
        # action = env.action_space.sample()
        action = 0
        # action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        env.render()
        rewards += reward
        # if reward < -100 or reward > 100:
        #   print(reward)

    print(rewards)
