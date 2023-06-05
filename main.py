###########################
#
#   @author mckvg
#   @author olin322
#
###########################

import CONSTANTS
import socket
import struct
import select
import time
import ctypes
import json
from envCube import envCube
import gym
from gym import spaces
from gym import spaces
from gym.spaces.space import Space
from gym.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, DDPG, HerReplayBuffer, DQN
from client import *
from CONSTANTS import SCALE
import math
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback


env = envCube()
check_env(env)

# TCPClient = client()

# Instantiate the agent
model = DQN(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log='./logs',
    learning_rate=5e-3,
)
print(model.policy)

# Callback
eval_callback = EvalCallback(env, best_model_save_path="./logs/BestModel0605_01/",
                             log_path="./logs/BestModel0605_01/", eval_freq=500,
                             deterministic=True, render=False)

# Train the agent and display a progress bar
model.learn(
    total_timesteps=int(3e6),
    tb_log_name='Sumo_pattern2_turn_left_DQN_alpha_5e-3_3M_call_1',
    progress_bar=True,
    callback=eval_callback
)

# Save the agent
model.save("Sumo_pattern2_turn_left_DQN_alpha_5e-3_3M_call_1")
del model  # delete trained model to demonstrate loading
#
# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
model = DQN.load("Sumo_pattern2_turn_left_DQN_alpha_5e-3_3M_call_1", env=env)

print(model.policy)


# model = DQN.load("./logs/BestModel0605_01/best_model", env=env)

# # Evaluate the agent
# # NOTE: If you use wrappers with your environment that modify rewards,
# #       this will be reflected here. To evaluate with original rewards,
# #       wrap environment in a "Monitor" wrapper before other wrappers.
# mean_reward, std_reward = evaluate_policy(
#     model,
#     model.get_env(),
#     deterministic = True,
# #     render = True,
# #     n_eval_episodes=10,
# # )
# # print(mean_reward, std_reward)
#
# # best_model = A2C.load("./logs/BestModel/best_model.zip",env=env)
# #
# # best_mean_reward, best_std_reward = evaluate_policy(
# #     best_model,
# #     best_model.get_env(),
# #     deterministic = True,
# #     render = True,
# #     n_eval_episodes=10,
# # )
# # print(best_mean_reward, best_std_reward)

eposides = 1
for ep in range(eposides):
    obs = env.reset()
    done = False
    rewards = 0
    # step = 0

    while not done:
        # print(env.episode_step)
        # action = env.action_space.sample()
        # if 60 <= env.episode_step <= 65:
        #     action = 3
        # else:
        #     action = 0

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        rewards += reward

        # step += 1
        # action = env.action_space.sample()
        # if 60 <= step <= 65:
        #     action = 3
        # else:
        #     action = 0
        # action = 0

        # if reward < -100 or reward > 100:
        #   print(reward)

    print(rewards)
    # print(step)

    # tensorboard --logdir ./logs
