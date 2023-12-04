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
# check_env(env)

# # TCPClient = client()
#
# Instantiate the agent
model = DQN(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log='./logs',
    learning_rate=1e-2,
)
print(model.policy)

# # Callback
# eval_callback = EvalCallback(env, best_model_save_path="./logs/BestModel0418_01/",
#                              log_path="./logs/BestModel0418_01/", eval_freq=500,
#                              deterministic=True, render=False)
#
# # Train the agent and display a progress bar
# model.learn(
#     total_timesteps=int(1e5),
#     tb_log_name='Sumo_pattern1_straight_DQN_alpha_1e-2_100k_call_1',
#     progress_bar=True,
#     callback=eval_callback
# )
#
# # Save the agent
# model.save("Sumo_pattern1_straight_DQN_alpha_1e-2_100k_call_1")
# del model  # delete trained model to demonstrate loading
# #
# # Load the trained agent
# # NOTE: if you have loading issue, you can pass `print_system_info=True`
# # to compare the system on which the model was trained vs the current one
# model = DQN.load("./turnleft/Sumo_pattern1_turn_left_DQN_alpha_8e-3_5M_call_1", env=env)
#
# print(model.policy)


model = DQN.load("TurnLeftBestModel1009_01/best_model", env=env)

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

    # TCPClient.tcp_client()
    # data = {
    #     'TickId': env.episode_step,
    #     'X': env.vehicle.x/SCALE,
    #     'Y': env.vehicle.y/SCALE,
    #     'YawAngle': env.vehicle.yaw_angle
    # }
    # TCPClient.send_data(TCPClient.client_socket, data)
    # TCPClient.receive_data(TCPClient.client_socket)
    while not done:
        # print(env.episode_step)
        action, _states = model.predict(obs, deterministic=True)
        # action = env.action_space.sample()
        # if 60 <= env.episode_step <= 65:
        #     action = 3dwd
        # else:
        #     action = 0
        obs, reward, done, info = env.step(action)
        env.render()
        rewards += reward

        # if reward < -100 or reward > 100:
        #   print(reward)

    # TCPClient.client_socket.close()
    print(rewards)

    # tensorboard --logdir ./logs
