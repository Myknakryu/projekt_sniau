import gymnasium as gym
import numpy as np

# Tensorflow annoyance with AMDGPU since ROCm support is unstable as HELL
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
from SimulationAgent import SimulationAgent
from threading import Thread, Lock
from multiprocessing import Process, Pipe
import time
import copy

if __name__ == "__main__":
    simulation = SimulationAgent()
    simulation.run()


# env = gym.make("LunarLander-v2", render_mode = None)

# def get_gaes(rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
#         deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
#         deltas = np.stack(deltas)
#         gaes = copy.deepcopy(deltas)
#         for t in reversed(range(len(deltas) - 1)):
#             gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

#         target = gaes + values
#         if normalize:
#             gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
#         return np.vstack(gaes), np.vstack(target)

# actor_instance = Actor(env.action_space.n, env.observation_space.shape)
# critic_instance = Critic(env.action_space.n, env.observation_space.shape)

# state, _ = env.reset()

# state_size = env.observation_space.shape
# action_size = env.action_space.n

# TOTAL_EPISODES = 1000
# episode = 0
# epochs = 10
# shuffle = False
# replay_count = 0

# print(state)
# print(state.shape)
# state = np.reshape(state, [1, state_size[0]])

# done = False
# score = 0
# while True:
#     states = []
#     next_states = []
#     actions = []
#     rewards = []
#     predictions = []
#     dones = []
#     while not done:
#         env.render()
#         prediction = actor_instance.predict(state)[0]
#         action = np.random.choice(action_size)
#         action_oneshot = np.zeros([action_size])
#         action_oneshot[action] = 1
#         next_state, reward, done, _, _ = env.step(action)
#         states.append(state)
#         next_states.append(np.reshape(next_state, [1, state_size[0]]))
#         actions.append(action_oneshot)
#         dones.append(done)
#         rewards.append(reward)
#         predictions.append(prediction)
#         state = np.reshape(next_state, [1, state_size[0]])
#         score += reward
#         if done:
#             episode += 1
#             print("score {}".format(score))

#             states = np.vstack(states)
#             next_states = np.vstack(next_states)
#             actions = np.vstack(actions)
#             predictions = np.vstack(predictions)
#             values = critic_instance.predict(states)
#             next_values = critic_instance.predict(next_states)
#             advantages, target = get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
#             old_values = values  # compute old values as necessary
#             target = np.hstack([old_values, target])  # include old values in your targets
#             y_true = np.hstack([advantages, predictions, actions])
#             a_loss = actor_instance.model.fit(states, y_true, epochs=epochs, verbose=0, shuffle=shuffle)
#             c_loss = critic_instance.model.fit([states, values], target, epochs=epochs, verbose=0, shuffle=shuffle)

#             # print("actor_loss: {}".format(a_loss.history['loss']))
#             # print("critic_loss: {}".format(c_loss.history['loss']))

#             state, _ = env.reset()
#             done, score = False, 0
#             state = np.reshape(state, [1, state_size[0]])

#     if episode >= TOTAL_EPISODES:
#         break
# env.close()