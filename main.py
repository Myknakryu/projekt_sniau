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

from threading import Thread, Lock
from multiprocessing import Process, Pipe
import time
import copy

env = gym.make("LunarLander-v2", render_mode = None)


# PPO Actor

class Actor:
    def __init__(self, action_space, observation_space_shape) -> None:
        self.action_space = action_space
        self.model = keras.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=observation_space_shape, 
                            kernel_initializer=tf.random_normal_initializer(stddev=0.01)),
            keras.layers.Dense(256, activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01)),
            keras.layers.Dense(64, activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01)),
            keras.layers.Dense(self.action_space, activation='softmax')
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.00025), loss=self.ppo_loss)


    def ppo_loss(self, y_true, y_pred):
        # https://arxiv.org/pdf/1707.06347.pdf
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.01
        
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))
        
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)
        
        total_loss = actor_loss - entropy

        return total_loss

    def predict(self, state):
        return self.model.predict(state, verbose=None)
    
# PPO critic
class Critic:
    def __init__(self, action_space, observation_space_shape) -> None:
        X_input = Input(shape=observation_space_shape)
        old_values = Input(shape=(1,))
        hidden = Dense(512, activation='relu', kernel_initializer='he_uniform')(X_input)
        hidden = Dense(256, activation='relu', kernel_initializer='he_uniform')(hidden)
        hidden = Dense(64, activation='relu', kernel_initializer='he_uniform')(hidden)
        output = Dense(1, activation=None)(hidden)
        self.model = keras.Model(inputs=[X_input, old_values], outputs=output)
        self.model.compile(loss=[self.critic_PPO2_loss(old_values)], optimizer=Adam(learning_rate=0.00025))

    def critic_PPO2_loss(self, values):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = 0.2
            values = y_true[:, :-1]  # old values are now part of y_true
            true_value = y_true[:, -1:]  # adjust as necessary to match your data shape
            clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (true_value - clipped_value_loss) ** 2
            v_loss2 = (true_value - y_pred) ** 2
            
            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            return value_loss
        return loss

    def predict(self, state):
        return self.model.predict([state, np.zeros((state.shape[0], 1))], verbose=None)

def get_gaes(rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

actor_instance = Actor(env.action_space.n, env.observation_space.shape)
critic_instance = Critic(env.action_space.n, env.observation_space.shape)

state, _ = env.reset()

state_size = env.observation_space.shape
action_size = env.action_space.n

TOTAL_EPISODES = 1000
episode = 0
epochs = 10
shuffle = False
replay_count = 0

print(state)
print(state.shape)
state = np.reshape(state, [1, state_size[0]])

done = False
score = 0
MAX_DURATION = 1000
while True:
    states = []
    next_states = []
    actions = []
    rewards = []
    predictions = []
    dones = []
    counter = 0
    while not done:
        env.render()
        prediction = actor_instance.predict(state)[0]
        action = np.random.choice(action_size)
        action_oneshot = np.zeros([action_size])
        action_oneshot[action] = 1
        next_state, reward, done, _, _ = env.step(action)
        states.append(state)
        next_states.append(np.reshape(next_state, [1, state_size[0]]))
        actions.append(action_oneshot)
        dones.append(done)
        rewards.append(reward)
        predictions.append(prediction)
        state = np.reshape(next_state, [1, state_size[0]])
        score += reward
        counter += 1
        if counter > MAX_DURATION:
            done = True
        if done:
            episode += 1
            print("score {}".format(score))

            states = np.vstack(states)
            next_states = np.vstack(next_states)
            actions = np.vstack(actions)
            predictions = np.vstack(predictions)
            values = critic_instance.predict(states)
            next_values = critic_instance.predict(next_states)
            advantages, target = get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
            old_values = values  # compute old values as necessary
            target = np.hstack([old_values, target])  # include old values in your targets
            y_true = np.hstack([advantages, predictions, actions])
            a_loss = actor_instance.model.fit(states, y_true, epochs=epochs, verbose=0, shuffle=shuffle)
            c_loss = critic_instance.model.fit([states, values], target, epochs=epochs, verbose=0, shuffle=shuffle)

            # print("actor_loss: {}".format(a_loss.history['loss']))
            # print("critic_loss: {}".format(c_loss.history['loss']))

            state, _ = env.reset()
            done, score = False, 0
            state = np.reshape(state, [1, state_size[0]])
            break
    if episode >= TOTAL_EPISODES:
        break
env.close()