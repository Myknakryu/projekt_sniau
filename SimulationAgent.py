import gymnasium as gym
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import random
from Actor import Actor
from Critic import Critic
from threading import Thread, Lock
from multiprocessing import Process, Pipe
from ThreadedEnvironment import ThreadedEnvironment
from matplotlib.animation import FuncAnimation
import time
import copy

class SimulationAgent:
    def __init__(self,max_steps, epochs, max_episodes, plotting = True):
        self.env = gym.make('LunarLander-v2')
        self.max_episode_steps = max_steps
        self.env._max_episode_steps = self.max_episode_steps
        self.action_size = self.env.action_space.n
        self.observation_size = self.env.observation_space.shape
        self.epochs = epochs
        self.episode = 0
        self.max_episodes = max_episodes
        self.replay_count = 0
        self.scores = []
        self.mean_vals = []
        self.episodes = []
        if plotting:
            matplotlib.use('Qt5Agg')
            plt.plot()
            plt.ion()
            plt.show()

        self.Actor = Actor(action_space= self.action_size,
                            observation_space_shape = self.observation_size)
        self.Critic = Critic(action_space = self.action_size, 
                             observation_space_shape = self.observation_size)
    def redraw(self):
            plt.cla()
            plt.plot(self.episodes, self.scores, label='Uzyskana nagroda')
            plt.plot(self.episodes, self.mean_vals, label='Średnia z 50 epizodów')
            plt.legend()
            plt.xlabel("Epizod")
            plt.ylabel("Wartość nagrody")
            plt.xlim([0, self.episode])
            plt.grid()
            plt.draw()
            plt.pause(0.01)

    def new_action(self, state):
        reshaped_state = np.reshape(state, [1, self.observation_size[0]])
        prediction = self.Actor.predict(reshaped_state)[0]
        action = np.random.choice(self.action_size, p=prediction)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        return action, action_onehot, prediction

    
    def get_gaes(self, rewards, dones, values, next_values):
        deltas = [r + 0.99 * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * 0.891 * gaes[t + 1]

        target = gaes + values
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones, next_states):
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)

        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
        y_true = np.hstack([advantages, predictions, actions])
        
        a_loss = self.Actor.model.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=False)
        c_loss = self.Critic.model.fit([states, values], target, epochs=self.epochs, verbose=0, shuffle=False)


        self.replay_count += 1

    def run(self):
        state, _ = self.env.reset()
        state = np.reshape(state, [1, self.observation_size[0]])
        done, score = False, 0
        while True:
            states = [] 
            next_states = []
            actions = [] 
            rewards = [] 
            predictions = [] 
            dones = []
            while not done:
                self.env.render()
                action, action_onehot, prediction = self.new_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.observation_size[0]]))
                actions.append(action_onehot)
                rewards.append(reward)
                dones.append(done)
                predictions.append(prediction)
                state = np.reshape(next_state, [1, self.observation_size[0]])
                score += reward
                if done:
                    self.episodes.append(self.episode)
                    self.episode += 1
                    self.replay(states, actions, rewards, predictions, dones, next_states)
                    self.scores.append(score)
                    print("Episode: {} of {}, Current score: {}, Avg. Score: {}".format(
                        self.episode, self.max_episodes, score, np.mean(self.scores)))
                    self.mean_vals(np.mean(self.scores[-50:]))
                    self.redraw()
                    state, _ = self.env.reset()
                    done, score = False, 0
                    state = np.reshape(state, [1, self.observation_size[0]])

            if self.episode >= self.max_episodes:
                break
        self.env.close()

    def run_parallelized(self, threads):
        works, parent_connections, child_connections = [], [], []
        for idx in range(threads):
            parent_connection, child_connection = Pipe()
            work = ThreadedEnvironment(idx, child_connection, self.observation_size, self.action_size, True)
            work.start()
            works.append(work)
            parent_connections.append(parent_connection)
            child_connections.append(child_connection)

        states = [[] for _ in range(threads)]
        next_states = [[] for _ in range(threads)]
        actions = [[] for _ in range(threads)]
        rewards = [[] for _ in range(threads)]
        dones = [[] for _ in range(threads)]
        predictions = [[] for _ in range(threads)]
        score = [0 for _ in range(threads)]

        state = [0 for _ in range(threads)]
        for worker_id, parent_conn in enumerate(parent_connections):
            state[worker_id] = parent_conn.recv()

        while self.episode < self.max_episodes:
            predictions_list = self.Actor.predict(np.reshape(state, [threads, self.observation_size[0]]))
            actions_list = [np.random.choice(self.action_size, p=i) for i in predictions_list]

            for worker_id, parent_conn in enumerate(parent_connections):
                parent_conn.send(actions_list[worker_id])
                action_onehot = np.zeros([self.action_size])
                action_onehot[actions_list[worker_id]] = 1
                actions[worker_id].append(action_onehot)
                predictions[worker_id].append(predictions_list[worker_id])

            for worker_id, parent_conn in enumerate(parent_connections):
                next_state, reward, done, _ = parent_conn.recv()

                states[worker_id].append(state[worker_id])
                next_states[worker_id].append(next_state)
                rewards[worker_id].append(reward)
                dones[worker_id].append(done)
                state[worker_id] = next_state
                score[worker_id] += reward

                if done:
                    self.episodes.append(self.episode)
                    self.scores.append(score[worker_id])
                    mean_val = np.mean(self.scores[-50:])
                    self.mean_vals.append(mean_val)
                    print("Episode: {} of {}, Current score: {}, Avg. Score: {}".format(
                        self.episode, self.max_episodes, score[worker_id], mean_val))
                    score[worker_id] = 0
                    self.redraw()
                    if(self.episode < self.max_episodes):
                        self.episode += 1
                        
            for worker_id in range(threads):
                if len(states[worker_id]) >= self.max_episodes:
                    self.replay(states[worker_id], actions[worker_id], rewards[worker_id], predictions[worker_id], dones[worker_id], next_states[worker_id])
                    states[worker_id] = []
                    next_states[worker_id] = []
                    actions[worker_id] = []
                    rewards[worker_id] = []
                    dones[worker_id] = []
                    predictions[worker_id] = []

        works.append(work)
        for work in works:
            work.terminate()
            work.join()
        
        self.redraw()

    def load(self, actor_file, critic_file ):
        self.Actor.model.load_weights(actor_file)
        self.Critic.model.load_weights(critic_file)

    def save(self, actor_file, critic_file):
        self.Actor.model.save_weights(actor_file)
        self.Critic.model.save_weights(critic_file)


    def play(self):
        self.env = gym.make('LunarLander-v2', render_mode= 'human')
        self.env._max_episode_steps = self.max_episode_steps
        for _ in range(self.max_episodes):
            state, _ = self.env.reset()
            state = np.reshape(state,[1,self.observation_size[0]])
            done = False
            score = 0
            while not done:
                self.env.render()
                action = np.argmax(self.Actor.predict(state)[0])
                state, reward, done, truncated, _ = self.env.step(action)
                state = np.reshape(state, [1, self.observation_size[0]])
                score += reward
                if done or truncated:
                    break
        self.env.close()