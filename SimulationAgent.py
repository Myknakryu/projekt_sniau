import gymnasium as gym
import numpy as np
import random
from Actor import Actor
from Critic import Critic
import copy

class SimulationAgent:
    def __init__(self):
        self.env = gym.make('LunarLander-v2')
        self.env._max_episode_steps = 1000
        self.action_size = self.env.action_space.n
        self.observation_size = self.env.observation_space.shape
        self.max_avg = 50
        self.epochs = 10
        self.episode = 0
        self.max_episodes = 1000
        self.replay_count = 0
        self.scores = []
        self.Actor = Actor(action_space= self.action_size,
                            observation_space_shape = self.observation_size)
        self.Critic = Critic(action_space = self.action_size, 
                             observation_space_shape = self.observation_size)

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
        c_loss = self.Critic.model.fit(states, y=values, epochs=self.epochs, verbose=0, shuffle=False)


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
                    self.episode += 1
                    self.replay(states, actions, rewards, predictions, dones, next_states)
                    self.scores.append(score)
                    print("Episode: {} of {}, Current score: {}, Avg. Score: {}".format(
                        self.episode, self.max_episodes, score, np.mean(self.scores)))
                    state, _ = self.env.reset()
                    done, score = False, 0
                    state = np.reshape(state, [1, self.observation_size[0]])

            if self.episode >= self.max_episodes:
                break
        self.env.close()