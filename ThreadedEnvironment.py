from threading import Thread, Lock
from multiprocessing import Process, Pipe

import gymnasium as gym
import numpy as np
class ThreadedEnvironment(Process):
    def __init__(self, idx, child_connection, observation_size, action_size, renderer):
        super(ThreadedEnvironment, self).__init__()
        self.env = gym.make('LunarLander-v2', render_mode = None)
        self.env._max_episode_steps = 1000
        self.idx =idx
        self.child_connection = child_connection
        self.observation_size = observation_size
        self.action_size = action_size
        self.render = renderer

    def run(self):
        super(ThreadedEnvironment, self).run()
        state, _ = self.env.reset()
        state = np.reshape(state, [1, self.observation_size[0]])
        self.child_connection.send(state)
        while True:
            action = self.child_connection.recv()
            if self.render and self.idx == 0:
                self.env.render()

            state, reward, done, trunc, info = self.env.step(action)
            state = np.reshape(state, [1, self.observation_size[0]])
            if done or trunc:
                state, _ = self.env.reset()
                state = np.reshape(state, [1, self.observation_size[0]])

            self.child_connection.send([state, reward, done, info])

