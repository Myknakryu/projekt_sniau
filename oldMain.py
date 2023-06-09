import gymnasium as gym
import random
import numpy as np
import tflearn
from gymnasium.utils import play
from alive_progress import alive_bar
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter

LR = 1e-3
env = gym.make("CartPole-v1", render_mode = None)

env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    with alive_bar(initial_games) as bar:
        for _ in range (initial_games):
            score = 0
            game_memory = []
            prev_observation = []
            for _ in range(goal_steps):
                action = random.randrange(0,2)
                observation, reward, done, truncated, info = env.step(action)

                if len(prev_observation) > 0:
                    game_memory.append([prev_observation, action])

                prev_observation = observation
                score += reward
                if done:
                    break
            
            if score >= score_requirement:
                accepted_scores.append(score)
                for data in game_memory:
                    if data[1] == 1:
                        output = [0,1]

                    elif data[1] == 0:
                        output = [1,0]

                    training_data.append([data[0], output])

            env.reset()
            scores.append(score)
            bar()
    print("avg score accepted: ", mean(accepted_scores))
    print('median: ', median(accepted_scores))
    print('count: ', Counter(accepted_scores))
    return training_data

initial_population()

def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
    
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate = LR, loss = 'categorical_crossentropy', name = 'targets')
    
    model = tflearn.DNN(network, tensorboard_dir = 'log')

    return model

def train_model(training_data, model=False):
    x = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size=len(x[0]))

    model.fit({'input':x}, {'targets':y}, n_epoch=10, snapshot_step=500, show_metric=True, run_id='tak')
    
    return model

training_data = initial_population()
model = train_model(training_data)


scores = []
choices = []
env = gym.make("CartPole-v1", render_mode = 'human')
for each_game in range(10):
    score = 0;
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])

        choices.append(action)
        new_observation, reward, done, truncated, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward;
        if done:
            break
    scores.append(score)

print('Average: ', mean(scores))
