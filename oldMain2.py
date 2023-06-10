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
env = gym.make("LunarLander-v2", render_mode = None)

env.reset()
goal_steps = 500
score_requirement = 100
initial_games = 10000

print("-- Observations",env.observation_space)
print("-- actionspace",env.action_space)

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
                action = env.action_space.sample()
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
                    match data[1]:
                        case 0:
                            output = [1,0,0,0]
                        case 1:
                            output = [0,1,0,0]
                        case 2:
                            output = [0,0,1,0]
                        case 3:
                            output = [0,0,0,1]

                    training_data.append([data[0], output])

            env.reset()
            scores.append(score)
            bar()
    print("avg score accepted: ", mean(accepted_scores))
    print('median: ', median(accepted_scores))
    print('count: ', Counter(accepted_scores))
    return training_data


try:
    training_data = np.load('initial.npy', allow_pickle=True)
except IOError:
    training_data = initial_population()
    save_data = np.array(training_data)
    np.save('initial.npy', save_data)

def neural_network_model():
    network = input_data(shape=[None, env.observation_space.shape[0]], name='input')

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

    network = fully_connected(network, env.action_space.n, activation='softmax')
    network = regression(network, optimizer='adam', loss = 'categorical_crossentropy', name = 'targets')
    
    model = tflearn.DNN(network)

    return model

def train_model(training_data, model=False):
    x = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model()

    model.fit({'input':x}, {'targets':y}, n_epoch=10, snapshot_step=5000, show_metric=True, run_id='tak')
    
    return model

model = train_model(training_data)


scores = []
choices = []
env = gym.make("LunarLander-v2", render_mode = 'human')
for each_game in range(10):
    score = 0;
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0,4)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])

        choices.append(action)
        new_observation, reward, done, truncated, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward;
        if done:
            break
    scores.append(score)

print('Average: ', mean(scores))
