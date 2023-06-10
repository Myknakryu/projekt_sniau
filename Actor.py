import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K

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
   
