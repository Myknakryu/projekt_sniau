import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K

class Critic:
    def __init__(self, action_space, observation_space_shape) -> None:
        X_input = Input(shape=observation_space_shape)
        hidden = Dense(512, activation='relu', kernel_initializer='he_uniform')(X_input)
        hidden = Dense(256, activation='relu', kernel_initializer='he_uniform')(hidden)
        hidden = Dense(64, activation='relu', kernel_initializer='he_uniform')(hidden)
        output = Dense(1, activation=None)(hidden)
        self.model = keras.Model(inputs=X_input, outputs=output)
        self.model.compile(loss=self.critic_PPO2_loss(), optimizer=Adam(learning_rate=0.00025))

    def critic_PPO2_loss(self):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = 0.2
            old_values = y_true # we take the old values as the true values
            clipped_value_loss = old_values + K.clip(y_pred - old_values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2
            
            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            return value_loss
        return loss

    def predict(self, state):
        return self.model.predict(state, verbose=None)