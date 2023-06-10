import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K

class Critic:
    def __init__(self, action_space, observation_space_shape) -> None:
        X_input = Input(shape=observation_space_shape)
        old_values = Input(shape=(1,))

        hidden = Dense(512, activation='relu', kernel_initializer='he_uniform')(X_input)
        hidden = Dense(256, activation='relu', kernel_initializer='he_uniform')(hidden)
        hidden = Dense(64, activation='relu', kernel_initializer='he_uniform')(hidden)
        output = Dense(1, activation=None)(hidden)
        
        self.model = keras.Model(inputs=[X_input, old_values], outputs=output)
        self.model.compile(loss=self.critic_PPO2_loss(old_values), optimizer=Adam(learning_rate=0.00025))

    def critic_PPO2_loss(self, values):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = 0.2
            clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2
            
            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            return value_loss
        return loss

    def predict(self, state):
        return self.model.predict([state, np.zeros((state.shape[0], 1))], verbose=None)