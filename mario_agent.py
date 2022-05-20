from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import clone_model
import random

import numpy as np

INPUT = 84
ACTION_SPACE = 7

class mario_agent:
    def __init__(self):
        self.nn = self.create_nn()
        self.target_nn = clone_model(self.nn)

    def create_nn(self):
        nn = Sequential()
        nn.add(Flatten())
        nn.add(Dense(units=256, activation='relu'))
        nn.add(Dense(units=256, activation='relu'))
        nn.add(Dense(units=ACTION_SPACE, activation='softmax'))
        nn.compile(loss='mse')
        return nn

    def get_action(self, state):

        epsilon = 1.0 #exploration variable
        r = random.random()
        if r < epsilon:
            action = random.randrange(0, ACTION_SPACE)
        else:
            q_vals = self.nn(np.array(state).reshape(-1, *state.shape))[0]
            action = np.argmax(q_vals)
        
        if epsilon < 0.01:
            epsilon *= 0.995
        return action

    def update_target_nn(self):
        self.target_nn = clone_model(self.nn)

    def get_target_max_q_val(self, state):
        return np.amax(self.target_nn(np.array(state).reshape(-1, *state.shape))[0])

    def compute_loss(self, state, target):
        self.nn.fit(x=state, y=target)

    def get_q_vals(self, state):
        return self.nn(state)

