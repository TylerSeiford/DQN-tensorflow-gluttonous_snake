# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import math
import random

# Game Library
from game import Game

# Definition of a Snake Agent
class GravitationalSnakeAgent:
    def __init__(self, game):
        self.game = game

        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(self.game.settings.width+2, self.game.settings.height+2, 2)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(4)
        ])

        self.model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        self.probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])

    def play_a_game(self):
        game.restart_game()
        while not game.game_end():
            state = np.zeros((1, self.game.settings.width+2, self.game.settings.width+2, 2))
            state[0] = game.current_state()
            move_prediction = self.probability_model.predict(state)
            move = np.argmax(move_prediction)
            reward = game.do_move(move)
            # print(state[0], move_prediction, move, reward)
        return game.snake.score
    def choose_a_move(self, state):
            _state = np.zeros((1, self.game.settings.width+2, self.game.settings.width+2, 2))
            _state[0] = state
            move = np.argmax(self.probability_model.predict(_state))
            return move



game = Game()
agents = []
scores = []

# create agents and scores
for i in range(10):
    agents.append(GravitationalSnakeAgent(game))
    scores.append(0)

print(len(agents), len(scores))

# run games
for i in range(10):
    scores[i] = agents[i].play_a_game()
    scores[i] += 1
    print(i, ": ", scores[i])


# gsa train
g = 6.674e-11
r = 2
# Loop over every agent
for i in range(10):
    # Loop over every layer but the first
    for k in range(1, 3):
        # Loop over every trainable weight in the layer
        for l in range(len(agents[i].model.layers[k].weights)):
            if(l == 0):
                for m in range(agents[i].model.layers[k].weights[l].shape[0]):
                    for n in range(agents[i].model.layers[k].weights[l][m].shape[0]):

                        valueI = agents[i].model.layers[k].weights[l][m][n].numpy()
                        accel = 0

                        # Loop over every other agent
                        for j in range(10):
                            if i != j:
                                valueJ = agents[j].model.layers[k].weights[l][m][n].numpy()
                                accel += ((g * scores[i] * scores[j]) / math.pow(valueJ - valueI, 2)) / scores[i]
                        valueI += r * random.random() * accel
                        print("Old:", agents[i].model.layers[k].weights[l][m][n])
                        print("New:", tf.convert_to_tensor(valueI, dtype=float32))
                        agents[j].model.layers[k] = keras.layers.Dense(31, activation='relu')

                        # print("(", i, j, k, l, m, n, ")", value)
            else:
                continue

#            base_agent.f.acc += ((self.config.g * base_agent.score * other_agent.score) / ((other_agent.f.value - base_agent.f.value) ^ 2)) / base_agent.score
#        base_agent.f.value += 2 * random.random() * base_agent.f.acc