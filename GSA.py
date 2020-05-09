# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Python libraries
import os
import math
import random

# Helper libraries
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import pygame

# Game Library
from game import Game
from search_ai import get_move

# Definition of a Snake Agent
class SnakeAgent:
    def __init__(self, game):
        self.game = game
        self.checkpoint_path = "training/cp-{epoch:04d}.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(self.game.settings.width+2, self.game.settings.height+2, 2)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(4)
        ])

        self.model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        self.probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
    def train(self, gameStates, optimalMoves, epochs=250):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
        self.model.fit(gameStates, optimalMoves, epochs=epochs, use_multiprocessing=True, callbacks=[cp_callback])
    def save(self):
        self.model.save_weights(self.checkpoint_path.format(epoch=0))
    def restore(self):
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        self.model.load_weights(latest)
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
    def generate_data(self, game, game_count=784, move_count=50):
        states = []
        moves = []
        games_played = 0
        moves_played = 0

        while games_played < game_count:
            game.restart_game()
            moves_played = 0
            while not game.game_end() and moves_played < move_count:
                state = game.current_state()
                move = get_move(game)
                reward = game.do_move(move)
                # print(move, reward)

                states.append(state)
                moves.append(move)
                moves_played += 1
            games_played += 1

        statesArr = np.asarray(states)
        movesArr = np.asarray(moves)
        print("Data:", len(states))
        return statesArr, movesArr


if __name__ == "__main__":
    game = Game()
    agent = SnakeAgent(game)
    states, moves = agent.generate_data(game, game_count=5, move_count=100)
    agent.train(states, moves, epochs=10)
    agent.save()

    game.restart_game()
    agent2 = SnakeAgent(game)
    agent.restore()