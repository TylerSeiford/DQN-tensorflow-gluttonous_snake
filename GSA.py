# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Python libraries
import os
import math
import random
import statistics
import logging

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
    # Create an AI using a game
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
    # Train the AI using some perfect game states and moves.  Returns the training accuracy.
    def train(self, gameStates, optimalMoves, epochs=250):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
        accuracy = self.model.fit(gameStates, optimalMoves, epochs=epochs, use_multiprocessing=True, callbacks=[cp_callback]).history['accuracy']
        return accuracy[len(accuracy) - 1]
    # Test the AI using some perfect game states and moves.  Returns the validation accuracy.
    def test(self, gameStates, optimalMoves):
        return self.model.evaluate(gameStates, optimalMoves)
    # Save the AI
    def save(self):
        self.model.save_weights(self.checkpoint_path.format(epoch=0))
    # Restore the AI
    def restore(self):
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        self.model.load_weights(latest)
    # Play a game and return the score
    def play_a_game(self):
        self.game.restart_game()
        while not self.game.game_end():
            state = np.zeros((1, self.game.settings.width+2, self.game.settings.width+2, 2))
            state[0] = self.game.current_state()
            move_prediction = self.probability_model.predict(state)
            move = np.argmax(move_prediction)
            reward = self.game.do_move(move)
            # print(state[0], move_prediction, move, reward)
        return self.game.snake.score
    # Choose a move depending on a game state
    def choose_a_move(self, state):
            _state = np.zeros((1, self.game.settings.width+2, self.game.settings.width+2, 2))
            _state[0] = state
            move = np.argmax(self.probability_model.predict(_state))
            return move
    # Generate some perfect game data using BFS/DFS
    def generate_data(self, game_count=784, move_count=50):
        states = []
        moves = []
        games_played = 0
        moves_played = 0

        while games_played < game_count:
            self.game.restart_game()
            moves_played = 0
            while not self.game.game_end() and moves_played < move_count:
                state = self.game.current_state()
                move = get_move(self.game)
                reward = self.game.do_move(move)
                # print(move, reward)

                states.append(state)
                moves.append(move)
                moves_played += 1
            games_played += 1

        statesArr = np.asarray(states)
        movesArr = np.asarray(moves)
        print("Data:", len(states))
        return statesArr, movesArr


def genData(validation_states, validation_moves, game_count=10, move_count=50, epochs=5):
    game = Game()
    agent = SnakeAgent(game)

    states, moves = agent.generate_data(game_count=game_count, move_count=move_count)
    trainingAccuracy = agent.train(states, moves, epochs=epochs)
    print("Trained TensorFlow! ", trainingAccuracy)

    testingAccuracy = agent.test(validation_states, validation_moves)[1]
    print("Tested TensorFlow! ", testingAccuracy)

    scores = []
    for i in range(10):
        scores.append(agent.play_a_game())
    meanScore = statistics.mean(scores)
    print("Tested Games! ", meanScore)
    
    agent.save()

    print("")
    print("Games, Moves, Epochs, Training, Validation, Playing")
    print(game_count, move_count, epochs, trainingAccuracy, testingAccuracy, meanScore)
    del agent
    del game
    return trainingAccuracy, testingAccuracy, meanScore

if __name__ == "__main__":
    print("Generating validation data")
    game = Game()
    agent = SnakeAgent(game)
    states, moves = agent.generate_data(game_count=10, move_count=500)
    del agent
    del game
    print("Generated validation data")

    logging.basicConfig(filename='log.txt',
                            filemode='a',
                            level=logging.DEBUG,
                            format='%(message)s')
    logging.info("Games, Moves, Epochs, Training, Validation, Playing")

    for game_count in (1, 10, 50, 100): # (1, 10, 50, 100, 250, 500, 1000)
        for move_count in (1, 10, 50, 100): # (1, 10, 50, 100, 250, 500, 1000)
            for epoch in (1, 5, 10, 50, 100): # (1, 5, 10, 50, 100, 250, 500)
                train, test, mean = genData(states, moves, game_count, move_count, epoch)
                logging.info('%s %s %s %s %s %s', game_count, move_count, epoch, train, test, mean)