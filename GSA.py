# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from game import Game

game = Game()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30, 30, 2)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(4)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

game.restart_game()
while not game.game_end():
    state = np.zeros((1, 30, 30, 2))
    state[0] = game.current_state()
    move_prediction = probability_model.predict(state)
    move = np.argmax(move_prediction)
    print(move, game.do_move(move))