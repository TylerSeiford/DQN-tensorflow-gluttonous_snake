import tensorflow as tf
import random
import numpy as np
import os
from collections import deque

from game import Game
from GSA_config import Config

class GravitationalSnakeAgent:
    def __init__(self, sess, game):
        self.sess = sess
        config = Config()
        self.game = game
        self.settings = self.game.settings
        self.explore = config.explore
        self.memory_size = config.memory_size
        self.batch_size = config.batch_size
        self.n_actions = config.n_actions
        self.g = config.g
        self.rand = config.rand
        self.model_file = config.model_file

        # total learning steps
        self.learn_step = 0
        
        self.memory = deque(maxlen=self.memory_size)
        
        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.loss_list = []
        self.restore_model()

    def conv_network(self, scope_name, state):
        settings = self.settings

        with tf.variable_scope(scope_name):
            conv1 = tf.layers.conv2d(state, filters=32, kernel_size=3,
                             strides=1, padding="SAME",
                             activation=tf.nn.relu, name="conv1")
            conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=3,
                             strides=1, padding="SAME",
                             activation=tf.nn.relu, name="conv2")
            conv3 = tf.layers.conv2d(conv2, filters=128, kernel_size=3,
                             strides=1, padding="SAME",
                             activation=tf.nn.relu, name="conv3")
            conv4 = tf.layers.conv2d(conv3, filters=4, kernel_size=1,
                             strides=1, padding="SAME",
                             activation=tf.nn.relu, name="conv4")

            conv4_flat = tf.reshape(conv4, shape=[-1, 4 * (settings.width+2) * (settings.height+2)])

            h_fc1 = tf.layers.dense(conv4_flat, 128, activation=tf.nn.relu)
            q_value = tf.layers.dense(h_fc1, self.n_actions)

        return q_value

    def _build_net(self):
        
        settings = self.settings
        with tf.name_scope("inputs"):
            self.state = tf.placeholder(tf.float32, shape=[None, settings.width+2, settings.height+2, 8], name="s")
            self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
            self.a = tf.placeholder(tf.float32, [None, self.n_actions], name='a')  # input Action

        # evaluate_net
        self.q_eval = self.conv_network('eval_net', self.state)
        
        # target_net
        self.q_next = self.conv_network('target_net', self.state)

        action_value = tf.reduce_sum(tf.multiply(self.q_eval, self.a), reduction_indices=1)        
        self.loss = tf.reduce_mean(tf.square(self.r - action_value))
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.loss)

    def restore_model(self):
        sess = self.sess
        model_file = self.model_file
        self.saver = tf.train.Saver()
        
        if os.path.exists(model_file + '.meta'):
            self.saver.restore(sess, model_file)
        else:
            sess.run(tf.global_variables_initializer())

    def get_model_params(self):
        gvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self.sess.run(gvars))}

    def choose_action(self, s_t):
        a_t = np.zeros([self.n_actions])
        action_index = 0
 
        q_eval = self.sess.run(self.q_eval, feed_dict={self.state : [s_t]})[0]
        action_index = np.argmax(q_eval)
            
        a_t[action_index] = 1
            
        return a_t, action_index

    def play_a_game(self):
        game = self.game   
        game.restart_game()
        score = 0
        step = 0
        game_state = game.current_state()
        s_t = np.concatenate((game_state, game_state, game_state, game_state), axis=2)        
        
        while not game.game_end():
            a_t, action_index = self.choose_action(s_t)                 
            # run the selected action and observe next state and reward
            move = action_index
            r_t = game.do_move(move)
            
            if r_t == 1:
                score += 1

            game_state = game.current_state()
            end = game.game_end()
            s_t1 = np.append(game_state, s_t[:, :, :-2], axis=2)
            
            self.memory.append((s_t, a_t, r_t, s_t1, end))
            self.learn_step += 1

            s_t = s_t1
            step += 1
        return step, score

    def gsa_evolve(self, agents):
        for base_agent in agents:
            for f in base_agent.features:
                for other_agent in agents:
                    base_agent.f.acc += ((self.g * base_agent.score * other_agent.score) / ((other_agent.f.value - base_agent.f.value) ^ 2)) / base_agent.score
                base_agent.f.value += 2 * random.random() * base_agent.f.acc

    def train(self):
        try:
            game_num = 0
            scores = []
            score_means = []
            while self.learn_step < self.explore:
                step, score = self.play_a_game()                      
                game_num += 1
                scores.append(score)

                if game_num % 10 == 0:
                    score_mean = np.mean(scores)
                    score_means.append(score_mean)
                    print("game: {} step length: {} score: {:.2f}".format(game_num, step, score_mean))
                    scores = []
               
            self.plot_loss(score_means)
            
        except KeyboardInterrupt:
            print('[INFO] Interrupt manually, try saving checkpoint for now...')
            self.saver.save(self.sess, './model/snake')
            self.plot_loss(score_means)

    def plot_loss(self, score_means):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(score_means)), score_means)
        plt.ylabel('Score')
        plt.xlabel('training steps')
        plt.show()

if __name__ == "__main__":
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    game = Game()
    gsa = GravitationalSnakeAgent(sess, game)
    gsa.train()