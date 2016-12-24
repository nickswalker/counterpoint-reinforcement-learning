import random
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from rl.action import Action
from rl.agent.agent import Agent
from rl.domain import Domain
from rl.task import Task
from rl.valuefunction import FeatureExtractor


class DuelingQNetworkAgent(Agent):
    def __init__(self, domain: Domain, task: Task, feature_extractor: FeatureExtractor, epsilon=0.1, alpha=0.5,
                 gamma=1.0, value_function=None):
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.alpha = alpha / domain.history_length
        self.gamma = gamma
        self.feature_extractor = feature_extractor

        num_actions = len(domain.get_actions())
        num_state_features = feature_extractor.num_features()

        if value_function is None:
            self.value_function = DoubleQNetworkValueFunction(num_actions, num_state_features, self.alpha)
        else:
            self.value_function = value_function

        self.world = domain
        self.task = task

        self.current_cumulative_reward = 0.0
        self.previousaction = None
        self.previousstate = None
        self.name = "Double Dueling DQN"

        self.experience_buffer = ExperienceBuffer()
        self.total_steps = 0
        self.update_frequency = 4
        self.batch_size = 32
        self.pretraining_steps = 100

    def choose_action(self, state_features) -> Action:

        max_action, qvalues = self.value_function.getqvalues(state_features)
        index = max_action

        if np.random.rand(1) < self.epsilon:
            index = np.random.randint(0, self.value_function.num_actions)
        return self.world.index_to_action[index]

    def act(self):
        state = self.world.get_current_state()

        phi = self.feature_extractor.extract(state)
        action = self.choose_action(phi)
        action_index = self.world.action_to_index[action]

        self.world.apply_action(action)
        state_prime = self.world.get_current_state()
        phi_prime = self.feature_extractor.extract(state_prime)

        terminal = self.task.stateisfinal(state_prime)

        r = self.task.reward(state, action, state_prime)
        self.total_steps += 1
        if self.total_steps > self.pretraining_steps and self.total_steps % self.update_frequency == 0:
            trainBatch = self.experience_buffer.sample(self.batch_size)  # Get a random batch of experiences.
            loss = self.value_function.update(trainBatch)
            # print(loss)


        self.current_cumulative_reward += r

        experience = (phi, action_index, r, phi_prime, terminal)
        self.experience_buffer.add(experience)

        if terminal:
            step_summary = "r %d" % self.current_cumulative_reward
            #print(step_summary)

    def get_cumulative_reward(self) -> float:
        return self.current_cumulative_reward

    def episode_ended(self):
        # Observe final transition if needed
        self.current_cumulative_reward = 0.0
        self.previousaction = None
        self.previousstate = None
        self.epsilon *= 0.99


class DoubleQNetworkValueFunction:
    def __init__(self, num_actions: int, num_state_features: int, learning_rate: float, n_hidden: int = 512,
                 gamma=0.99):
        self.num_actions = num_actions
        self.main_network = DuelingQNetworkValueFunction(num_actions, num_state_features, learning_rate,
                                                         n_hidden=n_hidden)
        self.target_network = DuelingQNetworkValueFunction(num_actions, num_state_features, learning_rate,
                                                           n_hidden=n_hidden)
        self.gamma = gamma
        self.tau = 0.1
        self.session = tf.Session()
        init = tf.initialize_all_variables()
        self.session.run(init)

        trainables = tf.trainable_variables()

        self.target_operations = self.create_target_move_ops(trainables, self.tau)

    def update(self, experience):
        states = np.vstack(experience[:, 0])
        actions = experience[:, 1]
        rewards = experience[:, 2]
        states_prime = np.vstack(experience[:, 3])
        terminal = experience[:, 4]

        batch_size = len(experience)
        # Below we perform the Double-DQN update to the target Q-values
        Q1 = self.session.run(self.main_network.argmax, feed_dict={self.main_network.inputs: states_prime})
        Q2 = self.session.run(self.target_network.q_out, feed_dict={self.target_network.inputs: states_prime})
        end_multiplier = -(terminal - 1)
        doubleQ = Q2[range(batch_size), Q1]
        targetQ = rewards + (self.gamma * doubleQ * end_multiplier)
        # Update the network with our target values.
        _, loss = self.session.run([self.main_network.update_network, self.main_network.loss],
                                   feed_dict={self.main_network.inputs: states,
                                              self.main_network.target_qvalues: targetQ,
                                              self.main_network.action_indices: actions})

        # Move the target network part of the way towards the main network
        self.update_target(self.target_operations)
        return loss

    def getqvalues(self, state_features: List[float]):
        # Predictions come from the main network
        return self.main_network.getqvalues(state_features, self.session)

    def create_target_move_ops(self, all_tf_vars, tau):
        # This relies on us only having set up two identical q networks, main first then target
        total_vars = len(all_tf_vars)
        half_vars = int(total_vars / 2)
        op_holder = []
        for idx, var in enumerate(all_tf_vars[0:half_vars]):
            op_holder.append(all_tf_vars[idx + half_vars].assign(
                (var.value() * tau) + ((1 - tau) * all_tf_vars[idx + half_vars].value())))
        return op_holder

    def update_target(self, op_holder):
        for op in op_holder:
            self.session.run(op)

    def closesession(self):
        self.session.close()


class DuelingQNetworkValueFunction:
    def __init__(self, num_actions: int, num_state_features: int, learning_rate: float, n_hidden: int = 512):
        self.n_s_feat = num_state_features
        self.n_actions = num_actions
        half_hidden = int(n_hidden / 2)
        with tf.name_scope("qnetwork"):
            # State features go in...
            self.inputs = tf.placeholder(shape=(None, self.n_s_feat), dtype=tf.float32)
            # and through a broad fully-connected layer.
            self.front = tf.contrib.layers.fully_connected(self.inputs, n_hidden)
            # Then we break the fully-connected output into two halves.
            self.streamAC, self.streamVC = tf.split(1, 2, self.front)
            # Advantage maps from the net to actions
            self.advantage_w = tf.Variable(tf.random_normal([half_hidden, num_actions]))
            # Value maps from the net to a scalar value
            self.value_w = tf.Variable(tf.random_normal([half_hidden, 1]))
            self.advantage_out = tf.matmul(self.streamAC, self.advantage_w)
            self.value_out = tf.matmul(self.streamVC, self.value_w)

            # Then combine them together to get our final Q-values. Value is fixed for all entries. Advantage is the size
            # of q_out, but we'll actually just take the deviation from the mean for each entry (which is a smaller value).
            self.q_out = self.value_out + tf.sub(self.advantage_out,
                                                 tf.reduce_mean(self.advantage_out, reduction_indices=1,
                                                                keep_dims=True))
            # An argmax (the entry with the highest value) yields the best action
            self.argmax = tf.argmax(self.q_out, 1)

        with tf.name_scope("loss"):
            # Below we obtain the loss by taking the sum of squares difference between the
            # target and prediction Q values.
            self.target_qvalues = tf.placeholder(shape=[None], dtype=tf.float32)
            self.action_indices = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.action_indices, num_actions, dtype=tf.float32)

            # Get the q values for each action in the buffer (masking out non selecting q values)
            action_values = tf.reduce_sum(tf.mul(self.q_out, self.actions_onehot), reduction_indices=1)

            self.td_error = tf.square(self.target_qvalues - action_values)
            self.loss = tf.reduce_mean(self.td_error)
            trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.update_network = trainer.minimize(self.loss)

    def getqvalues(self, state_features: List[float], session):
        prepped = np.array(state_features).reshape([1, self.n_s_feat])
        feed = {self.inputs: prepped}
        max_action, q_values = session.run([self.argmax, self.q_out], feed_dict=feed)
        return max_action[0], q_values[0]

    def update(self, state_features: List[float], target: List[float], session):
        prepped_state = np.array(state_features).reshape([1, self.n_s_feat])
        prepped_target = np.array(target).reshape([1, self.n_actions])
        feed = {self.inputs: prepped_state, self.target_qvalues: prepped_target}
        _, loss = session.run([self.update_network, self.loss], feed_dict=feed)
        return loss


class ExperienceBuffer:
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience: Tuple):
        experience = np.reshape(np.array(experience), [1, 5])
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])
