from typing import List

import numpy as np
import tensorflow as tf

from rl.action import Action
from rl.agent.agent import Agent
from rl.domain import Domain
from rl.task import Task
from rl.valuefunction import FeatureExtractor


class QNetworkAgent(Agent):
    def __init__(self, domain: Domain, task: Task, feature_extractor: FeatureExtractor, epsilon=0.1, alpha=0.6,
                 gamma=0.95):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.feature_extractor = feature_extractor

        num_actions = len(domain.get_actions())
        num_state_features = feature_extractor.num_features()

        self.value_function = QNetwork(num_actions, num_state_features)

        self.world = domain
        self.task = task

        self.current_cumulative_reward = 0.0
        self.previousaction = None
        self.previousstate = None
        self.name = "Q-network"

    def choose_action(self, state_features) -> Action:

        max_action, qvalues = self.value_function.getqvalues(state_features)
        index = max_action

        if np.random.rand(1) < self.epsilon:
            index = np.random.randint(0, self.value_function.n_actions)
        return self.world.index_to_action[index]

    def act(self):
        state = self.world.get_current_state()

        phi = self.feature_extractor.extract(state)
        action = self.choose_action(phi)
        action_index = self.world.action_to_index[action]

        self.previousstate = state
        self.previousaction = action

        self.world.apply_action(action)
        state_prime = self.world.get_current_state()
        phi_prime = self.feature_extractor.extract(state_prime)

        if self.task.stateisfinal(state_prime):
            terminal = True
        else:
            terminal = False

        r = self.task.reward(state, action, state_prime)
        _, q_primes = self.value_function.getqvalues(phi_prime)
        max_q_prime = np.max(q_primes)

        old = q_primes[action_index]
        q_primes[action_index] = r + self.gamma * max_q_prime
        loss = self.value_function.update(phi, q_primes)
        _, updated_values = self.value_function.getqvalues(phi)

        step_summary = "Loss %f r %f Old %f New %f Change %f" % (
            loss, r, old, updated_values[action_index], updated_values[action_index] - old)
        print(step_summary)

        self.current_cumulative_reward += r
        if terminal:
            ()
            # print(self._log_policy() + " " + str(self.current_cumulative_reward))

    def get_cumulative_reward(self) -> float:
        return self.current_cumulative_reward

    def episode_ended(self):
        # Observe final transition if needed
        self.current_cumulative_reward = 0.0
        self.previousaction = None
        self.previousstate = None


class QNetwork:
    def __init__(self, num_actions: int, num_state_features: int):
        self.n_s_feat = num_state_features
        self.n_actions = num_actions

        with tf.name_scope("qnetwork"):
            # State features
            self.inputs = tf.placeholder(shape=(1, self.n_s_feat), dtype=tf.float32)
            # A matrix with one entry per state-action pair, initialized to small values
            self.weight_matrix = tf.Variable(tf.random_uniform((self.n_s_feat, self.n_actions), 0, .01))
            # One q-value is a dot product of the state features with the action values. This mat multiply
            # produces all the q_values at once.
            self.q_out = tf.matmul(self.inputs, self.weight_matrix)
            self.argmax = tf.argmax(self.q_out, 1)

        with tf.name_scope("loss"):
            # All we need to train the network is a loss function. We'll use the MSE between some Q values
            # we would like our network to produce given an input. So, we'll give the network some the updated
            # q values (standard q update) and the state features for the state we just observed a reward from.
            self.target_qvalues = tf.placeholder(shape=[1, self.n_actions], dtype=tf.float32)
            self.loss = tf.reduce_sum(tf.square(self.target_qvalues - self.q_out))
            trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
            self.updateModel = trainer.minimize(self.loss)

        self.session = tf.Session()
        init = tf.initialize_all_variables()
        self.session.run(init)

    def getqvalues(self, state_features: List[float]):
        prepped = np.array(state_features).reshape([1, self.n_s_feat])
        feed = {self.inputs: prepped}
        max_action, q_values = self.session.run([self.argmax, self.q_out], feed_dict=feed)
        return max_action[0], q_values[0]

    def update(self, state_features: List[float], target: List[float]):
        prepped_state = np.array(state_features).reshape([1, self.n_s_feat])
        prepped_target = np.array(target).reshape([1, self.n_actions])
        feed = {self.inputs: prepped_state, self.target_qvalues: prepped_target}
        _, w, loss = self.session.run([self.updateModel, self.weight_matrix, self.loss], feed_dict=feed)
        return loss

    def closesession(self):
        self.session.close()
