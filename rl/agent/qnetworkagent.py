from typing import List

import numpy as np
import tensorflow as tf

from rl import domain
from rl.action import Action
from rl.agent.agent import Agent
from rl.domain import Domain
from rl.state import State
from rl.task import Task
from rl.valuefunction import FeatureExtractor


class QNetworkAgent(Agent):
    def __init__(self, domain: Domain, task: Task, feature_extractor: FeatureExtractor, epsilon=0.1, alpha=0.6,
                 gamma=0.95):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.feature_extractor = feature_extractor
        self.value_function = QNetwork(2)

        self.world = domain
        self.task = task

        self.current_cumulative_reward = 0.0
        self.previousaction = None
        self.previousstate = None

    def choose_action(self, state: State) -> Action:
        state_features = self.feature_extractor.extract(state)
        max_action, qvalues = self.value_function.getqvalues(state_features)
        a = max_action
        if np.random.rand(1) < self.epsilon:
            a = domain.get_actions(state).sample()
        return a

    def act(self):
        state = self.world.get_current_state()
        action = self.choose_action(state)

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
        q_primes = self.value_function.getqvalues(phi_prime)
        max_q_prime = np.max(q_primes)
        q_primes[action] = r + self.gamma * max_q_prime
        self.value_function.update(phi_prime, q_primes)

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


class QNetwork():
    def __init__(self, num_voices: int):
        n_s_feat = 12 * num_voices
        n_actions = pow(12, num_voices)

        self.inputs = tf.placeholder(shape=(1, n_s_feat), dtype=tf.float32)
        self.weight_matrix = tf.Variable(tf.random_uniform((n_s_feat, n_actions), 0, .01))
        self.q_out = tf.matmul(self.inputs, self.weight_matrix)
        self.argmax = tf.argmax(self.q_out, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.next_qvalues = tf.placeholder(shape=[1, n_actions], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.next_qvalues - self.q_out))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.updateModel = trainer.minimize(loss)

        self.session = tf.Session()

    def getqvalues(self, state_features: List[float]):
        assert len(state_features) == 24
        feed = {self.inputs: np.array(state_features)}
        max_action, q_values = self.session.run([self.argmax, self.q_out], feed_dict=feed)
        return max_action, q_values

    def update(self, state_features: List[float], target: List[float]):
        feed = {self.inputs: np.array(state_features), self.next_qvalues: target}
        self.session.run([self.updateModel, self.weight_matrix], feed_dict=feed)

    def closesession(self):
        self.session.close()
