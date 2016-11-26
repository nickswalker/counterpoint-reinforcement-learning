from typing import List

import numpy as np
import tensorflow as tf

from rl.action import Action
from rl.agent.agent import Agent
from rl.domain import Domain
from rl.task import Task
from rl.valuefunction import FeatureExtractor


class DuelingQNetworkAgent(Agent):
    def __init__(self, domain: Domain, task: Task, feature_extractor: FeatureExtractor, epsilon=0.1, alpha=0.1,
                 gamma=0.95):
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.alpha = alpha / domain.history_length
        self.gamma = gamma
        self.feature_extractor = feature_extractor

        num_actions = len(domain.get_actions())
        num_state_features = feature_extractor.num_features()

        self.value_function = DuelingQNetworkValueFunction(num_actions, num_state_features, self.alpha)

        self.world = domain
        self.task = task

        self.current_cumulative_reward = 0.0
        self.previousaction = None
        self.previousstate = None
        self.name = "Double Dueling DRQN"

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

        q_primes[action_index] = r + self.gamma * max_q_prime
        loss = self.value_function.update(phi, q_primes)
        _, updated_values = self.value_function.getqvalues(phi)
        self.epsilon *= 0.9999
        if terminal:
            step_summary = "Loss %.2f r %d" % (loss, self.current_cumulative_reward)
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


class DuelingQNetworkValueFunction:
    def __init__(self, num_actions: int, num_state_features: int, learning_rate: float, n_hidden: int = 512):
        self.n_s_feat = num_state_features
        self.n_actions = num_actions
        self.tau = 0.1
        half_hidden = int(n_hidden / 2)
        with tf.name_scope("qnetwork"):
            # State features go in...
            self.inputs = tf.placeholder(shape=(1, self.n_s_feat), dtype=tf.float32)
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
            self.target_qvalues = tf.placeholder(shape=[1, self.n_actions], dtype=tf.float32)

            self.td_error = tf.square(self.target_qvalues - self.q_out)
            self.loss = tf.reduce_sum(tf.square(self.target_qvalues - self.q_out))
            trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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
        _, loss = self.session.run([self.updateModel, self.loss], feed_dict=feed)
        return loss

    def updateTargetGraph(self, tfVars, tau):
        total_vars = len(tfVars)
        op_holder = []
        for idx, var in enumerate(tfVars[0:total_vars / 2]):
            op_holder.append(tfVars[idx + total_vars / 2].assign(
                (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars / 2].value())))
        return op_holder

    def updateTarget(self, op_holder, sess):
        for op in op_holder:
            sess.run(op)

    def closesession(self):
        self.session.close()
