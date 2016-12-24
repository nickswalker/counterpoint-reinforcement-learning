import random

from rl.action import Action
from rl.agent.agent import Agent
from rl.domain import Domain
from rl.state import State
from rl.task import Task
from rl.valuefunction.tabular import StateActionValueTable


class TrueOnlineSarsaLambda(Agent):
    def __init__(self, domain: Domain, task: Task, epsilon=0.1, alpha=0.6, gamma=1.0, lamb=0.95, expected=False,
                 feature_extractor=None,
                 name="Sarsa"):
        """
        :param domain: The world the agent is placed in.
        :param task: The task in the world, which defines the reward function.
        """
        super().__init__(domain, task, name)
        self.world = domain
        self.task = task
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb
        self.expected = expected
        self.previousaction = None
        self.previousstate = None

        self.value_function = StateActionValueTable(domain.get_actions())
        self.current_cumulative_reward = 0.0
        self.eligibility = {}
        self.v_old = 0

    def act(self):
        """Execute one action on the world, possibly terminating the episode.

        """
        state = self.domain.get_current_state()
        action = self.choose_action(state)

        self.world.apply_action(action)

        # For the first time step, we won't have received a reward yet.
        # We're just notifying the learner of our starting state and action.
        if self.previousstate is None and self.previousaction is None:
            ()
        else:
            self.update(self.previousstate, self.previousaction, state, action)

        self.previousaction = action
        self.previousstate = state

        state_prime = self.world.get_current_state()
        if self.task.stateisfinal(state_prime):
            self.update(state, action, state_prime, None, terminal=True)

    def update(self, state: State, action: Action, state_prime: State, action_prime: Action, terminal=False):
        reward = self.task.reward(state, action, state_prime)
        v_s = self.value_function.actionvalue(state, action)
        if terminal:
            v_s_prime = 0
        else:
            if self.expected:
                v_s_prime = self.expected_value(state_prime)
            else:
                v_s_prime = self.value_function.actionvalue(state_prime, action_prime)
        delta_v = v_s - self.v_old
        self.v_old = v_s_prime
        # Terminal states are defined to have value 0

        delta = reward + self.gamma * v_s_prime - v_s

        s_a = (state, action)
        visited_e_update = (1.0 - self.alpha) * self.eligibility.get(s_a, 0.0) + 1
        self.eligibility[s_a] = visited_e_update

        for pair, e_value in self.eligibility.items():
            trace_v_s = self.value_function.actionvalue(pair[0], pair[1])
            new_value = trace_v_s + self.alpha * (delta + delta_v) * e_value
            self.value_function.setactionvalue(pair[0], pair[1], new_value)
            self.eligibility[pair] = self.gamma * self.lamb * e_value
        # Value of state was updated by the trace
        v_s = self.value_function.actionvalue(state, action)
        new_value = v_s - self.alpha * delta_v
        self.value_function.setactionvalue(state, action, new_value)
        self.current_cumulative_reward += reward

    def choose_action(self, state) -> Action:
        """Given a state, pick an action according to an epsilon-greedy policy.

        :param state: The state from which to act.
        :return:
        """
        if random.random() < self.epsilon:
            actions = self.domain.get_actions()
            return random.sample(actions, 1)[0]
        else:
            best_actions = self.value_function.bestactions(state)
            return random.sample(best_actions, 1)[0]

    def expected_value(self, state):
        actions = self.domain.get_actions(state)
        expectation = 0.0
        best_actions = self.value_function.bestactions(state)
        num_best_actions = len(best_actions)
        nonoptimal_mass = self.epsilon

        if num_best_actions > 0:
            a_best_action = random.sample(best_actions, 1)[0]
            greedy_mass = (1.0 - self.epsilon)
            expectation += greedy_mass * self.value_function.actionvalue(state, a_best_action)
        else:
            nonoptimal_mass = 1.0

        if nonoptimal_mass > 0.0:
            # No best action, equiprobable random policy
            total_value = 0.0
            for action in actions:
                total_value += self.value_function.actionvalue(state, action)
            expectation += nonoptimal_mass * total_value / len(actions)

        return expectation

    def get_cumulative_reward(self):
        return self.current_cumulative_reward

    def episode_ended(self):
        # Observe final transition if needed
        self.current_cumulative_reward = 0.0
        self.previousaction = None
        self.previousstate = None
        self.epsilon *= .9999
        self.eligibility = {}
        self.v_old = 0

    def logepisode(self):
        print("Episode reward: " + str(self.current_cumulative_reward))
