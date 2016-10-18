from rl.domain import Domain


class Task(object):
    def __init__(self, domain: Domain):
        self.domain = domain

    def stateisfinal(self, state):
        raise NotImplementedError("Should have implemented this")

    def reward(self, state, action, state_prime):
        raise NotImplementedError("Should have implemented this")
