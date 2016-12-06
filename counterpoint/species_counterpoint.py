from abc import abstractmethod

from abjad.tools.durationtools.Duration import Duration

from counterpoint.composition_environment import CompositionEnvironment, CompositionState
from rl.task import Task


class CounterpointTask(Task):
    def __init__(self, domain: CompositionEnvironment):
        super().__init__(domain)
        self.domain = domain

    def stateisfinal(self, state: CompositionState):
        if self.domain.composition_parameters.duration <= state.preceding_duration:
            return True
        return False

    def reward(self, state, action, state_prime):
        return 0


class GradedCounterpointTask(CounterpointTask):
    def __init__(self, domain: CompositionEnvironment, differential: bool = True):
        super().__init__(domain)
        self.domain = domain
        self.differential = differential
        self.prev_duration = Duration(0)
        self.prev_grade = 0.0

    @abstractmethod
    def grade_composition(self, composition):
        return 0

    def reward(self, state, action, state_prime):
        if not self.differential:
            if state_prime.preceding_duration == Duration(11):
                return self.grade_composition(self.domain)
            else:
                return 0
        if self.prev_duration == state_prime.preceding_duration:
            assert False
            return
        elif self.prev_duration < state_prime.preceding_duration:
            new_grade = self.grade_composition(self.domain)
            reward = new_grade - self.prev_grade
            self.prev_grade = new_grade
        elif self.prev_duration > state_prime.preceding_duration:
            # This should only be the first step
            reward = self.grade_composition(self.domain)
            self.prev_grade = reward
        self.prev_duration = state_prime.preceding_duration
        return reward


