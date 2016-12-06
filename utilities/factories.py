from enum import Enum
from typing import Tuple

from abjad.tools.durationtools.Duration import Duration
from abjad.tools.metertools import Meter
from abjad.tools.tonalanalysistools import Scale

from counterpoint.composition_environment import CompositionEnvironment
from counterpoint.composition_parameters import CompositionParameters
from counterpoint.constants import soprano_range, tenor_range
from counterpoint.continuous_music_features import ContinuousMusicStateActionFeatureExtractor
from counterpoint.music_features import MusicStateFeatureExtractor
from counterpoint.species.species_one import SpeciesOneCounterpoint
from counterpoint.species_counterpoint import CounterpointTask
from rl.agent.duelingqnetworkagent import DuelingQNetworkAgent
from rl.agent.qlearning import QLearning
from rl.agent.qnetworkagent import QNetworkAgent
from rl.agent.sarsacmac import SarsaCMAC
from rl.agent.sarsavfa import SarsaVFA
from rl.agent.trueonlinesarsalambdavfa import TrueOnlineSarsaLambdaVFA
from rl.domain import Domain
from rl.task import Task


class Approach(Enum):
    QLearning = 0
    Sarsa = 1
    TrueOnlineSarsaLambda = 2
    QNetwork = 3
    DDDQN = 4
    DDDRQN = 5
    SarsaCMAC = 6,
    SarsaLinear = 7


    def __str__(self):
        if self == Approach.QLearning:
            return "Q-learning"
        elif self == Approach.Sarsa:
            return "Sarsa"
        elif self == Approach.SarsaCMAC:
            return "Sarsa CMAC"
        elif self == Approach.TrueOnlineSarsaLambda:
            return "True Online Sarsa(l)"
        elif self == Approach.QNetwork:
            return "QNetwork"
        elif self == Approach.DDDQN:
            return "Double Dueling DQN"
        elif self == Approach.DDDRQN:
            return "Double Dueling DRQN"


def make_environment_factory(meter: Meter, scale: Scale,
                             task_class: CounterpointTask = SpeciesOneCounterpoint, history_length: int = 2,
                             position_invariant=False):
    def generate_environment() -> Tuple[Domain, Task]:
        composition_parameters = CompositionParameters([("contrapuntal", soprano_range), ("cantus", tenor_range)],
                                                       meter,
                                                       scale, Duration(11))
        domain = CompositionEnvironment(composition_parameters, history_length=history_length,
                                        position_invariant=position_invariant)
        task = task_class(domain)
        return domain, task

    return generate_environment


def make_agent_factory(approach: Approach, initial_value=0.5, epsilon=0.9, alpha=0.5, lmbda=0.95,
                       time_invariant: bool = False):
    def generate_agent(domain, task, table):
        extractor = MusicStateFeatureExtractor(domain.composition_parameters.num_pitches_per_voice,
                                               domain.history_length, time_invariant)
        if approach == Approach.QLearning:
            agent = QLearning(domain, task, epsilon=epsilon, alpha=alpha)
        elif approach == Approach.SarsaLinear:
            agent = SarsaVFA(domain, task, extractor, epsilon=epsilon, alpha=alpha)
        elif approach == Approach.SarsaCMAC:
            extractor = ContinuousMusicStateActionFeatureExtractor(domain.composition_parameters.num_pitches_per_voice,
                                                                   domain.history_length, time_invariant)
            agent = SarsaCMAC(domain, task, extractor, epsilon=epsilon, alpha=alpha)
        elif approach == Approach.TrueOnlineSarsaLambda:
            agent = TrueOnlineSarsaLambdaVFA(domain, task, extractor, epsilon=epsilon, alpha=alpha, lamb=lmbda, )
        elif approach == Approach.QNetwork:
            agent = QNetworkAgent(domain, task, feature_extractor=extractor, epsilon=epsilon, alpha=alpha,
                                  value_function=table)
        elif approach == Approach.DDDQN:
            agent = DuelingQNetworkAgent(domain, task, extractor, epsilon=epsilon, alpha=alpha, value_function=table)

        return agent

    return generate_agent
