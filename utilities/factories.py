from enum import Enum
from typing import Tuple, List

from abjad.tools.durationtools.Duration import Duration
from abjad.tools.metertools import Meter
from abjad.tools.scoretools import Voice
from abjad.tools.tonalanalysistools import Scale

from counterpoint.composition_environment import CompositionEnvironment
from counterpoint.composition_parameters import CompositionParameters
from counterpoint.constants import soprano_range, tenor_range
from counterpoint.music_features import MusicFeatureExtractor
from counterpoint.species_counterpoint import CounterpointTask, SpeciesOneCounterpoint
from rl.agent.duelingqnetworkagent import DuelingQNetworkAgent
from rl.agent.qlearning import QLearning
from rl.agent.qnetworkagent import QNetworkAgent
from rl.agent.sarsa import Sarsa
from rl.agent.trueonlinesarsalambda import TrueOnlineSarsaLambda
from rl.domain import Domain
from rl.task import Task


class Approach(Enum):
    QLearning = 0
    Sarsa = 1
    TrueOnlineSarsaLambda = 2
    QNetwork = 3
    DDDQN = 4
    DDDRQN = 5


def make_environment_factory(given_voices: List[Voice], meter: Meter, scale: Scale,
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


def make_agent_factory(approach: Approach, initial_value=0.5, epsilon=0.9, alpha=0.5, lmbda=0.95):
    def generate_agent(domain, task, table):
        extractor = MusicFeatureExtractor(domain.composition_parameters.num_pitches_per_voice,
                                          domain.history_length)
        if approach == Approach.QLearning:
            agent = QLearning(domain, task, epsilon=epsilon, alpha=alpha)
        elif approach == Approach.Sarsa:
            agent = Sarsa(domain, task, epsilon=epsilon, alpha=alpha, lamb=lmbda, expected=False)
        elif approach == Approach.TrueOnlineSarsaLambda:
            agent = TrueOnlineSarsaLambda(domain, task, extractor, epsilon=epsilon, alpha=alpha, lamb=lmbda,
                                          expected=False)
        elif approach == Approach.QNetwork:
            agent = QNetworkAgent(domain, task, feature_extractor=extractor, epsilon=epsilon, alpha=alpha,
                                  value_function=table)
        elif approach == Approach.DDDQN:
            agent = DuelingQNetworkAgent(domain, task, extractor, epsilon=epsilon, alpha=alpha, value_function=table)

        return agent

    return generate_agent
