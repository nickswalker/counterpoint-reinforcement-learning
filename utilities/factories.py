from typing import Tuple, List

from abjad.tools.metertools import Meter
from abjad.tools.scoretools import Voice
from abjad.tools.tonalanalysistools import Scale

from counterpoint.composition_environment import CompositionEnvironment
from counterpoint.constants import soprano_range
from counterpoint.music_features import MusicFeatureExtractor
from counterpoint.species_counterpoint import CounterpointTask, SpeciesOneCounterpoint
from rl.agent.qlearning import QLearning
from rl.agent.qnetworkagent import QNetworkAgent
from rl.agent.sarsa import Sarsa
from rl.agent.trueonlinesarsalambda import TrueOnlineSarsaLambda
from rl.domain import Domain
from rl.task import Task


def make_environment_factory(given_voices: List[Voice], meter: Meter, scale: Scale,
                             task_class: CounterpointTask = SpeciesOneCounterpoint):
    def generate_environment() -> Tuple[Domain, Task]:
        domain = CompositionEnvironment(given_voices, [("contrapuntal", soprano_range)], meter, scale)
        task = task_class(domain)
        return domain, task

    return generate_environment


def make_agent_factory(initial_value=0.5,
                       epsilon=0.1,
                       alpha=0.5,
                       lmbda=0.95,
                       expected=False, true_online=False, q_learning=False, approximation=False, q_network=False):
    def generate_agent(domain, task):
        if true_online:
            agent = TrueOnlineSarsaLambda(domain, task, epsilon=epsilon, alpha=alpha, lamb=lmbda, expected=False)
        elif q_learning:
            agent = QLearning(domain, task)
        elif q_network:
            agent = QNetworkAgent(domain, task, MusicFeatureExtractor())
        else:
            if approximation:
                agent = Sarsa(domain, task, epsilon=epsilon, alpha=alpha, expected=expected)
            else:
                agent = TrueOnlineSarsaLambda(domain, task, MusicFeatureExtractor(), epsilon=epsilon, alpha=alpha,
                                              expected=expected)
        return agent

    return generate_agent
