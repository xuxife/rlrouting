import numpy as np

from base_policy import *
from config import *


class Qroute(Policy):
    """ Qroute use Q routing as policy. 

    Attritubes:
        Qtable : Stores the Q scores of all nodes.
    """
    attrs = ['Qtable']

    def __init__(self, network, initQ=InitQ):
        self.links = network.links
        self.Qtable = {source:
                       np.ones((len(self.links), len(neighbors)))
                       for source, neighbors in self.links.items()}

    def choose(self, source, dest):
        scores = self.Qtable[source][dest]
        score_max = scores.max()
        choice = np.random.choice(np.argwhere(scores == score_max).flatten())
        return self.links[source][choice]

    def get_reward(self, source, action, packet):
        return {'max_Q_y': self.Qtable[action][packet.dest].max()}

    def learn(self, rewards, lr={'q': LearnRateQ}):
        for reward in rewards:
            q, t = reward.queue_time, reward.trans_time
            source, dest, action = reward.source, reward.dest, reward.action
            action_max = reward.agent_info['max_Q_y']
            action_index = self.links[source].index(action)
            old_score = self.Qtable[source][dest][action_index]
            self.Qtable[source][dest][action_index] += lr['q'] * \
                (-q-t + action_max - old_score)


