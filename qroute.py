import numpy as np

from base_policy import *
from config import *


class Qroute(Policy):
    """ Qroute use Q routing as policy. 

    Attritubes:
        Qtable : Stores the Q scores of all nodes.
    """
    attrs = Policy.attrs + ['Qtable']

    def __init__(self, network, initQ=InitQ):
        self.links = network.links
        self.Qtable = {source:
                       np.random.normal(
                           initQ, 1, (len(self.links), len(neighbors)))
                       for source, neighbors in self.links.items()}
        for source, table in self.Qtable.items():
            # Q_x(z, x) = 0, forall z in x.neighbors (not useful)
            table[source] = 0
            # Q_x(z, y) = -1 if z == y else 0
            table[self.links[source]] = -np.eye(table.shape[1]) * TransTime

    def choose(self, source, dest):
        scores = self.Qtable[source][dest]
        score_max = scores.max()
        choice = np.random.choice(np.argwhere(scores == score_max).flatten())
        return self.links[source][choice]

    def get_reward(self, source, action, packet):
        return {'max_Q_y': self.Qtable[action][packet.dest].max()}

    def learn(self, rewards, lr={'q': LearnRateQ}):
        for reward in filter(lambda r: r.action != r.dest, rewards):
            source, dest, action = reward.source, reward.dest, reward.action
            action_max = reward.agent_info['max_Q_y']
            action_index = self.links[source].index(action)
            old_score = self.Qtable[source][dest][action_index]
            self.Qtable[source][dest][action_index] += lr['q'] * \
                (-reward.agent_info['q_y'] + action_max - old_score)


class CDRQ(Qroute):
    attrs = Qroute.attrs + ['confidence']

    def __init__(self, network, decay=0.9, initQ=InitQ):
        super().__init__(network, initQ)
        network.dual = True  # enable DUAL mode
        self.decay = decay
        self.confidence = {source:
                           np.zeros((len(self.links), len(neighbors)))
                           for source, neighbors in self.links.items()}
        for source, table in self.confidence.items():
            # C_x(z, y) = 1 if z == y else 0
            table[self.links[source]] = np.eye(len(self.links[source]))
        self.is_updated_confi = {source:
                                 np.zeros(
                                     (len(self.links), len(neighbors)), dtype=bool)
                                 for source, neighbors in self.links.items()}
        for source, table in self.is_updated_confi.items():
            table[self.links[source]] = np.eye(table.shape[1], dtype=bool)

    def _choose(self, source, dest):
        " same to choose but return different "
        scores = self.Qtable[source][dest]
        score_max = scores.max()
        choice = np.random.choice(np.argwhere(scores == score_max).flatten())
        return choice, score_max

    def get_reward(self, source, action, packet):
        z_x, max_Q_x = self._choose(source, packet.source)
        z_y, max_Q_y = self._choose(action, packet.dest)
        return {
            'max_Q_x': max_Q_x,
            'max_Q_y': max_Q_y,
            'C_x': self.confidence[source][packet.source][z_x],
            'C_y': self.confidence[action][packet.dest][z_y],
        }

    def learn(self, rewards, lr={'f': 0.85, 'b': 0.95}):
        for reward in rewards:
            x, y = reward.source, reward.action
            source, dest = reward.packet.source, reward.packet.dest
            agent_info = reward.agent_info
            x_index, y_index = self.links[y].index(x), self.links[x].index(y)
            " forward "
            if y != dest:
                old_Q_x = self.Qtable[x][dest][y_index]
                eta_forward = max(
                    agent_info['C_y'], 1-self.confidence[x][dest][y_index])
                self.Qtable[x][dest][y_index] += lr['f'] * eta_forward * \
                    (agent_info['max_Q_y'] - agent_info['q_y'] - old_Q_x)
                self.confidence[x][dest][y_index] += eta_forward * \
                    (agent_info['C_y']-self.confidence[x][dest][y_index])
                self.is_updated_confi[x][dest][y_index] = True
            " backward "
            if x != source:
                old_Q_y = self.Qtable[y][source][x_index]
                eta_backward = max(
                    agent_info['C_x'], 1-self.confidence[y][source][x_index])
                self.Qtable[y][source][x_index] += lr['b'] * eta_backward * \
                    (agent_info['max_Q_x'] - agent_info['q_x'] - old_Q_y)
                self.confidence[y][source][x_index] += eta_backward * \
                    (agent_info['C_x']-self.confidence[y][source][x_index])
                self.is_updated_confi[y][source][x_index] = True

        for source, table in self.is_updated_confi.items():
            self.confidence[source][~table] *= self.decay
            table.fill(False)
            table[self.links[source]] = np.eye(table.shape[1], dtype=bool)
