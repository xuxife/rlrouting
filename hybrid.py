import numpy as np

from qroute import *
from base_policy import *
from config import *


class PolicyGradient(Policy):
    attrs = Policy.attrs + ['Theta', 'discount']

    def __init__(self, network, initP=InitP):
        self.links = network.links
        self.Theta = {source:
                      np.ones((len(self.links), len(neighbors))) * initP
                      for source, neighbors in self.links.items()}

    def choose(self, source, dest):
        """ choose returns the choice following weighted random sample """
        e_theta = np.exp(self.Theta[source][dest])
        choice = np.random.choice(
            self.links[source], p=e_theta/e_theta.sum())
        return choice

    def _gradient(self, source, dest, action):
        """ gradient returns a vector with length of neibors of source """
        e_theta = np.exp(self.Theta[source][dest])
        gradient = - e_theta/e_theta.sum()
        gradient[self.links[source].index(action)] += 1
        return gradient


class HybridQ(Qroute, PolicyGradient):
    attrs = Qroute.attrs + PolicyGradient.attrs

    def __init__(self, network, initQ=InitQ, initP=InitP, discount=Discount):
        self.discount = discount
        Qroute.__init__(self, network, initQ)
        PolicyGradient.__init__(self, network, initP)

    def get_reward(self, source, action, packet):
        return {
            'max_Q_y': self.Qtable[action][packet.dest].max(),
            'max_Q_x_d': self.Qtable[source][packet.dest].max(),
        }

    def learn(self, rewards, lr={'q': LearnRateQ, 'p': LearnRateP}):
        for reward in filter(lambda r: r.action != r.dest, rewards):
            source, dest, action = reward.source, reward.dest, reward.action
            agent_info = reward.agent_info
            action_index = self.links[source].index(action)
            old_score = self.Qtable[source][dest][action_index]
            self.Qtable[source][dest][action_index] += lr['q'] * \
                (-agent_info['q_y'] + self.discount *
                 agent_info['max_Q_y'] - old_score)
            self.Theta[source][dest] += lr['p'] * \
                (-agent_info['q_y'] + self.discount*agent_info['max_Q_y'] - agent_info['max_Q_x_d']) * \
                self._gradient(source, dest, action)
        for theta in self.Theta.values():
            np.clip(theta, -2, 2)


class HybridCDRQ(CDRQ, PolicyGradient):
    attrs = CDRQ.attrs + PolicyGradient.attrs

    def __init__(self, network, initQ=InitQ, initP=InitP, discount=Discount):
        self.discount = discount
        CDRQ.__init__(self, network, initQ=initQ)
        PolicyGradient.__init__(self, network, initP)

    def get_reward(self, source, action, packet):
        info = CDRQ.get_reward(self, source, action, packet)
        info['max_Q_x_d'] = self.Qtable[source][packet.dest].max()
        return info

    def learn(self, rewards, lr={'f': 0.85, 'b': 0.95, 'q': LearnRateQ, 'p': LearnRateP}):
        pass
