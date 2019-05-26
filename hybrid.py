import numpy as np

from qroute import *
from base_policy import *
from config import *


class PolicyGradient(Policy):
    attrs = Policy.attrs + set(['Theta', 'discount'])

    def __init__(self, network, initP=InitP):
        super().__init__(network)
        self.Theta = {source:
                      np.ones((len(self.links), len(neighbors))) * initP
                      for source, neighbors in self.links.items()}

    def _softmax(self, source, dest):
        e_theta = np.exp(self.Theta[source][dest])
        return e_theta/e_theta.sum()

    def choose(self, source, dest):
        """ choose returns the choice following weighted random sample """
        choice = np.random.choice(
            self.links[source], p=self._softmax(source, dest))
        return choice

    def _gradient(self, source, dest, action_idx):
        """ gradient returns a vector with length of neibors of source """
        gradient = -self._softmax(source, dest)
        gradient[action_idx] += 1
        return gradient


class HybridQ(PolicyGradient, Qroute):
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

    def learn(self, rewards, lr={'q': LearnRateQ, 'p': LearnRateP, 'entropy': 0.001}):
        for reward in filter(lambda r: r.action != r.dest, rewards):
            source, dest, action = reward.source, reward.dest, reward.action
            info = reward.agent_info
            action_idx = self.links[source].index(action)
            softmax = self._softmax(source, dest)
            r = -info['q_y'] - lr['entropy'] * np.log(softmax[action_idx])
            old_score = self.Qtable[source][dest][action_idx]
            self.Qtable[source][dest][action_idx] += lr['q'] * \
                (r + self.discount * info['max_Q_y'] - old_score)
            gradient = softmax
            gradient[action_idx] += 1
            self.Theta[source][dest] += lr['p'] * gradient * \
                (r + self.discount * info['max_Q_y'] - info['max_Q_x_d'])
        for theta in self.Theta.values():
            np.clip(theta, -2, 2)


class HybridCDRQ(PolicyGradient, CDRQ):
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
