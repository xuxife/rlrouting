import numpy as np

from qroute import *
from base_policy import *


class PolicyGradient(Policy):
    attrs = Policy.attrs + set(['Theta', 'discount'])

    def __init__(self, network, initP=0):
        super().__init__(network)
        self.Theta = {source:
                      np.full((len(self.links), len(neighbors)), initP)
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
        """ gradient returns a vector with length of neighbors of source """
        gradient = -self._softmax(source, dest)
        gradient[action_idx] += 1
        return gradient


class HybridQ(PolicyGradient, Qroute):
    attrs = Qroute.attrs + PolicyGradient.attrs

    def __init__(self, network, add_entropy=True, initQ=0, initP=0, discount=0.99):
        self.discount = discount
        self.add_entropy = add_entropy
        self.limit = (-700, 700)
        Qroute.__init__(self, network, initQ)
        PolicyGradient.__init__(self, network, initP)

    def choose(self, source, dest):
        return PolicyGradient.choose(self, source, dest)

    def get_reward(self, source, action, packet):
        return {
            'max_Q_y': self.Qtable[action][packet.dest].max(),
            'max_Q_x_d': self.Qtable[source][packet.dest].max(),
        }

    def learn(self, rewards, lr={'q': 0.1, 'p': 0.1, 'e': 0.1}):
        for reward in filter(lambda r: r.action != r.dest, rewards):
            source, dest, action = reward.source, reward.dest, reward.action
            info = reward.agent_info
            action_idx = self.links[source].index(action)
            softmax = self._softmax(source, dest)
            if self.add_entropy:
                r = -info['q_y'] - lr['e'] * np.log2(softmax[action_idx])
            else:
                r = -info['q_y']
            old_score = self.Qtable[source][dest][action_idx]
            self.Qtable[source][dest][action_idx] += lr['q'] * \
                (r + self.discount * info['max_Q_y'] - old_score)
            gradient = softmax
            gradient[action_idx] += 1
            self.Theta[source][dest] += lr['p'] * gradient * \
                (r + self.discount * info['max_Q_y'] - info['max_Q_x_d'])
        for theta in self.Theta.values():
            _ = theta.clip(*self.limit, out=theta)


class HybridCDRQ(PolicyGradient, CDRQ):
    attrs = CDRQ.attrs + PolicyGradient.attrs

    def __init__(self, network, initQ=0, initP=0, discount=0.99):
        self.discount = discount
        CDRQ.__init__(self, network, initQ=initQ)
        PolicyGradient.__init__(self, network, initP)

    def choose(self, source, dest):
        return PolicyGradient.choose(self, source, dest)

    def get_reward(self, source, action, packet):
        info = CDRQ.get_reward(self, source, action, packet)
        info['max_Q_x_d'] = self.Qtable[source][packet.dest].max()
        return info

    def learn(self, rewards, lr={'f': 0.85, 'b': 0.95, 'q': 0.1, 'p': 0.1}):
        pass
