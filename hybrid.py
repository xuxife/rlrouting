import numpy as np

from qroute import *


class PolicyGradient(Policy):
    attrs = Policy.attrs | set(['Theta', 'discount'])

    def __init__(self, network, initP=0):
        super().__init__(network)
        self.Theta = {source:
                      np.random.normal(
                          initP, 1, (len(self.links), len(neighbors)))
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
    attrs = Qroute.attrs | PolicyGradient.attrs

    def __init__(self, network, add_entropy=True, initQ=0, initP=0, discount=0.99):
        self.discount = discount
        self.add_entropy = add_entropy
        self.limit = (-2, 2)
        PolicyGradient.__init__(self, network, initP)
        Qroute.__init__(self, network, initQ)

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
            gradient = -softmax
            gradient[action_idx] += 1
            self.Theta[source][dest] += lr['p'] * gradient * \
                (r + self.discount * info['max_Q_y'] - info['max_Q_x_d'])
        for theta in self.Theta.values():
            theta.clip(*self.limit, out=theta)


class HybridCDRQ(PolicyGradient, CDRQ):
    attrs = CDRQ.attrs | PolicyGradient.attrs

    def __init__(self, network, add_entropy=False, initQ=0, initP=0, decay=0.9, discount=0.99):
        self.discount = discount
        self.add_entropy = add_entropy
        PolicyGradient.__init__(self, network, initP)
        CDRQ.__init__(self, network, decay=decay, initQ=initQ)

    def get_reward(self, source, action, packet):
        info = CDRQ.get_reward(self, source, action, packet)
        info['max_Q_x_d'] = self.Qtable[source][packet.dest].max()
        info['max_Q_y_s'] = self.Qtable[action][packet.source].max()
        return info

    def learn(self, rewards, lr={'f': 0.85, 'b': 0.95, 'q': 0.1, 'p': 0.1, 'e': 0.1}):
        for reward in rewards:
            x, y = reward.source, reward.action
            source, dest = reward.packet.source, reward.packet.dest
            info = reward.agent_info
            x_idx, y_idx = self.action_idx[y][x], self.action_idx[x][y]
            softmax_f = self._softmax(x, dest)
            softmax_b = self._softmax(y, source)
            if self.add_entropy:
                r_f = -info['q_y'] - lr['e'] * np.log2(softmax_f[y_idx])
                r_b = -info['q_x'] - lr['e'] * np.log2(softmax_b[x_idx])
            else:
                r_f, r_b = -info['q_y'], -info['q_x']
            if y != dest:
                old_Q_f = self.Qtable[x][dest][y_idx]
                eta_f = max(info['C_f'], 1-self.confidence[x][dest][y_idx])
                self.Qtable[x][dest][y_idx] += lr['f'] * eta_f * \
                    (r_f + self.discount * info['max_Q_f'] - old_Q_f)
                self.confidence[x][dest][y_idx] += eta_f * \
                    (info['C_f']-self.confidence[x][dest][y_idx])
                self.updated_conf[x][dest][y_idx] = True
                gradient_f = -softmax_f
                gradient_f[y_idx] += 1
                self.Theta[x][dest] += lr['p'] * gradient_f * \
                    (r_f + self.discount * info['max_Q_f'] - info['max_Q_x_d'])
            if x != source:
                old_Q_b = self.Qtable[y][source][x_idx]
                eta_b = max(info['C_b'], 1-self.confidence[y][source][x_idx])
                self.Qtable[y][source][x_idx] += lr['b'] * eta_b * \
                    (r_b + self.discount * info['max_Q_b'] - old_Q_b)
                self.confidence[y][source][x_idx] += eta_b * \
                    (info['C_b']-self.confidence[y][source][x_idx])
                self.updated_conf[y][source][x_idx] = True
                gradient_b = -softmax_b
                gradient_b[x_idx] += 1
                self.Theta[y][source] += lr['p'] * gradient_b * \
                    (r_b + self.discount * info['max_Q_b'] - info['max_Q_y_s'])

        for node, table in self.updated_conf.items():
            self.confidence[node][~table] *= self.decay
            table.fill(False)
            table[self.links[node]] = np.eye(table.shape[1], dtype=bool)
