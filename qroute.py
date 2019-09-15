import numpy as np

from base_policy import Policy


class Qroute(Policy):
    attrs = Policy.attrs | set(['Qtable'])

    def __init__(self, network, initQ=0, discount=0.99):
        super().__init__(network)
        self.discount = discount
        self.Qtable = {x: np.random.normal(
            initQ, 1, (len(self.links), len(ys)))
            for x, ys in self.links.items()}
        for x, table in self.Qtable.items():
            # Q_x(z, x) = 0, forall z in x.neighbors
            table[x] = 0
            # Q_x(z, y) = -1 if z == y else 0
            table[self.links[x]] = -np.eye(table.shape[1])

    def choose(self, source, dest, idx=False):
        scores = self.Qtable[source][dest]
        # score_max = scores.max()
        # choice = np.random.choice(np.argwhere(scores == score_max).flatten())
        choice = np.argmax(scores)
        return (choice, scores.max()) if idx else self.links[source][choice]

    def get_info(self, source, action, packet):
        return {'max_Q_y': self.Qtable[action][packet.dest].max()}

    def _extract(self, reward):
        " s -> ... -> w -> x -> y -> z -> ... -> d"
        "                  | (current at x)       "
        x, y, d = reward.source, reward.action, reward.dest
        info = reward.agent_info
        r = -info['q_y'] - info['t_y']
        return r, info, x, y, d

    def learn(self, rewards, lr={'q': 0.1}):
        for reward in rewards:
            r, info, x, y, d = self._extract(reward)
            y_idx = self.action_idx[x][y]
            old_score = self.Qtable[x][d][y_idx]
            self.Qtable[x][d][y_idx] += lr['q'] * \
                (r + self.discount * info['max_Q_y'] - old_score)


class CQ(Qroute):
    attrs = Qroute.attrs | set(['confidence'])

    def __init__(self, network, decay=0.9, initQ=0):
        super().__init__(network, initQ)
        self.decay = decay
        self.confidence = {x: np.zeros_like(
            self.Qtable[x]) for x in self.Qtable.keys()}
        # the decision of sending to the destination is undoubtedly correct
        for x, ys in self.links.items():
            # base case: C_x(z, y) = 1 if z == y else 0
            self.confidence[x][ys] = np.eye(len(ys))

    def get_info(self, source, action, packet):
        z_idx, max_Q_f = self.choose(action, packet.dest, idx=True)
        return {
            'max_Q_f': max_Q_f,
            'C_f': self.confidence[action][packet.dest][z_idx]
        }

    def learn(self, rewards, lr={}):
        for reward in rewards:
            r, info, x, y, d = self._extract(reward)
            y_idx = self.action_idx[x][y]
            old_Q = self.Qtable[x][d][y_idx]
            old_conf = self.confidence[x][d][y_idx]
            eta = max(info['C_f'], 1-old_conf)
            self.Qtable[x][d][y_idx] += eta * \
                (r + info['max_Q_f'] - old_Q)
            self.confidence[x][d][y_idx] += eta * (info['C_f']-old_conf)
            # counteract the effect of confidence_decay()
            self.confidence[x][d][y_idx] /= self.decay
        self.confidence_decay()

    def confidence_decay(self):
        for table in self.confidence.values():
            table *= self.decay


class CDRQ(CQ):
    mode = 'dual'

    def get_info(self, source, action, packet):
        w_idx, max_Q_b = self.choose(source, packet.source, idx=True)
        z_idx, max_Q_f = self.choose(action, packet.dest, idx=True)
        return {
            'max_Q_b': max_Q_b,
            'max_Q_f': max_Q_f,
            'C_b': self.confidence[source][packet.source][w_idx],
            'C_f': self.confidence[action][packet.dest][z_idx],
        }

    def learn(self, rewards, lr={}):
        for reward in rewards:
            r_f, info, x, y, dest = self._extract(reward)
            x_idx, y_idx = self.action_idx[y][x], self.action_idx[x][y]
            " forward "
            old_Q_f = self.Qtable[x][dest][y_idx]
            old_conf_f = self.confidence[x][dest][y_idx]
            eta_f = max(info['C_f'], 1-old_conf_f)
            self.Qtable[x][dest][y_idx] += eta_f * \
                (r_f + info['max_Q_f'] - old_Q_f)
            self.confidence[x][dest][y_idx] += eta_f * \
                (info['C_f']-old_conf_f)
            self.confidence[x][dest][y_idx] /= self.decay
            " backward "
            source = reward.packet.source
            r_b = -info['q_x']-info['t_x']
            old_Q_b = self.Qtable[y][source][x_idx]
            old_conf_b = self.confidence[y][source][x_idx]
            eta_b = max(info['C_b'], 1-old_conf_b)
            self.Qtable[y][source][x_idx] += eta_b * \
                (r_b + info['max_Q_b'] - old_Q_b)
            self.confidence[y][source][x_idx] += eta_b * \
                (info['C_b']-old_conf_b)
            self.confidence[y][source][x_idx] /= self.decay

        self.confidence_decay()
