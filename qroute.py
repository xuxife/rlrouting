import numpy as np

from base_policy import Policy


class Qroute(Policy):
    attrs = Policy.attrs | set(['Qtable'])

    def __init__(self, network, initQ=0):
        super().__init__(network)
        self.Qtable = {source:
                       np.random.normal(
                           initQ, 1, (len(self.links), len(neighbors)))
                       for source, neighbors in self.links.items()}
        for source, table in self.Qtable.items():
            # Q_x(z, x) = 0, forall z in x.neighbors (not useful)
            table[source] = 0
            # Q_x(z, y) = -1 if z == y else 0
            table[self.links[source]] = -np.eye(table.shape[1])

    def choose(self, source, dest, score=False):
        scores = self.Qtable[source][dest]
        score_max = scores.max()
        choice = np.random.choice(np.argwhere(scores == score_max).flatten())
        return (choice, score_max) if score else self.links[source][choice]

    def get_info(self, source, action, packet):
        return {'max_Q_y': self.Qtable[action][packet.dest].max()}

    def learn(self, rewards, lr={'q': 0.1}):
        for reward in rewards:
            source, dest, action = reward.source, reward.dest, reward.action
            info = reward.agent_info
            r = -info['q_y'] - info['t_y']
            action_idx = self.action_idx[source][action]
            old_score = self.Qtable[source][dest][action_idx]
            self.Qtable[source][dest][action_idx] += lr['q'] * \
                (r + info['max_Q_y'] - old_score)


class CQ(Qroute):
    attrs = Qroute.attrs | set(['confidence'])

    def __init__(self, network, decay=0.9, initQ=0):
        super().__init__(network, initQ)
        self.decay = decay
        self.confidence = {source:
                           np.zeros((len(self.links), len(neighbors)))
                           for source, neighbors in self.links.items()}
        for source, neighbors in self.links.items():
            # base case: C_x(z, y) = 1 if z == y else 0
            self.confidence[source][neighbors] = np.eye(len(neighbors))

    def get_info(self, source, action, packet):
        z, max_Q = self.choose(action, packet.dest, score=True)
        return {
            'max_Q_f': max_Q,
            'C_f': self.confidence[action][packet.dest][z]
        }

    def learn(self, rewards, lr={}):
        for reward in rewards:
            x, y = reward.source, reward.action
            dest = reward.packet.dest
            info = reward.agent_info
            y_idx = self.action_idx[x][y]
            if y == dest:
                # in this case, the decision of sending to the destination is undoubtedly correct
                # or say, eta = 0
                continue
            r = -info['q_y']-info['t_y']
            old_Q = self.Qtable[x][dest][y_idx]
            old_conf = self.confidence[x][dest][y_idx]
            eta = max(info['C_f'], 1-old_conf)
            self.Qtable[x][dest][y_idx] += eta * \
                (r + info['max_Q_f'] - old_Q)
            self.confidence[x][dest][y_idx] += eta * (info['C_f']-old_conf)
            # counteract the effect of confidence_decay()
            self.confidence[x][dest][y_idx] /= self.decay
        self.confidence_decay()

    def confidence_decay(self):
        for table in self.confidence.values():
            table *= self.decay


class CDRQ(CQ):
    mode = 'dual'

    def get_info(self, source, action, packet):
        z_b, max_Q_b = self.choose(source, packet.source, score=True)
        z_f, max_Q_f = self.choose(action, packet.dest, score=True)
        return {
            'max_Q_b': max_Q_b,
            'max_Q_f': max_Q_f,
            'C_b': self.confidence[source][packet.source][z_b],
            'C_f': self.confidence[action][packet.dest][z_f],
        }

    def learn(self, rewards, lr={}):
        for reward in rewards:
            x, y = reward.source, reward.action
            source, dest = reward.packet.source, reward.packet.dest
            info = reward.agent_info
            x_idx, y_idx = self.action_idx[y][x], self.action_idx[x][y]
            " forward "
            if y != dest:
                r_f = -info['q_y']-info['t_y']
                old_Q_f = self.Qtable[x][dest][y_idx]
                old_conf_f = self.confidence[x][dest][y_idx]
                eta_f = max(info['C_f'], 1-old_conf_f)
                self.Qtable[x][dest][y_idx] += eta_f * \
                    (r_f + info['max_Q_f'] - old_Q_f)
                self.confidence[x][dest][y_idx] += eta_f * \
                    (info['C_f']-old_conf_f)
                self.confidence[x][dest][y_idx] /= self.decay
            " backward "
            if x != source:
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
