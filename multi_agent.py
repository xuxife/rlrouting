import numpy as np

from hybrid import *


class MaHybridQ(HybridQ):
    """ Multi-agent Hybrid Q routing with Eligibility Traces """
    attrs = HybridQ.attrs | set(['discount_trace', 'Trace'])

    def __init__(self, network, initQ=0, initP=0, discount=0.99, discount_trace=0.6):
        super().__init__(network, initQ=initQ, initP=initP, discount=discount)
        self.discount_trace = discount_trace
        self.reward_shape = 0
        self.Trace = {k: np.zeros_like(v, dtype=np.float64)
                      for k, v in self.Theta.items()}

    def learn(self, rewards, lr={'q': 0.1, 'p': 0.1}):
        r_len = len(rewards)
        x, dest, y_idx = np.zeros(r_len, dtype=np.int), np.zeros(
            r_len, dtype=np.int), np.zeros(r_len, dtype=np.int)
        r, max_Q_y, max_Q_x_d = np.zeros(r_len), np.zeros(r_len), np.zeros(r_len)
        for i, reward in enumerate(rewards):
            r[i], info, x[i], y, dest[i] = self._extract(reward)
            y_idx[i] = self.action_idx[x[i]][y]
            max_Q_y[i] = info['max_Q_y']
            max_Q_x_d[i] = info['max_Q_x_d']

        delta = r.sum() + self.reward_shape + self.discount * \
            max_Q_y.sum() - max_Q_x_d.sum()
        self.reward_shape = 0
        # update Eligibility Trace
        for trace in self.Trace.values():
            trace *= self.discount_trace
        for i in range(r_len):
            self.Trace[x[i]][dest[i]] += \
                self._gradient(x[i], dest[i], y_idx[i])
            # update Q table
            old_Q_score = self.Qtable[x[i]][dest[i]][y_idx[i]]
            self.Qtable[x[i]][dest[i]][y_idx[i]] += lr['q'] * \
                (r[i] + self.discount*max_Q_y[i] - old_Q_score)
        # Update Theta
        for x, theta in self.Theta.items():
            theta += lr['p'] * delta * self.Trace[x]

    def __repr__(self):
        return "<MultiAgent discount:{} discount_trace:{}>".format(self.discount, self.discount_trace)

    def clean(self):
        for trace in self.Trace.values():
            trace.fill(0.0)
