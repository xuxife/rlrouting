import numpy as np

from hybrid import HybridQ


class MaHybridQ(HybridQ):
    """ Multi-agent Hybrid Q routing with Eligibility Traces """
    attrs = HybridQ.attrs | set(['Trace'])

    def __init__(self, network, initQ=0, initP=0, discount=0.99, discount_trace=0.6):
        super().__init__(network, initQ=initQ, initP=initP, discount=discount)
        self.discount_trace = discount_trace
        self.reward_shape = 0
        self.Trace = {k: np.zeros_like(v, dtype=np.float64)
                      for k, v in self.Theta.items()}

    def learn(self, rewards, lr={'q': 0.1, 'p': 0.1}):
        num_rewards = len(rewards)
        q, t = np.zeros(num_rewards), np.zeros(num_rewards)
        x, dest, y_idx = np.zeros(num_rewards, dtype=np.int), np.zeros(
            num_rewards, dtype=np.int), np.zeros(num_rewards, dtype=np.int)
        max_Q_y, max_Q_x_d = np.zeros(num_rewards), np.zeros(num_rewards)
        for i, reward in enumerate(rewards):
            x[i], dest[i] = reward.source, reward.dest
            y_idx[i] = self.action_idx[x[i]][reward.action]
            info = reward.agent_info
            q[i], t[i] = info['q_y'], info['t_y']
            max_Q_y[i] = info['max_Q_y']
            max_Q_x_d[i] = info['max_Q_x_d']

        r = - q - t
        delta = r.sum() + self.reward_shape + self.discount * \
            max_Q_y.sum() - max_Q_x_d.sum()
        self.reward_shape = 0
        # update Eligibility Trac
        for trace in self.Trace.values():
            trace *= self.discount_trace
        for i in range(num_rewards):
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

    def clean_trace(self):
        for trace in self.Trace.values():
            trace.fill(0.0)
