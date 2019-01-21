import numpy as np
from hybrid import HybridQ
from config import *


class MaHybridQ(HybridQ):
    """ Multi-agent Hybrid Q routing with Eligibility Traces """

    def __init__(self, network, initQ=InitQ, initP=InitP, discount=Discount, discount_trace=DiscountTrace):
        super().__init__(network, initQ, initP)
        assert 0 <= discount <= 1, "discount factor should be in [0, 1]"
        assert 0 <= discount_trace <= 1, "discount factor of eligibility trace should be in [0, 1]"
        self.Trace = {}
        self.discount = discount
        self.discount_trace = discount_trace
        self.reward_shape = 0
        self.Trace = np.zeros(self.Qtable.shape)

    def learn(self, reward_list, lrq=LearnRateQ, lrp=LearnRateP):
        assert isinstance(
            reward_list, list), "input should be a list of rewards"
        num_rewards = len(reward_list)
        q, t = np.zeros(num_rewards), np.zeros(num_rewards)
        source, dest, action = np.zeros(num_rewards, dtype=np.int), np.zeros(
            num_rewards, dtype=np.int), np.zeros(num_rewards, dtype=np.int)
        action_max, source_max = np.zeros(num_rewards), np.zeros(num_rewards)
        for i, reward in enumerate(reward_list):
            q[i], t[i] = reward.queue_time, reward.trans_time
            source[i], dest[i], action[i] = reward.source, reward.dest, reward.action
            action_max[i] = reward.agent_info['action_max']
            source_max[i] = reward.agent_info['source_max']

        r = - q - t  # r_t = -(q+t) + r_shaping
        delta = r.sum() + self.reward_shape + self.discount * \
            action_max.sum() - source_max.sum()
        self.reward_shape = 0
        # update Eligibility Trace
        self.Trace *= self.discount_trace
        for i in range(num_rewards):
            self.Trace[source[i]][dest[i]][:self.neibor_num[source[i]]
                                           ] += self.gradient(source[i], dest[i], action[i])
            # update Q table
            action_index = self.links[source[i]].index(action[i])
            old_Q_score = self.Qtable[source[i]][dest[i]][action_index]
            self.Qtable[source[i]][dest[i]][action_index] += lrq * \
                (r[i] + self.discount*action_max[i] - old_Q_score)
        # Update Theta
        self.Theta += lrp * delta * self.Trace

    def drop_penalty(self, event, penalty=DropPenalty):
        self.reward_shape += penalty

    def __repr__(self):
        return "<MultiAgent discount:{} discount_trace:{}>".format(self.discount, self.discount_trace)

    def clean_trace(self):
        self.Trace = np.zeros(self.Qtable.shape)
