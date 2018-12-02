import numpy as np
from hybrid import HybridQ
from config import *

class MaHybridQ(HybridQ):
    """ Multi-agent Hybrid Q routing with Eligibility Traces """
    def __init__(self, network, initQ=InitQ, initP=InitP, discount=Discount, discount_trace=DiscountTrace):
        super().__init__(network, initQ, initP)
        assert 0 <= discount <= 1, "discount factor should be in [0, 1]"
        assert 0 <= discount_trace <= 1, "discount factor of eligibility trace should be in [0, 1]"
        self.Trace    = {}
        self.discount = discount
        self.discount_trace = discount_trace
        for source, neibors in network.links.items():
            self.Trace[source] = np.zeros(len(neibors))

    def learn(self, reward_list, lrq=LearnRateQ, lrp=LearnRateP):
        assert isinstance(reward_list, list), "input should be a list of rewards"
        num_rewards = len(reward_list)
        q, t = np.zeros(num_rewards), np.zeros(num_rewards)
        source, dest, action = np.zeros(num_rewards, dtype=np.int), np.zeros(num_rewards, dtype=np.int), np.zeros(num_rewards, dtype=np.int)
        action_max, source_max = np.zeros(num_rewards), np.zeros(num_rewards)
        for i, reward in enumerate(reward_list):
            q[i], t[i] = reward.queue_time, reward.trans_time
            source[i], dest[i], action[i] = reward.source, reward.dest, reward.action
            action_max[i] = reward.agent_info['action_max']
            source_max[i] = reward.agent_info['source_max']

        r = - q - t  # r_t = -(q+t) + r_shaping
        r_sum = sum(r)
        action_max_sum = sum(action_max)
        source_max_sum = sum(source_max)

        for i in range(num_rewards):
            # update Eligibility Trace
            self.Trace[source[i]] += self.discount_trace * self.gradient(source[i], dest[i], action[i])
            # update Theta
            self.Theta[source[i]][dest[i]] += lrp * self.Trace[source[i]] * (
                r_sum + self.discount*action_max_sum - source_max_sum)
            # update Q table
            old_Q_score = self.Qtable[source[i]][dest[i]][action[i]]
            self.Qtable[source[i]][dest[i]][action[i]] += lrq * (r[i] + self.discount*action_max[i] - old_Q_score)
            
