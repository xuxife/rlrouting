import math
import pickle
import numpy as np

from config import *


class HybridQ:
    def __init__(self, network, initQ=InitQ, initP=InitP, discount=Discount):
        self.discount = discount
        self.nw, self.links = network, network.links
        node_num = len(self.links)
        self.neibor_num = np.array([len(self.links[i])
                                    for i in self.links.keys()])
        self.Qtable = InitQ * \
            np.ones((node_num, node_num, self.neibor_num.max()))
        self.Theta = InitP * \
            np.ones((node_num, node_num, self.neibor_num.max()))

    def choose(self, source, dest):
        """ choose returns the choice following weighted random sample """
        theta = self.Theta[source][dest][:self.neibor_num[source]]
        e_theta = np.exp(theta)
        choice = np.random.choice(
            self.links[source], p=e_theta/e_theta.sum(axis=0))
        return choice

    def get_reward(self, source, dest, action):
        return {
            'action_max': self.Qtable[action][dest][:self.neibor_num[action]].max(),
            'source_max': self.Qtable[source][dest][:self.neibor_num[source]].max(),
        }

    def learn(self, reward_list, lrq=LearnRateQ, lrp=LearnRateP):
        for reward in reward_list:
            q, t = reward.queue_time, reward.trans_time
            source, dest, action = reward.source, reward.dest, reward.action
            action_max = reward.agent_info['action_max']
            source_max = reward.agent_info['source_max']
            action_index = self.links[source].index(action)
            old_Q_score = self.Qtable[source][dest][action_index]
            self.Qtable[source][dest][action_index] += lrq * \
                (-q-t + self.discount*action_max - old_Q_score)
            delta = lrp *\
                (-q-t + self.discount*action_max - source_max) * \
                self.gradient(source, dest, action)
            self.Theta[source][dest][:self.neibor_num[source]] += delta
        np.clip(self.Theta, -2, 2)

    def gradient(self, source, dest, action):
        """ gradient returns a vector with length of neibors of source """
        theta = self.Theta[source][dest][:self.neibor_num[source]]
        e_theta = np.exp(theta)
        gradient = - e_theta/e_theta.sum(axis=0)
        for i, n in enumerate(self.links[source]):
            if n == action:
                gradient[i] += 1
        return gradient

    def store(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def drop_penalty(self, event, penalty=DropPenalty):
        pass

    def __repr__(self):
        return "<HybridQ discount:{}>".format(self.discount)
