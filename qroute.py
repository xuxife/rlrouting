import math
import pickle
import numpy as np
from scipy import sparse

from config import *


class Qroute:
    """ Qroute use Q routing as policy. 

    Attritubes:
        Qtable : Stores the Q scores of all nodes.
    """

    def __init__(self, network, initQ=InitQ):
        self.links = network.links
        node_num = len(self.links)
        self.neibor_num = np.array([len(self.links[i])
                                    for i in range(len(self.links))])
        self.Qtable = InitQ * \
            np.ones((node_num, node_num, self.neibor_num.max()))

    def choose(self, source, dest):
        scores = self.Qtable[source][dest][:self.neibor_num[source]]
        choice = np.random.choice(np.argwhere(
            scores == scores.max()).flatten())
        return self.links[source][choice]

    def get_reward(self, source, dest, action):
        return {'action_max': self.Qtable[action][dest][:self.neibor_num[action]].max()}

    def learn(self, reward_list, lrq=LearnRateQ, lrp=0):
        for reward in reward_list:
            q, t = reward.queue_time, reward.trans_time
            source, dest, action = reward.source, reward.dest, reward.action
            action_max = reward.agent_info['action_max']
            action_index = self.links[source].index(action)
            old_score = self.Qtable[source][dest][action_index]
            self.Qtable[source][dest][action_index] += lrq * \
                (-q-t + action_max - old_score)

    def store(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def drop_penalty(self, event):
        pass
