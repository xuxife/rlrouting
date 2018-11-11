import math
import random
from collections import OrderedDict

from config import *

class Qroute:
    def __init__(self, network, initQ=InitQ):
        self.Qtable = {}
        for source, neibors in network.links.items():
            self.Qtable[source] = {}
            for dest in network.links:
                if dest == source:
                    continue
                self.Qtable[source][dest] = OrderedDict({k: initQ for k in neibors})
    
    def choose(self, source, dest):
        max_score   = math.inf
        max_neibors = []
        for neibor, score in self.Qtable[source][dest].items():
            if score < max_score:
                max_score   = score
                max_neibors = [neibor]
            elif score == max_score:
                max_neibors.append(neibor)
        choice = random.choice(max_neibors)
        return choice

    def get_reward(self, source, dest, action):
        agent_info = {}
        agent_info['action_max'] = max(self.Qtable[source][dest].values())
        return agent_info

    def learn(self, reward, lr=LearnRateQ):
        q, t = reward.queue_time, reward.trans_time
        source, dest, action = reward.source, reward.dest, reward.action
        action_max = reward.agent_info['action_max']
        old_score = self.Qtable[source][dest][action]
        self.Qtable[source][dest][action] += lr*(-q-t + action_max - old_score)