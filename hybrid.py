import math
import random
from collections import OrderedDict

from config import LearnRateQ, LearnRateP

class HybridQ:
    def __init__(self, network, initQ, initP):
        self.Qtable, self.Theta = {}, {}
        for source, neibors in network.links.items():
            self.Qtable[source] = {}
            self.Theta [source] = {}
            for dest in network.links:
                if dest == source:
                    continue
                self.Qtable[source][dest] = OrderedDict({k: initQ for k in neibors})
                self.Theta [source][dest] = OrderedDict({k: initP for k in neibors})
    
    def choose(self, source, dest):
        """ choose returns the choice following weighted random sample, and the Q score of the choice
        """
        population, weight = zip(*[(k, v) for k, v in self.exp_theta(source, dest).items()])
        choice = random.choices(population, weight)
        return choice, self.Qtable[source][dest][choice]

    def exp_theta(self, source, dest):
        return OrderedDict({k: math.exp(v) for k, v in self.Theta[source][dest].items()})
    
    def get_reward(self, source, dest, action):
        agent_info = {}
        agent_info['action_min'] = min(self.Qtable[action][dest].values())
        agent_info['source_min'] = min(self.Qtable[source][dest].values())
        return agent_info

    def learn(self, reward, lrq=LearnRateQ, lrp=LearnRateP):
        q, t = reward.queue_time, reward.trans_time
        source, dest, action = reward.source, reward.dest, reward.action
        action_min = reward.agent_info['action_min']
        source_min = reward.agent_info['source_min']
        old_Q_score = self.Qtable[source][dest][action]
        self.Qtable[source][dest][action] += lrq * (q + t + action_min - old_Q_score)
        for neibor in self.Theta[source][dest]:
            self.Ptable[source][dest][neibor] -= lrp * (q + t + action_min - source_min) * self.gradient(source, dest, action, neibor)

    def gradient(self, source, dest, action, theta):
        exp_theta = self.exp_theta(souce, dest)
        exp_sum = sum(exp_theta.values())
        if theta == action:
            return 1 - exp_theta[theta]/exp_sum
        return - exp_theta[theta]/exp_sum
    
