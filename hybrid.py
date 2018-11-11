import math
import random
from collections import OrderedDict

from config import LearnRateQ, LearnRateP

class HybridQ:
    def __init__(self, network, initQ, initP):
        self.Qtable, self.Theta = {}, {}
        for node, neibors in network.links.items():
            self.Qtable[node] = {}
            self.Theta[node] = {}
            for dest_node in network.links:
                if dest_node == node:
                    continue
                self.Qtable[node][dest_node] = OrderedDict({k: initQ for k in neibors})
                self.Theta[node][dest_node] = OrderedDict({k: initP for k in neibors})
    
    def choose(self, source, dest):
        """ choose returns the choice following weighted random sample, and the Q score of the choice
        """
        population, weight = zip(*[(k, v) for k, v in self.exp_theta(source, dest).items()])
        choice = random.choices(population, weight)
        return choice, self.Qtable[source][dest][choice]

    def exp_theta(self, source, dest):
        return OrderedDict({k: math.exp(v) for k, v in self.Theta[source][dest].items()})
    
    def min_q_score(self, source, dest):
        return min(self.Qtable[source][dest].values())

    def get_reward(self, source, dest, action):
        agent_info = {}
        agent_info['action_min'] = self.min_q_score(action, dest)
        agent_info['source_min'] = self.min_q_score(source, dest)
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
    