import math
import random
from collections import OrderedDict

from config import LearnRate

class HybridQ:
    def __init__(self, network, initQ, initP):
        self.Qtable, self.Ptable = OrderedDict(), OrderedDict()
        for node, neibors in network.links.items():
            self.Qtable[node] = {}
            self.Ptable[node] = {}
            for dest_node in network.links:
                if dest_node == node:
                    continue
                self.Qtable[node][dest_node] = {k: initQ for k in neibors}
                self.Ptable[node][dest_node] = {k: initP for k in neibors}
    
    def choose(self, source, dest):
        """ choose returns the choice following weighted random sample, and the Q score of the choice
        """
        population, weight = zip(*[(k, v) for k, v in self.exp_p_score.items()])
        choice = random.choices(population, weight)
        return choice, self.Qtable[choice]

    @property
    def exp_p_score(self, source, dest):
        return OrderedDict({k: math.exp(v) for k, v in self.Ptable[source][dest].items()})
    
    def learn(self, reward, lrq=LearnRate, lrp=LearnRate):
        q, t = reward.queue_time, reward.trans_time
        source, dest = reward.source, reward.dest
        next_score = reward.score
        old_Q_score = self.Qtable[source][dest]
        self.Qtable[source][dest] += lrq * (q + t + next_score - old_Q_score)
        _, min_score = self.choose(source, dest)
        for neibor in self.Ptable[source][dest]:
            self.Ptable[source][dest][neibor] -= lrp * (q + t + next_score - min_score) * self._gradient(source, dest, reward.action, neibor)

    def _gradient(self, source, dest, action, theta):
        exp_p_score = self.exp_p_score
        exp_sum = sum(exp_p_score.values())
        if theta == action:
            return 1 - exp_p_score[theta]/exp_sum
        return - exp_p_score[theta]/exp_sum
    