import math
import random
from collections import OrderedDict

from config import LearnRate

class Qroute:
    def __init__(self, network, initQ):
        self.Qtable = Order
        for node, neibors in network.links.items():
            self.Qtable[node] = OrderedDict()
            for dest_node in network.links:
                if dest_node == node:
                    continue
                self.Qtable[node][dest_node] = {k: initQ for k in neibors}
    
    def choose(self, source, dest):
        min_score = math.inf
        min_neibors = []
        for neibor, score in self.Qtable[source][dest].items():
            if score < min_score:
                min_score = score
                min_neibors = [neibor]
            elif score == min_score:
                min_neibors.append(neibor)
        choice = random.choice(min_neibors)
        return choice, min_score

    def learn(self, reward, lr=LearnRate):
        q = reward.queue_time
        t = reward.trans_time
        next_score = reward.score
        old_score = self.Qtable[reward.source][reward.dest][reward.action]
        self.Qtable[reward.source][reward.dest][reward.action] += lr*(q+t+next_score-old_score)