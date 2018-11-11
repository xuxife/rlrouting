import math
import random
from collections import OrderedDict

from config import LearnRateQ

class Qroute:
    def __init__(self, network, initQ):
        self.Qtable = {}
        for source, neibors in network.links.items():
            self.Qtable[source] = {}
            for dest in network.links:
                if dest == source:
                    continue
                self.Qtable[source][dest] = OrderedDict({k: initQ for k in neibors})
    
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

    def get_reward(self, source, dest, action):
        agent_info = {}
        _, next_min_score = self.choose(action, dest)
        agent_info['next_min'] = next_min_score
        return agent_info

    def learn(self, reward, lr=LearnRateQ):
        q = reward.queue_time
        t = reward.trans_time
        source, dest, action = reward.source, reward.dest, reward.action
        # _, next_score = self.choose(reward.action, reward.dest)
        next_min_score = reward.agent_info['next_min']
        old_score = self.Qtable[source][dest][action]
        self.Qtable[source][dest][action] += lr*(q+t+next_min_score-old_score)