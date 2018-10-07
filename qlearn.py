import math
import random

class Qlearn:
    Qtable = {}
    def __init__(self, links, initQ):
        for node, neibors in links.items():
            self.Qtable[node] = {}
            for dest_node in links.keys():
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

    def learn(self, reward, lr):
        q = reward.queue_time
        t = reward.trans_time
        next_score = reward.score
        old_score = self.Qtable[reward.source][reward.dest][reward.action]
        self.Qtable[reward.source][reward.dest][reward.action] += lr*(q+t+next_score-old_score)