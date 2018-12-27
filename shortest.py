import numpy as np


class Shortest:
    """ Shortest agent determine the action based on the shortest path distance. """

    def __init__(self, network):
        node_num = len(network.nodes)
        self.distance = np.zeros((node_num, node_num), dtype=np.int)
        self.choice = np.zeros((node_num, node_num), dtype=np.int)
        for node, neibors in network.links.items():
            for neibor in neibors:
                self.distance[node][neibor] = 1
                self.choice[node][neibor] = neibor
            for dest_node in network.links.keys():
                if dest_node != node and dest_node not in neibors:
                    self.distance[node][dest_node] = len(
                        network.links) + 1  # large enough
        changing = True
        while changing:
            changing = False
            for source, neibors in network.links.items():
                for dest in network.nodes:
                    for neibor in neibors:
                        if self.distance[source][dest] > self.distance[neibor][dest] + 1:
                            self.distance[source][dest] = self.distance[neibor][dest] + 1
                            self.choice[source][dest] = neibor
                            changing = True

    def choose(self, source, dest):
        """ Return the action with shortest distance and the distance """
        return self.choice[source][dest]

    def learn(self, reward, lrq=0, lrp=0):
        pass

    def get_reward(self, source, dest, action):
        pass

    def drop_penalty(self, event):
        pass
