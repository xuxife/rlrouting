import numpy as np
from env import *


class Shortest:
    """ Shortest agent determine the action based on the shortest path distance. """

    def __init__(self, network):
        self.nw = network
        node_num = len(self.nw.nodes)
        self.distance = np.full((node_num, node_num), np.inf)
        np.fill_diagonal(self.distance, 0)
        self.choice = np.zeros((node_num, node_num), dtype=np.int)
        for node, neibors in network.links.items():
            for neibor in neibors:
                self.distance[node][neibor] = 1
                self.choice[node][neibor] = neibor
        self.dijkstra()

    def dijkstra(self):
        changing = True
        while changing:
            changing = False
            for source, neibors in self.nw.links.items():
                for dest in self.nw.nodes:
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

    def drop_penalty(self, event, penalty=DropPenalty):
        pass


class GlobalRoute(Shortest):
    def dijkstra(self):
        changing = True
        while changing:
            changing = False
            for source, neibors in self.nw.links.items():
                for dest in self.nw.nodes:
                    for neibor in neibors:
                        neibor_dis = self.distance[neibor][dest] + \
                            1 + len(self.nw.nodes[neibor].queue)
                        if self.distance[source][dest] > neibor_dis:
                            self.distance[source][dest] = neibor_dis
                            self.choice[source][dest] = neibor
                            changing = True

    def learn(self, reward, lrq=0, lrp=0):
        self.dijkstra()
