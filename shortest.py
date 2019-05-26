import numpy as np
from heapq import *

from base_policy import *


class Shortest(Policy):
    """ Shortest agent determine the action based on the shortest path distance. """
    attrs = Policy.attrs + set(['distance', 'choice', 'mask'])

    def __init__(self, network):
        super().__init__(network)
        self.distance = np.full((len(self.links), len(self.links)), np.inf)
        self.choice = np.full_like(self.distance, -1, dtype=np.int)
        self.mask = np.ones_like(self.distance, dtype=np.bool)
        for node, neighbors in self.links.items():
            self.distance[node, node] = 0
            self.choice[node, node] = node
            self.mask[node, node] = False
            for neighbor in neighbors:
                self.distance[node, neighbor] = 1
                self.choice[node, neighbor] = neighbor
                self.mask[node, neighbor] = False
        self.unit = lambda x: 1  # regard unit distance as 1
        self._calc_distance()

    def choose(self, source, dest):
        """ Return the action with shortest distance and the distance """
        return self.choice[source][dest]

    def _calc_distance(self):
        self.distance[self.mask] = np.inf
        changing = True
        while changing:
            changing = False
            for source in self.links.keys():
                for dest in self.links.keys():
                    for neighbor in self.links[source]:
                        new_distance = self.distance[neighbor][dest] + \
                            self.unit(neighbor)
                        if self.distance[source][dest] > new_distance:
                            self.distance[source][dest] = new_distance
                            self.choice[source][dest] = neighbor
                            changing = True


class GlobalRoute(Shortest):
    def __init__(self, network):
        super().__init__(network)
        self.nodes = network.nodes
        self.queue_size = np.zeros(len(self.nodes), dtype=np.int)
        self.unit = lambda x: 1+self.queue_size[x]
        self._calc_distance()

    def learn(self, rewards, lr={}):
        for i, node in self.nodes.items():
            self.queue_size[i] = len(node.queue)
        self._calc_distance()
