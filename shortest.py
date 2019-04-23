import numpy as np
from heapq import *

from base_policy import *


class Shortest(Policy):
    """ Shortest agent determine the action based on the shortest path distance. """
    attrs = ['distance', 'choice']

    def __init__(self, network):
        self.links = network.links
        self.distance = np.empty((len(self.links), len(self.links)))
        self.prev = np.ones_like(self.distance, dtype=np.int) * -1
        self.choice = np.ones_like(self.distance, dtype=np.int) * -1
        self.learn([])

    def choose(self, source, dest):
        """ Return the action with shortest distance and the distance """
        return self.choice[source][dest]

    def learn(self, rewards, lr={}):
        self.distance.fill(np.inf)
        np.fill_diagonal(self.distance, 0)
        for i in range(len(self.links)):
            self._dijkstra(i)
        self._back_trace()

    def _dijkstra(self, source, cost=lambda x, y: 1):
        S = set()
        Q = [(0, source)]
        heapify(Q)
        while Q:
            dis, u = heappop(Q)
            if u in S:
                continue
            S.add(u)
            for v in self.links[u]:
                alt = dis + cost(u, v)
                if self.distance[source][v] > alt:
                    self.distance[source][v] = alt
                    self.prev[source][v] = u
                    heappush(Q, (alt, v))

    def _back_trace(self):
        for source in range(self.choice.shape[0]):
            for dest in range(self.choice.shape[1]):
                if source == dest:
                    continue
                by = dest
                while by != source:
                    last_by = by
                    by = self.prev[source][last_by]
                self.choice[source][dest] = last_by


class GlobalRoute(Shortest):
    def __init__(self, network):
        self.nodes = network.nodes
        self.queue_size = np.zeros(len(self.nodes), dtype=np.int)
        super().__init__(network)

    def learn(self, rewards, lr={}):
        for i, node in self.nodes.items():
            self.queue_size[i] = len(node.queue)
        self.distance.fill(np.inf)
        np.fill_diagonal(self.distance, 0)
        for i in range(len(self.links)):
            self._dijkstra(i, lambda x, y: 1+self.queue_size[y])
        self._back_trace()
