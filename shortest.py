import numpy as np

from base_policy import Policy


class Shortest(Policy):
    """ Shortest agent determine the action based on the shortest path distance. """
    attrs = Policy.attrs | set(['distance', 'choice', 'mask'])

    def __init__(self, network):
        super().__init__(network)
        self.distance = np.full((len(self.links), len(self.links)), np.inf)
        self.mask = np.ones_like(self.distance, dtype=np.bool)
        self.choice = {n: np.zeros((len(self.links), len(v)), dtype=np.bool)
                       for n, v in self.links.items()}
        for x, neighbors in self.links.items():
            self.distance[x, x] = 0
            self.mask[x, x] = False
            for y in neighbors:
                self.distance[x, y] = 1
                self.choice[x][y][self.action_idx[x][y]] = True
                self.mask[x, y] = False
        self.unit = lambda x: 1  # regard unit distance as 1
        self._calc_distance()

    def choose(self, source, dest, random=False):
        """ Return the action with shortest distance and the distance """
        if random:
            return np.random.choice(self.links[source][self.choice[source][dest]])
        else:
            return self.links[source][self.choice[source][dest]][0]

    def _calc_distance(self):
        self.distance[self.mask] = np.inf
        changing = True
        while changing:
            changing = False
            for x in self.links.keys():
                if x == 19:
                    print(self.distance[x])
                for y in self.links[x]:
                    new_dis = self.distance[y] + self.unit(y)
                    greater = self.distance[x] > new_dis
                    if x == 19:
                        print(y, new_dis)
                    if greater.any():
                        self.distance[x][greater] = new_dis[greater]
                        self.choice[x][greater, :].fill(False)
                        self.choice[x][greater, self.action_idx[x][y]] = True
                        changing = True
                    else:
                        self.choice[x][self.distance[x] ==
                                       new_dis, self.action_idx[x][y]] = True


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
