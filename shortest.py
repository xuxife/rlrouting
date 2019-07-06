import numpy as np

from base_policy import Policy


class Shortest(Policy):
    """ Shortest agent determine the action based on Dijkstra algorithm. 

    Parameters:
        multiway(bool): whether store ALL possible shortest paths or not.
        random(bool): only affect when `multiway=True`, choose the shortest path randomly from ALL shortest paths

    Attributes:
        distance (np.array(float64, (nodes, nodes))): stores the distance between nodes
        mask (np.array(bool, (nodes, nodes))): indicates `non-neighbors`, who are assumed infinity distance before dijkstra
        choice (Dict[Int, np.array(bool, (nodes, neighbors))]): choice[a][b, :] indicates Node a to Node b which (multiple) neighbor(s) can be chosen
    """
    attrs = Policy.attrs | set(['distance', 'choice', 'mask'])

    def __init__(self, network, multiway=False, random=False):
        super().__init__(network)
        self.multiway = multiway
        self.random = random
        self.distance = np.full((len(self.links), len(self.links)), np.inf)
        self.mask = np.ones_like(self.distance, dtype=np.bool)
        self.choice = {n: np.zeros((len(self.links), len(v)), dtype=np.bool)
                       for n, v in self.links.items()}
        for x, neighbors in self.links.items():
            self.distance[x, x] = 0
            self.mask[x, x] = False
            for y in neighbors:
                self.distance[x, y] = 1
                self.choice[x][y, self.action_idx[x][y]] = True
                self.mask[x, y] = False
        self.unit = lambda x: 1  # regard unit distance as 1
        self._calc_distance()

    def choose(self, source, dest):
        """ Return the action with shortest distance and the distance """
        choices = self.links[source][self.choice[source][dest]]
        return np.random.choice(choices) if self.random else choices[0]

    def _calc_distance(self):
        self.distance[self.mask] = np.inf
        changing = True
        while changing:
            changing = False
            for x, neighbors in self.links.items():
                for y in neighbors:
                    for z in self.links.keys():
                        new_dis = self.distance[y, z] + self.unit(y)
                        if self.distance[x, z] > new_dis:
                            self.distance[x, z] = new_dis
                            self.choice[x][z, :].fill(False)
                            self.choice[x][z, self.action_idx[x][y]] = True
                            changing = True
                        if self.multiway and np.isfinite(new_dis) and self.distance[x, z] == new_dis:
                            self.choice[x][z, self.action_idx[x][y]] = True

    def _calc_distance2(self):
        """ I don't know why this function did not work. 
        It SHOULD be same as (even faster than) the previous one.
        """
        self.distance[self.mask] = np.inf
        changing = True
        while changing:
            changing = False
            for x, neighbors in self.links.items():
                for y in neighbors:
                    new_dis = self.distance[y] + self.unit(y)
                    greater = self.distance[x] > new_dis
                    if greater.any():
                        self.distance[x][greater] = new_dis[greater]
                        self.choice[x][greater, :].fill(False)
                        self.choice[x][greater, self.action_idx[x][y]] = True
                        changing = True
                    if self.multiway:
                        equal = (np.isfinite(new_dis)) & (
                            self.distance[x] == new_dis)
                        if equal.any():
                            self.choice[x][equal, self.action_idx[x][y]] = True


class GlobalRoute(Shortest):
    def __init__(self, network, multiway=False, random=False):
        super().__init__(network, multiway=multiway, random=random)
        self.mask.fill(True)
        np.fill_diagonal(self.mask, False)
        self.queue_size = np.zeros(len(self.links), dtype=np.int)
        self.unit = lambda x: 1+self.queue_size[x]
        self._calc_distance()

    def receive(self, source, dest):
        self.queue_size[source] += 1

    def send(self, source, dest):
        self.queue_size[source] -= 1

    def learn(self, rewards, lr={}):
        # for i, node in self.nodes.items():
        #     self.queue_size[i] = len(node.queue)
        for choice in self.choice.values():
            choice.fill(False)
        self._calc_distance()
