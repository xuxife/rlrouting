import numpy as np

from base_policy import Policy


class Shortest(Policy):
    """ Shortest agent determine the action based on Dijkstra algorithm. 

    Attributes:
        distance (np.array(float64, (nodes, nodes))): stores the distance between nodes
        mask (np.array(bool, (nodes, nodes))): indicates `non-neighbors`, who are assumed infinity distance before dijkstra
        choice (Dict[Int, np.array(bool, (nodes, neighbors))]): choice[a][b, :] indicates Node a to Node b which (multiple) neighbor(s) can be chosen
    """
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
                self.choice[x][y, self.action_idx[x][y]] = True
                self.mask[x, y] = False
        self.unit = lambda x: 1  # regard unit distance as 1
        self._calc_distance()

    def choose(self, source, dest, random=False):
        """ Return the action with shortest distance and the distance """
        choices = self.links[source][self.choice[source][dest]]
        return np.random.choice(choices) if random else choices[0]

    def _calc_distance(self):
        self.distance[self.mask] = np.inf
        changing = True
        while changing:
            changing = False
            for x in self.links.keys():
                for y in self.links[x]:
                    new_dis = self.distance[y] + self.unit(y)
                    greater = self.distance[x] > new_dis
                    if greater.any():
                        self.distance[x][greater] = new_dis[greater]
                        self.choice[x][greater, :].fill(False)
                        self.choice[x][greater, self.action_idx[x][y]] = True
                        changing = True
                    equal = (~np.isinf(self.distance[x])) & (
                        self.distance[x] == new_dis)
                    if equal[self.mask[x]].any():
                        self.choice[x][equal, self.action_idx[x][y]] = True


class GlobalRoute(Shortest):
    def __init__(self, network):
        super().__init__(network)
        self.nodes = network.nodes
        self.queue_size = np.zeros(len(self.nodes), dtype=np.int)
        self.unit = lambda x: 1+self.queue_size[x]
        self._calc_distance()

    def receive(self, source, dest):
        self.queue_size[source] += 1

    def send(self, source, dest):
        self.queue_size[source] -= 1

    def learn(self, rewards, lr={}):
        for i, node in self.nodes.items():
            self.queue_size[i] = len(node.queue)
        self._calc_distance()
