class Shortest:
    """ Shortest agent determine the action based on the shortest path distance. """
    def __init__(self, network):
        self.distance = {}
        self.choice = {}
        for node, neibors in network.links.items():
            self.distance[node] = {}
            self.choice[node] = {}
            self.distance[node][node] = 0
            for neibor in neibors:
                self.distance[node][neibor] = 1
                self.choice[node][neibor] = neibor
            for dest_node in network.links.keys():
                if dest_node != node and dest_node not in neibors:
                    self.distance[node][dest_node] = len(network.links) + 1 # large enough
        changing = True
        while changing:
            changing = False
            for source in network.links.keys():
                for dest in network.links.keys():
                    for neibor in network.links[source]:
                        if self.distance[source][dest] > self.distance[neibor][dest] + 1:
                            self.distance[source][dest] = self.distance[neibor][dest] + 1
                            self.choice[source][dest] = neibor
                            changing = True

    def choose(self, source, dest):
        """ Return the action with shortest distance and the distance """
        return self.choice[source][dest]
    
    def learn(self, reward):
        return

    def get_reward(self, source, dest, action):
        return