class Shortest:
    """ Shortest agent determine the action based on the shortest path distance
        if multiple actions having the same distance, shortest would choose any one randomly
    """
    def __init__(self, links):
        self.distance = {}
        self.choice = {}
        for node, neibors in links.items():
            self.distance[node] = {}
            self.choice[node] = {}
            self.distance[node][node] = 0
            for neibor in neibors:
                self.distance[node][neibor] = 1
                self.choice[node][neibor] = neibor
            for dest_node in links.keys():
                if dest_node != node and dest_node not in neibors:
                    self.distance[node][dest_node] = len(links) + 1 # large enough
        changing = True
        while changing:
            changing = False
            for source in links.keys():
                for dest in links.keys():
                    for neibor in links[source]:
                        if self.distance[source][dest] > self.distance[neibor][dest] + 1:
                            self.distance[source][dest] = self.distance[neibor][dest] + 1
                            self.choice[source][dest] = neibor
                            changing = True

    def choose(self, source, dest):
        """ return the action with shortest distance and the distance
        """
        return self.choice[source][dest], self.distance[source][dest]
    
    def learn(self, reward):
        return