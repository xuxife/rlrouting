"""
This is an Implement of Efficient Traffic Load-Balancing via Incremental Expansion of Routing Choices (https://doi.org/10.1145/3243173).

In this experiment, the environment setting differs from the paper.
Every node can only send (take action) ONE packet out every timeslot.

@email: 116010252@link.cuhk.edu.cn
"""
import numpy as np

from shortest import Shortest


class BP(Shortest):
    mode = 'bp'

    def __init__(self, network):
        super().__init__(network)
        self.reset()

    def reset(self):
        self.Q = {
            n: np.zeros(len(self.links), dtype=np.int)
            for n in self.links.keys()
        }

    def receive(self, source, dest):
        self.Q[source][dest] += 1

    def send(self, source, dest):
        self.Q[source][dest] -= 1

    def _is_phase2(self, source):
        """ 
        phase1 means only direacting packets to the shortest connections
        phase2 means direacting packets to all connections 
        """
        # default Phase 1
        return np.zeros(len(self.links), dtype=np.bool)

    def choose(self, source, links):
        destinations = []  # .__len__ == len(links)
        for neighbor in links:
            diff = self.Q[source] - self.Q[neighbor]
            dests = (diff > 0) & (
                self._is_phase2(source) |
                (self.choice[source][:, self.action_idx[source][neighbor]]))
            # (diff > 0) ==> (self.Q[source] > 0)
            diff[~dests] = 0
            if dests.any():
                destinations.append(diff.argmax())
            else:
                destinations.append(None)
        return destinations


class LBP(BP):
    def __init__(self, network, l_max=10):
        super().__init__(network)
        self.l_max = l_max

    def _is_phase2(self, source):
        return self.Q[source] > self.l_max
