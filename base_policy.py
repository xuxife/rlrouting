import numpy as np
import pickle


class Policy:
    """
    `Policy` is the base class of all routing agents.

    Attributes:
        mode (string | None): identifier of what mode the policy needs Network running in.
        attrs (Set[string]): the attributes would be dumpped/loaded in `self.store`/`self.load`.
        links (Dict[Int, np.array(Int)]): the network graph, the connections.
        action_idx (Dict[Int, Dict[Int, Int]]): store the indexes of node's neighbors in `links`
    """
    mode = None
    attrs = set(['links'])

    def __init__(self, network):
        self.links = {k: np.array(v, dtype=np.int)
                      for k, v in network.links.items()}
        self.action_idx = {node:
                           {a: i for i, a in enumerate(neighbors)}
                           for node, neighbors in self.links.items()}

    def choose(self, source, dest):
        " choose decides which path would the `source` agent choose to `dest` "
        pass

    def get_info(self, source, dest, action):
        " necessary information for training "
        return {}

    def learn(self, rewards, lr={}):
        " learn and update tables from rewards given "
        pass

    def receive(self, source, dest):
        " [optional] define what the agent should do when a packet is received by a node "
        pass

    def send(self, source, dest):
        " [optional] define what the agent should do when a packet is sent by a node "
        pass

    def store(self, filename):
        " dump `attrs` by pickle "
        with open(filename, 'wb') as f:
            pickle.dump({k: self.__dict__[k] for k in self.attrs}, f)

    def load(self, filename):
        " load `attrs` by pickle "
        with open(filename, 'rb') as f:
            for k, v in pickle.load(f).items():
                self.__dict__[k] = v
