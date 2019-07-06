import numpy as np
import pickle


class Policy:
    """
    `Policy` is the base class of all routing policies.

    Attributes:
        mode (string | None): identifier of what mode the policy needs Network running in.
        attrs (Set[string]): the attributes would be dumpped/loaded in `self.store`/`self.load`.
        links (Dict[Int, np.array(Int)]): represents the network graph
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
        pass

    def get_info(self, source, dest, action):
        return {}

    def learn(self, rewards, lr={}):
        pass

    def receive(self, source, dest):
        pass

    def send(self, source, dest):
        pass

    def store(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({k: self.__dict__[k] for k in self.attrs}, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            for k, v in pickle.load(f).items():
                self.__dict__[k] = v
