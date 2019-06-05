import numpy as np
import pickle


class Policy:
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

    def store(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({k: self.__dict__[k] for k in self.attrs}, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            for k, v in pickle.load(f).items():
                self.__dict__[k] = v
