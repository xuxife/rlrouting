import pickle
from config import *


class Policy:
    attrs = ['links']

    def choose(self, source, dest):
        pass

    def get_reward(self, source, dest, action):
        return {}

    def learn(self, rewrads, lr={}):
        pass

    def store(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({k: self.__dict__[k] for k in self.attrs}, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            for k, v in pickle.load(f).items():
                self.__dict__[k] = v

    def drop_penalty(self, event, penalty=DropPenalty):
        pass
