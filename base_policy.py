import pickle


class Policy:
    attrs = set(['links'])

    def __init__(self, network):
        self.links = network.links

    def choose(self, source, dest):
        return

    def get_reward(self, source, dest, action):
        return {}

    def learn(self, rewards, lr={}):
        return

    def store(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({k: self.__dict__[k] for k in self.attrs}, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            for k, v in pickle.load(f).items():
                self.__dict__[k] = v
