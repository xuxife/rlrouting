import logging
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd


from env import *
from config import *
from shortest import Shortest
from qroute import Qroute
from hybrid import HybridQ
from multi_agent import MaHybridQ

np.random.seed(1)
# =============
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# =============

File = "6x6.net"
nw = Network(File)


def load_agent(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


# =============
# agent = Shortest(nw)
# agent = Qroute(nw)
agent = HybridQ(nw)
# agent = MaHybridQ(nw)
# =============
# agent.Qtable = load_agent('exp_data/qroute/{}.pkl'.format(l-0.25)).Qtable
# agent.Theta = load_agent('exp_data/hybrid/{}.pkl'.format(l-0.25)).Theta
# =============
nw.bind(agent)

# =============
# Train
# load = np.arange(0.5, 3.5, 0.25)
load = np.arange(0.25, 0.5, 0.25)
route_time = {}
drop_rate = {}
for l in load:
    # nw.bind(load_agent('exp_data/qroute/{}.pkl'.format(l)))
    #     nw.bind(load_agent('exp_data/hybrid/{}.pkl'.format(l)))
    #     nw.bind(load_agent('exp_data/hybrid/{}.pkl'.format(l-0.25)))
    nw.clean()
    route_time[l], drop_rate[l] = nw.train(10000, lambd=l, lrq=0.1, lrp=0.01)
# agent.store('exp_data/qroute/{}.pkl'.format(l))
#     agent.store('exp_data/hybrid/{}.pkl'.format(l))

df = pd.DataFrame(route_time)
df.plot()
drop = pd.DataFrame(drop_rate)
drop.plot()
plt.show()
# final = df.tail(1)
# final.T.plot()
# plt.plot(route_time[l])
# =============
# df.to_msgpack('exp_data/qroute0.25-3.75.msg')

# s = pd.read_msgpack('exp_data/shortest0.25-2.25.msg')
# n = pd.concat([final, s.tail(1)])
# plt.plot(route_time['3.5v3'])
