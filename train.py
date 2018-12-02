import logging
import numpy as np
import pickle

from env import *
from config import *
from shortest import Shortest
from qroute import Qroute
from hybrid import HybridQ
from multi_agent import MaHybridQ

np.random.seed(1)
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

File = "6x6.net"
nw = Network(File)

# =============
# agent = Shortest(nw)
agent = Qroute(nw)
# agent = HybridQ(nw)
# agent = MaHybridQ(nw)
# =============
# agentFile = "0.2/qroute3w0.3.pkl"
agentFile = "1.5/qroute.pkl"
with open(agentFile, "rb") as f:
    agent = pickle.load(f)
# =============

for node in nw.nodes.values():
    node.agent = agent

route_time = []
for i in range(10000):
    r = nw.step(lambd=1.8)
    if r is not None:
        agent.learn(r, lr=0.3)
        # agent.learn(r, lrq=0.01, lrp=0.01)
    route_time.append(nw.ave_route_time)

import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame(route_time)
df.plot()
plt.show()
