import logging
import numpy as np

from env import *
from config import *
from shortest import Shortest
from qroute import Qroute
from hybrid import HybridQ
from multi_agent import MaHybridQ

Duration = 1
np.random.seed(1)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

File = "6x6.net"
nw   = Network(File)

# =============
# agent = Shortest(nw)
agent = Qroute(nw)
# agent = HybridQ(nw)
# agent = MaHybridQ(nw)
# =============

for node in nw.nodes.values():
    node.agent = agent

while nw.clock < Duration:
    reward = nw.step()
    agent.learn(reward)
