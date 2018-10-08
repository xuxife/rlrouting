import logging
import random

from env import *
from config import *
from shortest import Shortest
from qlearn import Qlearn
from hybrid import HybridQ

File = "6x6.net"
Duration = 1

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
random.seed()

nw = Network(File)

# =============
# agent = Shortest(nw)
agent = Qlearn(nw, 1)
# =============

for node in nw.nodes.values():
    node.agent = agent

while nw.clock < Duration:
    reward = nw.step()
    agent.learn(reward)
