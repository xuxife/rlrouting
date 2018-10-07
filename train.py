import logging

from env import *
from config import *
from shortest import Shortest

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

File = "6x6.net"
Duration = 1
nw = Network(File)

# =============
agent = Shortest(nw.links)
# =============

for node in nw.nodes.values():
    node.agent = agent

while nw.clock < Duration:
    reward = nw.step()
    agent.learn(reward)
