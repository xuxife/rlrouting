import numpy as np
import logging
from collections import OrderedDict

from config import *


class Packet:
    """ Packet is an abstract packet.

    Args:
        source, dest (int): The start/end nodes' ID.
        birth (int, time): When the packet is generated.

    Attritubes:
        event (Event): The current event of this packet.
        start_queue (int, time): When the queuing started.
        queue_time/trans_time (int, time): Queuing time/Transportation time.
        hops (int): The number of hop (one hop is a jump from node to node).
        route_time (int, time): The duration from birth to end.
    """

    def __init__(self, source, dest, birth):
        self.source = source
        self.dest = dest
        self.current = source
        self.birth = birth

        self.hops = 0
        self.start_queue = 0
        self.queue_time = 0
        self.trans_time = 0

    def __repr__(self):
        return "Packet<{}->{}>".format(self.source, self.dest)


class Event:
    """ Event records a packet passing through a connection.

    Args:
        packet      (Packet): Which packet is being delivering of this event.
        from_node   (int)   : Where the delivery begins.
        to_node     (int)   : Where the delivery ends, the destination.
        arrive_time (int)   : When the corresponding packet would arrive to_node.
    """

    def __init__(self, packet, from_node, to_node, arrive_time):
        self.packet = packet
        self.from_node = from_node
        self.to_node = to_node
        self.arrive_time = arrive_time

    def __repr__(self):
        return "Event<{}->{} at {}>".format(self.from_node, self.to_node, self.arrive_time)


class Reward:
    """ Reward definds the backwrad reward from environment (what Network.step returns)

    Attributes:
        source, dest (int): Where the packet from/destination
        action (int): Which neibor the last node chose
        queue/trans_time (int, time): The time cost on queue/transmission on the last node/connection
        agent_info (:obj:): Extra information from agent.get_reward
    """

    def __init__(self, source, packet, choice, agent_info):
        self.source = source
        self.dest = packet.dest
        self.action = choice
        self.queue_time = packet.queue_time
        self.trans_time = packet.trans_time
        self.agent_info = agent_info  # extra information defined by agents

    def __repr__(self):
        return "Reward<{}->{} by {}|queue: {}, trans: {}>".format(
            self.source, self.dest, self.action, self.queue_time, self.trans_time)


class Node:
    """ Node is the unit actor in a network.

    Attributes:
        queue (list): Where Packets waiting for being delivered,
        sent  (dict): An pesudo stage where Packets already sent but not arrive the next node.

        end_packets (list): All packets end in this node.
    """

    def __init__(self, ID, clock, network):
        self.ID = ID
        self.clock = clock
        self.queue = []
        self.sent = {}
        self.network = network

    def link(self, neibor):
        self.sent[neibor] = []

    def __repr__(self):
        return "Node<{}, queue: {}, sent: {}>".format(self.ID, self.queue, self.sent)

    def receive(self, packet):
        """ Receive a packet.
        Update statistic attritubes if this packet ends here, else append the packet into queue.
        """
        packet.current = self.ID
        logging.debug("{}: {} receives {}".format(self.clock, self.ID, packet))

        if packet.dest == self.ID:  # when the packet arrives its destination
            logging.debug("{}: {} ends in {}".format(
                self.clock, packet, self.ID))
            self.network.active_packets -= 1
            self.network.end_packets += 1
            self.network.route_time += self.clock - packet.birth
            self.network.hops += packet.hops
            del packet
        else:
            packet.start_queue = self.clock
            self.queue.append(packet)

    def send(self):
        """ Send a packet ordered by queue.
        Call agent.choose to determine the next node for the packet,
        then check whether the connection is avaliable.

        Returns:
            Reward: Reward of this action.
            None if no packet is sent.
        """
        i = 0
        while i < len(self.queue):
            choice = self.agent.choose(self.ID, self.queue[i].dest)
            if len(self.sent[choice]) <= BandwidthLimit:
                p = self.queue.pop(i)
                logging.debug("{}: {} sends {} to {}".format(
                    self.clock, self.ID, p, choice))
                p.hops += 1
                p.queue_time = self.clock - p.start_queue
                p.trans_time = TransTime  # set the transmission delay
                self.network.event_queue.append(
                    Event(p, self.ID, choice, self.clock+p.trans_time))
                self.sent[choice].append(p)
                return Reward(self.ID, p, choice, self.agent.get_reward(self.ID, p.dest, choice))
            else:
                i += 1
        return None


class Network:
    """ Network simulates packtes routing between connected nodes.

    Args:
        file (string): The name of network file.

    Attributes:
        clock (int, time): The simulation time.
        nodes (list): nodes
        links (list): lists of connected nodes' ID.

        event_queue (list): A queue of following happen events.
        rewards     (list): A list of rewards after call step function.

        active_packets (list): The packets are still active in the network.
        end_packets    (list): The packets already ends in its destination.
    """

    def __init__(self, file):
        self.project = {}  # project from file identity to node ID
        self.nodes = OrderedDict()
        self.links = OrderedDict()
        self.agent = None

        self.clean()

        self.read_network(file)

    def read_network(self, file):
        with open(file, 'r') as f:
            lines = [l.split() for l in f.readlines()]
        ID = 0
        for l in lines:
            if l[0] == "1000":
                self.project[l[1]] = ID
                self.nodes[ID] = Node(ID, self.clock, self)
                self.links[ID] = []
                ID += 1
            elif l[0] == "2000":
                source, dest = self.project[l[1]], self.project[l[2]]
                self.nodes[source].link(dest)
                self.nodes[dest].link(source)
                self.links[source].append(dest)
                self.links[dest].append(source)

    def bind(self, agent):
        """ bind the given agent to every node """
        self.agent = agent
        for node in self.nodes.values():
            node.agent = agent

    def train(self, duration, lambd=Lambda, slot=SlotTime, lrq=LearnRateQ, lrp=LearnRateP):
        """ duration (second) is the length of running period
            slot (second) is the length of one time slot
            lambd (second^(-1)) is the Poisson parameter
        """
        step_num = int(duration / slot)
        route_time, drop_rate = np.zeros(step_num), np.zeros(step_num)
        for i in range(step_num):
            r = self.step(slot, lambd=lambd*slot)
            if r is not None:
                self.agent.learn(r, lrq=lrq, lrp=lrp)
            route_time[i] = self.ave_route_time
            drop_rate[i] = self.drop_rate
        return route_time, drop_rate

    def clean(self):
        """ reset the network attritubes """
        self.clock = 0
        self.event_queue = []
        self.all_packets = 0
        self.end_packets = 0
        self.drop_packets = 0  # the number of packets dropped in the last step
        self.active_packets = 0
        self.total_drop_packet = 0
        self.hops = 0
        self.route_time = 0
        for node in self.nodes.values():
            node.clock = self.clock
            node.queue = []
            for neibor in node.sent:
                node.sent[neibor] = []

    def step(self, duration, lambd=Lambda):
        """ step runs the whole network forward.

        Args:
            duration (int, duration): The duration of one step.
            lambd    (int, float)   : The Poisson parameter (lambda) of this step.

        Returns:
            list: A list of rewards from sending events happended in the timeslot.
        """
        packets = self.new_packet(lambd)
        for p in packets:
            self.inject(p)

        rewards = []
        self.broadcast()
        for node in self.nodes.values():
            reward = node.send()
            if reward is not None:
                rewards.append(reward)

        i = 0
        while i < len(self.event_queue):
            if IsDrop and self.event_queue[i].packet.hops >= len(self.nodes):
                # drop the packet if too many hops
                e = self.event_queue.pop(i)
                self.nodes[e.from_node].sent[e.to_node].remove(e.packet)
                self.drop_packets += 1
                self.active_packets -= 1
                self.agent.drop_penalty(e)
            elif self.event_queue[i].arrive_time <= self.clock + duration:
                e = self.event_queue.pop(i)
                self.nodes[e.from_node].sent[e.to_node].remove(e.packet)
                self.nodes[e.to_node].clock = e.arrive_time
                self.nodes[e.to_node].receive(e.packet)
            else:
                i += 1

        self.clock += duration
        self.broadcast()
        return rewards

    def new_packet(self, lambd):
        """ Generates new packets following Poisson(lambd).

        Args:
            lambd (int, float): The Poisson distribution parameter.
        Returns:
            list: A list of new packets having random sources and destinations..
        """
        size = np.random.poisson(lambd)
        packets = []
        nodes_id = list(self.nodes.keys())
        for _ in range(size):
            source = np.random.choice(nodes_id)
            dest = np.random.choice(nodes_id)
            while dest == source:
                dest = np.random.choice(nodes_id)
            packets.append(Packet(source, dest, self.clock))
        return packets

    def inject(self, packet):
        """ Injects the packet into network """
        self.all_packets += 1
        self.active_packets += 1
        self.nodes[packet.source].receive(packet)

    def broadcast(self):
        """ Broadcast the network clock to nodes """
        for node in self.nodes.values():
            node.clock = self.clock

    @property
    def ave_hops(self):
        return self.hops / self.end_packets if self.end_packets > 0 else 0

    @property
    def ave_route_time(self):
        return self.route_time / self.end_packets if self.end_packets > 0 else 0

    @property
    def drop_rate(self):
        return self.drop_packets / self.all_packets if self.all_packets > 0 else 0


def print6x6(network):
    print("Time: {}".format(network.clock))
    for i in range(6):
        print("┌─────┐     "*6)
        for j in range(5):
            print("│No.{:2d}│".format(6*i+j), end="")
            if 6*i+j+1 in network.links[6*i+j]:
                print(" {:2d}├ ".format(
                    len(network.nodes[6*i+j].sent[6*i+j+1])), end="")
            else:
                print(" "*5, end="")
        print("│No.{:2d}│".format(6*i+5))
        for j in range(5):
            print("│{:5d}│".format(len(network.nodes[6*i+j].queue)), end="")
            if 6*i+j in network.links[6*i+j+1]:
                print(" ┤{:<2d} ".format(
                    len(network.nodes[6*i+j+1].sent[6*i+j])), end="")
            else:
                print(" "*5, end="")
        print("│{:5d}│".format(len(network.nodes[6*i+5].queue)))
        print("└─────┘     "*6)
        if i == 5:
            print("="*6)
            break
        for j in range(6):
            if 6*i+j in network.links[6*i+j+6]:
                print("{:2d}┴ ┬{:<2d}     ".format(len(
                    network.nodes[6*i+j+6].sent[6*i+j]), len(network.nodes[6*i+j].sent[6*i+j+6])), end="")
            else:
                print(" "*12, end="")
        print()
