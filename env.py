import numpy as np
import logging
from collections import OrderedDict
from dataclasses import dataclass
from heapq import *

from base_policy import *
from config import *


@dataclass
class Packet:
    """ Packet is an abstract packet.

    Args:
        source, dest (int): The start/end nodes' ID.
        birth (int): When the packet is generated.

    Attritubes:
        trans_time (int): Transmission time.
        hops (int): The number of hops (one hop is a jump from node to node).
    """
    source: int
    dest: int
    birth: int
    hops: int = 0
    trans_time: int = 0

    def __repr__(self):
        return f"Packet<{self.source}->{self.dest}>"


@dataclass
class Event:
    """ Event records a packet passing through a connection.

    Args:
        packet      (Packet): Which packet is being delivering of this event.
        from_node   (int)   : Where the delivery begins.
        to_node     (int)   : Where the delivery ends, the destination.
        arrive_time (int)   : When the corresponding packet would arrive to_node.
    """
    packet: Packet
    from_node: int
    to_node: int
    arrive_time: int

    def __repr__(self):
        return f"Event<{self.from_node}->{self.to_node} at {self.arrive_time}>"

    def __lt__(self, other):
        return self.arrive_time < other.arrive_time


class Reward:
    """ Reward definds the backwrad reward from environment (what Network.step returns)

    Attributes:
        source, dest (int): Where the packet from/destination
        action (int): Which neighbor the last node chose
        agent_info (:obj:): Extra information from agent.get_reward
    """

    def __init__(self, source, packet, action, agent_info={}):
        self.source = source
        self.dest = packet.dest
        self.action = action
        self.packet = packet
        self.agent_info = agent_info  # extra information defined by agents

    def __repr__(self):
        return f"Reward<{self.source}->{self.dest} by {self.action}>"


class Clock:
    def __init__(self, now):
        self.t = now

    def __str__(self):
        return str(self.t)


class Node:
    """ Node is the unit actor in a network.

    Attributes:
        queue (List[Packet]): Where Packets waiting for being delivered,
        sent  (Dict[int, Packet]): An pesudo stage where Packets already sent but not arrive the next node.
    """

    def __init__(self, ID, clock, network):
        self.ID = ID
        self.clock = clock
        self.queue = []
        self.sent = {}
        self.network = network
        self.agent = network.agent

    def __repr__(self):
        return f"Node<{self.ID}, queue: {self.queue}, sent: {self.sent}>"

    def arrive(self, packet):
        logging.debug(f"{self.clock}: {packet} ends in {self.ID}")
        self.network.active_packets -= 1
        self.network.end_packets += 1
        self.network.route_time += self.clock.t - packet.birth
        self.network.hops += packet.hops
        del packet

    def receive(self, packet):
        """ Receive a packet. """
        logging.debug(f"{self.clock}: {self.ID} receives {packet}")
        if self.ID == packet.dest:
            self.arrive(packet)
        else:
            packet.start_queue = self.clock.t
            self.queue.append(packet)

    def send(self):
        """ Send a packet ordered by queue.
        Call agent.choose to determine the next node for the packet,
        then check whether the connection is avaliable.

        Returns:
            Reward: Reward of this action.
            None if no action is taken
        """
        i = 0
        while i < len(self.queue):
            dest = self.queue[i].dest
            action = self.agent.choose(self.ID, dest)
            if self.sent[action] <= BandwidthLimit:
                p = self.queue.pop(i)
                logging.debug(f"{self.clock}: {self.ID} sends {p} to {action}")
                p.hops += 1
                p.trans_time = TransTime  # set the transmission delay
                heappush(self.network.event_queue,
                         Event(p, self.ID, action, self.clock.t+p.trans_time))
                self.sent[action] += 1
                agent_info = self.agent.get_reward(self.ID, action, p)
                agent_info['q_y'] = max(
                    1, len(self.network.nodes[action].queue))
                if self.network.dual:
                    agent_info['q_x'] = max(1, len(self.queue))
                return Reward(self.ID, p, action, agent_info)
            else:
                i += 1
        return None


class Network:
    """ Network simulates packtes routing between connected nodes.

    Args:
        file (string): The name of network file.
        dual (bool): whether the network runs in DUAL mode. (for DRQ & CDRQ)

    Attributes:
        clock (Clock): The simulation time.
        nodes (Dict[Int, Node]): An ordered dictionary of all nodes in this network.
        links (Dict[Int, List[Int]]): lists of connected nodes' ID.
        agent (:obj:): bind an agent object who has methods `choose`, `learn`
        event_queue (List[Event]): A queue of following happen events.
        all_packets (int): The total number of packets in this simulation.
        end_packets (int): The packets already ends in its destination.
        drop_packets (int): The number of dropped packets
        active_packets (int): The number of active packets
        hops (int): The number of total hops of all packets
        route_time (int): The total routing time of all ended packets.
    """

    def __init__(self, file, dual=False, isdrop=False):
        self.project = {}  # project from file identity to node ID
        self.nodes = OrderedDict()
        self.links = OrderedDict()
        self.agent = Policy(self)
        self.dual = dual
        self.isdrop = isdrop

        self.clean()

        self.read_network(file)

    def clean(self):
        """ reset the network attritubes """
        self.clock = Clock(0)
        self.event_queue = []
        self.all_packets = 0
        self.end_packets = 0
        self.drop_packets = 0
        self.active_packets = 0
        self.hops = 0
        self.route_time = 0
        for node in self.nodes.values():
            node.clock = self.clock
            node.queue = []
            for neighbor in node.sent:
                node.sent[neighbor] = 0

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
                self.nodes[source].sent[dest] = 0
                self.nodes[dest].sent[source] = 0
                self.links[source].append(dest)
                self.links[dest].append(source)

    def bind(self, agent):
        """ bind the given agent to every node """
        self.agent = agent
        for node in self.nodes.values():
            node.agent = agent

    def train(self, duration, lambd=Lambda, slot=SlotTime, lr={}, penalty=DropPenalty, droprate=False):
        """ run `duration` steps in given lambda (poisson)

        Args:
            duration (second) : the length of running period
            lambd (second^(-1)) : the Poisson parameter
            slot (second) : the length of one time slot
            lr (Dict[str, float]): learning rate for Qtable or policy-table
            penalty (float): drop penalty
            droprate (bool): whether return droprate or not

        Returns:
            route_time (List[Real]): the vector of routing time in this training duration.
            drop_rate (List[Real]): the vector of packet-drop rate in this train.
        """
        step_num = int(duration / slot)
        route_time = np.zeros(step_num)
        if droprate:
            drop_rate = np.zeros(step_num)
        for i in range(step_num):
            r = self.step(slot, lambd=lambd*slot, penalty=penalty)
            if r is not None:
                if lr:
                    self.agent.learn(r, lr=lr)
                else:
                    self.agent.learn(r)
            route_time[i] = self.ave_route_time
            if droprate:
                drop_rate[i] = self.drop_rate
        return (route_time, drop_rate) if droprate else route_time

    def step(self, duration, lambd=Lambda, penalty=DropPenalty):
        """ step runs the whole network forward.

        Args:
            duration (int, duration): The duration of one step.
            lambd    (int, float)   : The Poisson parameter (lambda) of this step.

        Returns:
            List[Reward]: A list of rewards from sending events happended in the timeslot.
        """
        for p in self.new_packet(lambd):
            self.inject(p)

        rewards = []
        for node in self.nodes.values():
            reward = node.send()
            if reward is not None:
                rewards.append(reward)

        end_time = self.clock.t + duration
        closest_event = nsmallest(1, self.event_queue)
        while len(closest_event) > 0 and closest_event[0].arrive_time <= end_time:
            e = heappop(self.event_queue)
            closest_event = nsmallest(1, self.event_queue)
            self.nodes[e.from_node].sent[e.to_node] -= 1
            if self.isdrop and e.packet.hops >= len(self.hops):
                # drop the packet if too many hops
                self.drop_packets += 1
                self.active_packets -= 1
                self.agent.drop_penalty(e, penalty=penalty)
                continue
            self.clock.t = e.arrive_time
            self.nodes[e.to_node].receive(e.packet)

        self.clock.t = end_time
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
            packets.append(Packet(source, dest, self.clock.t))
        return packets

    def inject(self, packet):
        """ Injects the packet into network """
        self.all_packets += 1
        self.active_packets += 1
        self.nodes[packet.source].receive(packet)

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
                    network.nodes[6*i+j].sent[6*i+j+1]), end="")
            else:
                print(" "*5, end="")
        print("│No.{:2d}│".format(6*i+5))
        for j in range(5):
            print("│{:5d}│".format(len(network.nodes[6*i+j].queue)), end="")
            if 6*i+j in network.links[6*i+j+1]:
                print(" ┤{:<2d} ".format(
                    network.nodes[6*i+j+1].sent[6*i+j]), end="")
            else:
                print(" "*5, end="")
        print("│{:5d}│".format(len(network.nodes[6*i+5].queue)))
        print("└─────┘     "*6)
        if i == 5:
            print("="*6)
            break
        for j in range(6):
            if 6*i+j in network.links[6*i+j+6]:
                print("{:2d}┴ ┬{:<2d}     ".format(
                    network.nodes[6*i+j+6].sent[6*i+j], network.nodes[6*i+j].sent[6*i+j+6]), end="")
            else:
                print(" "*12, end="")
        print()
