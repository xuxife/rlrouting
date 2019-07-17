import numpy as np
import logging
from collections import OrderedDict
from dataclasses import dataclass
from heapq import *

from base_policy import Policy


@dataclass
class Packet:
    """ Packet is an abstract packet.

    Args:
        source, dest (int): The start/end nodes' ID.
        birth (int): When the packet is generated.

    Attributes:
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
    """ Reward defines the backward reward from environment (what Network.step returns)

    Attributes:
        source, dest (int): Where the packet from/destination
        action (int): Which neighbor the last node chose
        agent_info (:obj:): Extra information from agent.get_info
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
        sent  (Dict[int, Packet]): An pseudo stage where Packets already sent but not arrive the next node.
    """

    def __init__(self, ID, clock, network):
        self.ID = ID
        self.clock = clock
        self.queue = []
        self.sent = {}
        self.network = network

    @property
    def agent(self):
        return self.network.agent

    @property
    def mode(self):
        return self.network.mode

    def __repr__(self):
        return f"Node<{self.ID}, queue: {self.queue}, sent: {self.sent}>"

    def arrive(self, packet):
        """ The packet ends at this Node """
        logging.debug(f"{self.clock}: {packet} ends in {self.ID}")
        self.network.active_packets -= 1
        self.network.end_packets += 1
        self.network.route_time += self.clock.t - packet.birth
        self.network.hops += packet.hops
        del packet

    def receive(self, packet):
        """ Receive a packet """
        logging.debug(f"{self.clock}: {self.ID} receives {packet}")
        if self.ID == packet.dest:
            self.arrive(packet)
        else:
            packet.start_queue = self.clock.t
            self.queue.append(packet)
            self.agent.receive(self.ID, packet.dest)

    def send(self):
        """ Send a packet ordered by queue.
        Call agent.choose to determine the next node for the packet,
        then check whether the connection is avaliable.

        Returns:
            Reward: Reward of this action.
            None if no action is taken
        """
        if self.mode == 'bp' and len(self.queue) > 0:
            action, dest = self.agent.choose(self.ID, filter(
                lambda n: self.sent[n] < self.network.bandwidth, self.sent.keys()))
            if action is None:
                return None
            p = next(p for p in self.queue if p.dest == dest)
            self.queue.remove(p)
            self._send_packet(p, action)
            self.agent.send(self.ID, dest)
            return Reward(self.ID, p, action, {})

        i = 0
        while i < len(self.queue):
            dest = self.queue[i].dest
            action = self.agent.choose(self.ID, dest)
            if self.sent[action] < self.network.bandwidth:
                p = self.queue.pop(i)
                self._send_packet(p, action)
                self.agent.send(self.ID, dest)
                # then build Reward
                agent_info = self.agent.get_info(self.ID, action, p)
                agent_info['q_y'] = max(
                    1, len(self.network.nodes[action].queue))
                # agent_info['q_y'] = self.clock.t - p.start_queue
                agent_info['t_y'] = 0
                if self.mode == 'dual':
                    agent_info['q_x'] = max(1, len(self.queue))
                    agent_info['t_x'] = 0
                return Reward(self.ID, p, action, agent_info)
            else:
                i += 1
        return None

    def _send_packet(self, p, action):
        logging.debug(f"{self.clock}: {self.ID} sends {p} to {action}")
        p.hops += 1
        p.trans_time = self.network.transtime  # set the transmission delay
        heappush(self.network.event_queue,
                 Event(p, self.ID, action, self.clock.t+p.trans_time))
        self.sent[action] += 1


class Network:
    """ Network simulates packet routing between connected nodes.

    Args:
        file (string): The name of network file.
        bandwidth (int): the bandwidth limitation of a connection
        transtime (int, float): the time cost of transmitting a packet to next node
        is_drop (bool): whether the network drop packets on some conditions
        read_func (function): a specific function reads a network file into self.nodes & self.links

    Attributes:
        clock (Clock): The simulation time.
        nodes (Dict[Int, Node]): An ordered dictionary of all nodes in this network.
        links (Dict[Int, List[Int]]): lists of connected nodes' ID.
        agent (:obj:): bind an agent object who has methods `choose`, `learn`
        mode (string): Network mode, 
            None -> Standard mode, 'dual' -> Duality mode, 'bp' -> BackPressure mode
        event_queue (List[Event]): A queue of following happen events.
        all_packets (int): The total number of packets in this simulation.
        end_packets (int): The packets already ends in its destination.
        drop_packets (int): The number of dropped packets
        active_packets (int): The number of active packets
        hops (int): The number of total hops of all packets
        route_time (int): The total routing time of all ended packets.
    """

    def __init__(self, file, bandwidth=3, transtime=1, is_drop=False, read_func=None):
        self.projection = {}  # project from file identity to node ID
        self.bandwidth = bandwidth
        self.transtime = transtime
        self.nodes = OrderedDict()
        self.links = OrderedDict()
        self.agent = Policy(self)
        self.is_drop = is_drop

        self.clean()

        if read_func is None:
            self.read_network(file)
        else:
            read_func(self, file)

    @property
    def mode(self):
        return self.agent.mode

    def clean(self):
        """ reset the network attributes """
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
        if self.mode == 'bp':
            self.agent.reset()

    def read_network(self, file):
        with open(file, 'r') as f:
            lines = [l.split() for l in f.readlines()]
        ID = 0
        for l in lines:
            if l[0] == "1000":
                self.projection[l[1]] = ID
                self.nodes[ID] = Node(ID, self.clock, self)
                self.links[ID] = []
                ID += 1
            elif l[0] == "2000":
                source, dest = self.projection[l[1]], self.projection[l[2]]
                self.nodes[source].sent[dest] = 0
                self.nodes[dest].sent[source] = 0
                self.links[source].append(dest)
                self.links[dest].append(source)

    def train(self, duration, lambd, slot=1, lr={}, penalty=0, droprate=False):
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

    def step(self, duration, lambd, penalty=0):
        """ step runs the whole network forward.

        Args:
            duration (int, duration): The duration of one step.
            lambd    (int, float)   : The Poisson parameter (lambda) of this step.
            penalty  (int, float)   : The penalty of dropping a packet

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
            if self.is_drop and e.packet.hops >= len(self.hops):
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
            list: new packets having random sources and destinations.
        """
        packets = []
        nodes_id = list(self.nodes.keys())
        for _ in range(np.random.poisson(lambd)):
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
