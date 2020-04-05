import numpy as np
import logging
from collections import OrderedDict
from heapq import *

from base_policy import Policy


class Packet:
    """
    Args:
        source, dest (int): The start/destination node's ID.
        birth (int): When the packet is generated.

    Attributes:
        trans_time (int): Transmission time.
        hops (int): The number of hops.
    """

    def __init__(self, source, dest, birth):
        self.source = source
        self.dest = dest
        self.birth = birth
        self.hops = 0
        self.trans_time = 0

    def __repr__(self):
        return f"Packet<{self.source}->{self.dest}>"


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
        return f"Event<{self.from_node}->{self.to_node} at {self.arrive_time}>"

    def __lt__(self, other):
        return self.arrive_time < other.arrive_time


class Reward:
    """ Reward defines the backward reward from environment (what Network.step returns)

    Attributes:
        source, dest (int): Where the packet from/destination
        packet (Packet): which packet generates the reward
        action (int): Which neighbor the `source` chose
        agent_info (:obj:): Extra information from agent.get_info
    """

    def __init__(self, source, packet, action, agent_info={}):
        self.source = source
        self.dest = packet.dest
        self.action = action
        self.packet = packet
        self.agent_info = agent_info

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

    def end(self, packet):
        """ a packet ends at this node """
        logging.debug(f"{self.clock}: {packet} ends in {self.ID}")
        self.network.active_packets -= 1
        self.network.end_packets += 1
        self.network.route_time += self.clock.t - packet.birth
        self.network.hops += packet.hops
        del packet

    def receive(self, packet):
        """ receive a packet """
        logging.debug(f"{self.clock}: {self.ID} receives {packet}")
        if self.ID == packet.dest:
            self.end(packet)
        else:
            packet.start_queue = self.clock.t
            self.queue.append(packet)
            self.agent.receive(self.ID, packet.dest)

    def _send_packet(self, p, action):
        " send `packet` to `action` "
        logging.debug(f"{self.clock}: {self.ID} sends {p} to {action}")
        p.hops += 1
        self.sent[action] += 1
        p.trans_time = self.network.transtime  # set the transmission delay
        heappush(self.network.event_queue,
                 Event(p, self.ID, action, self.clock.t+p.trans_time))

    def send(self):
        """ Send a packet in queue order.
        agent.choose determines the action/next node

        Returns:
            Reward: Reward of this action.
            None if no action is taken
        """
        i = 0
        while i < len(self.queue):
            dest = self.queue[i].dest
            action = self.agent.choose(self.ID, dest)
            # if the connection to chosen `action` is full, skip the packet and send the next packet in queue
            if self.sent[action] < self.network.bandwidth:
                p = self.queue.pop(i)
                self._send_packet(p, action)
                self.agent.send(self.ID, dest)
                # then build Reward
                agent_info = self.agent.get_info(self.ID, action, p)
                # set the environment rewards
                # q: queuing delay; t: transmission delay
                # agent_info['q_y'] = max(
                #     1, len(self.network.nodes[action].queue))
                agent_info['q_y'] = self.clock.t - p.start_queue
                agent_info['t_y'] = 1
                if self.mode == 'dual':
                    agent_info['q_x'] = max(1, len(self.queue))
                    agent_info['t_x'] = 0
                return Reward(self.ID, p, action, agent_info)
            else:
                i += 1
        return None


class Network:
    """ Network simulates packet routing between connected nodes.

    Args:
        file (string): The name of network file.
        bandwidth (int): the bandwidth limitation of a connection/the maximum number of transmitting packets simultaneously
        transtime (int, float): the time delay of transmitting a packet to next node
        is_drop (bool): whether the network drop packet on some condition (the packet hops overpass number of all nodes)

    Attributes:
        clock (Clock): The simulation time.
        nodes (Dict[Int, Node]): An ordered dictionary of all nodes in this network.
        links (Dict[Int, List[Int]]): lists of connected nodes' ID.
        agent (:obj:): bind an agent, which follows class `Policy`
        mode (string): Network mode,
            None -> Default mode, 'dual' -> Duality mode
        event_queue (List[Event]): A queue of following happen events.
        all_packets (int): The total number of packets in this simulation.
        end_packets (int): The packets already ends in its destination.
        drop_packets (int): The number of dropped packets
        active_packets (int): The number of active packets
        hops (int): The number of total hops of all packets
        route_time (int): The total routing time of all ended packets.
    """

    def __init__(self, file, bandwidth=3, transtime=1, is_drop=False):
        self.bandwidth = bandwidth
        self.transtime = transtime
        self.nodes = OrderedDict()
        self.links = OrderedDict()
        self.agent = Policy(self)
        self.is_drop = is_drop
        self.clock = Clock(0)

        self.read_network(file)

        self.clean()

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

    def read_network(self, file):
        self.projection = {}  # project from file identity to node ID
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

    def new_packet(self, lambd):
        """ Generates new packets following Poisson(lambd).

        Args:
            lambd (int, float): The Poisson distribution parameter.
        Returns:
            list: new packets having random sources and destinations.
        """
        packets = []
        nodes_num = len(self.nodes)
        for _ in range(np.random.poisson(lambd)):
            source, dest = np.random.randint(0, nodes_num, size=2)
            while dest == source:  # assert: source != dest
                dest = np.random.randint(0, nodes_num)
            packets.append(Packet(source, dest, self.clock.t))
        return packets

    def inject(self, packets):
        """ Injects the packets into network """
        self.all_packets += len(packets)
        self.active_packets += len(packets)
        for packet in packets:
            self.nodes[packet.source].receive(packet)

    def step(self, duration, penalty=0):
        """ step runs the network forward `duration`
        one sending and one agent learning

        Args:
            duration (int, duration): The duration of one step.
            penalty  (int, float)   : The penalty of dropping a packet

        Returns:
            List[Reward]: A list of rewards from sending events happended in the timeslot.
        """
        rewards = list(filter(None, [node.send()
                                     for node in self.nodes.values()]))

        end_time = self.clock.t + duration
        next_event = nsmallest(1, self.event_queue)
        while len(next_event) > 0 and next_event[0].arrive_time <= end_time:
            e = heappop(self.event_queue)
            self.nodes[e.from_node].sent[e.to_node] -= 1
            next_event = nsmallest(1, self.event_queue)
            if self.is_drop and e.packet.hops >= len(self.nodes):
                # drop the packet if too many hops
                self.drop_packets += 1
                self.active_packets -= 1
                self.agent.drop_penalty(e, penalty=penalty)
                continue
            self.clock.t = e.arrive_time
            self.nodes[e.to_node].receive(e.packet)

        self.clock.t = end_time
        return rewards

    def train(self, duration, lambd, slot=1, freq=1, lr={}, penalty=0, droprate=False):
        """ train process the whole network forward
        new packet arriving at `lambd` (s^-1) rate
        call `step` to send and learn from rewards

        Args:
            duration (second) : the length of running period
            lambd (second^(-1)) : the Poisson parameter
            slot (second) : the length of one time slot (one new arriving)
            freq (int, second^(-1)) : the frequency of sending & learning (calliing `step`) in one time slot
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
            self.inject(self.new_packet(lambd*slot))
            for _ in range(freq):
                r = self.step(slot, penalty=penalty)
                if r is not None:
                    if lr:
                        self.agent.learn(r, lr=lr)
                    else:
                        self.agent.learn(r)
            route_time[i] = self.ave_route_time
            if droprate:
                drop_rate[i] = self.drop_rate
        return (route_time, drop_rate) if droprate else route_time

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
