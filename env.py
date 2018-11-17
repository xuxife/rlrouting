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
        self.source  = source
        self.dest    = dest
        self.current = source
        self.birth   = birth

        self.event       = None
        self.start_queue = 0
        self.queue_time  = 0
        self.trans_time  = 0

        self.hops       = 0
        self.route_time = 0

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
        self.packet      = packet
        self.from_node   = from_node
        self.to_node     = to_node
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
        self.source     = source
        self.dest       = packet.dest
        self.action     = choice
        self.queue_time = packet.queue_time
        self.trans_time = packet.trans_time
        self.agent_info = agent_info # extra information defined by agents

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
    def __init__(self, ID, clock):
        self.ID    = ID
        self.clock = clock
        self.queue = []
        self.sent  = {}

        self.end_packets  = []

    def link(self, neibor):
        self.sent[neibor] = []

    def __repr__(self):
        return "Node<{}, queue: {}, sent: {}>".format(self.ID, self.queue, self.sent)

    def receive(self, packet):
        """ Receive a packet.
        Update statistic attritubes if this packet ends here, else append the packet into queue.
        """
        logging.debug("{}: Node {} receives {}".format(self.clock, self.ID, packet))
        packet.current = self.ID
        if packet.dest == self.ID: # when the packet arrives its destination
            logging.debug("{}: {} reachs destination Node {}".format(self.clock, packet, self.ID))
            packet.route_time = self.clock - packet.birth
            self.end_packets.append(packet)
        else:
            packet.start_queue = self.clock
            self.queue.append(packet)

    def send(self):
        """ Send a packet ordered by queue.
        Call agent.choose to determine the next node for the packet, 
        then check whether the connection is avaliable.

        Returns:
            Event : Records the information of this delivery.
            Reward: Reward of this action.
            None if no packet is sent.
        """
        for p in self.queue:
            choice = self.agent.choose(self.ID, p.dest)
            if len(self.sent[choice]) < BandwidthLimit:
                logging.debug("{}: Node {} sends {} to {}".format(self.clock, self.ID, p, choice))
                self.queue.remove(p)
                p.queue_time  = self.clock - p.start_queue
                p.trans_time  = TransTime # set the transmission delay
                p.event       = Event(p, self.ID, choice, self.clock+p.trans_time)
                p.hops       += 1
                self.sent[choice].append(p)
                return p.event, Reward(self.ID, p, choice, self.agent.get_reward(self.ID, p.dest, choice))
        return None, None


class Network:
    """ Network simulates packtes routing between connected nodes. 

    Args:
        file (string): The name of network file.

    Attributes:
        clock (int, time): The simulation time.
        nodes (dict): Keys are nodes' ID, values are nodes.
        links (dict): Keys are nodes' ID, values are lists of connected nodes' ID.

        event_queue (list): A queue of following happen events.
        rewards     (list): A list of rewards after call step function.

        active_packets (list): The packets are still active in the network.
        end_packets    (list): The packets already ends in its destination.
    """
    def __init__(self, file):
        self.clock       = 0
        self.nodes       = OrderedDict()
        self.links       = OrderedDict()
        self.event_queue = []
        self.rewards     = []

        self.active_packets = []
        self.end_packets    = []

        self.read_network(file)

    def read_network(self, file):
        with open(file, 'r') as f:
            lines = [l.split() for l in f.readlines()]
        for l in lines:
            if l[0] == "1000":
                ID = int(l[1])
                self.nodes[ID] = Node(ID, self.clock)
                self.links[ID] = []
            if l[0] == "2000":
                source, dest = int(l[1]), int(l[2])
                self.nodes[source].link(dest)
                self.nodes[dest].link(source)
                self.links[source].append(dest)
                self.links[dest].append(source)

    def step(self, duration=TimeSlot, lambd=Lambda):
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

        self.broadcast()
        for node in self.nodes.values():
            event, reward = node.send()
            if event is not None and reward is not None:
                self.event_queue.append(event)
                self.rewards.append(reward)

        while duration >= 0:
            next_event = self.next_event()
            if next_event is None or duration < next_event.arrive_time - self.clock:
                self.clock += duration
                self.broadcast()
                rewards = self.rewards
                self.rewards = []
                self.check_end()
                return rewards
            else:
                next_step   = next_event.arrive_time - self.clock
                duration   -= next_step
                self.clock += next_step
                self.broadcast()
                self.event_queue.remove(next_event)
                p = next_event.packet
                self.nodes[next_event.from_node].sent[next_event.to_node].remove(p)
                self.nodes[next_event.to_node].receive(p)

    def new_packet(self, lambd):
        """ Generates new packets following Poisson(lambd).

        Args:
            lambd (int, float): The Poisson distribution parameter.
        Returns:
            list: A list of new packets having random sources and destinations..
        """
        size     = np.random.poisson(lambd)
        packets  = []
        nodes_id = list(self.nodes.keys())
        for _ in range(size):
            source = np.random.choice(nodes_id)
            dest   = np.random.choice(nodes_id)
            while dest == source:
                dest = np.random.choice(nodes_id)
            packets.append(Packet(source, dest, self.clock))
        return packets

    def inject(self, packet):
        """ Injects the packet into network """
        self.active_packets.append(packet)
        self.nodes[packet.source].receive(packet)

    def check_end(self):
        """ Scan the ended packets in all nodes, remove them from active nodes. """
        self.end_packets = []
        for node in self.nodes.values():
            self.end_packets += node.end_packets
        for p in self.end_packets:
            if p in self.active_packets:
                self.active_packets.remove(p)

    def broadcast(self):
        """ Broadcast the network clock to nodes """
        for node in self.nodes.values():
            node.clock = self.clock
    
    def next_event(self):
        """ Returns:
            Event: The closest event would happen.
        """
        if len(self.event_queue) > 0:
            event = self.event_queue[0]
            for e in self.event_queue:
                if e.arrive_time < event.arrive_time:
                    event = e
            return event
        return None

    def ave_hops(self, method="end"):
        """ ave_hops return the average hops of packets, 
            Three methods are provided:

                "end"   : return the average hops of packets already ended at its destination.
                "active": return the average hops of active packets in the network.
                "all"   : return the average hops of all packets (active+end).
        """
        if method == "end":
            if len(self.end_packets) > 0:
                return np.mean([packet.hops for packet in self.end_packets])
            return 0
        elif method == "active":
            if len(self.active_packets) > 0:
                return np.mean([packet.hops for packet in self.active_packets])
            return 0
        elif method == "all":
            if len(self.active_packets) > 0:
                return np.mean([packet.hops for packet in self.active_packets+self.end_packets])
            return 0


    def ave_route_time(self):
        if len(self.end_packets) > 0:
            return np.mean([packet.route_time for packet in self.end_packets])
        return 0


def print6x6(network):
    print("Time: {}".format(network.clock))
    for i in range(6):
        print("┌─────┐     "*6)
        for j in range(5):
            print("│No.{:2d}│".format(6*i+j), end="")
            if 6*i+j+1 in network.links[6*i+j]:
                print(" {:2d}├ ".format(len(network.nodes[6*i+j].sent[6*i+j+1])), end="")
            else:
                print(" "*5, end="")
        print("│No.{:2d}│".format(6*i+5))
        for j in range(5):
            print("│{:5d}│".format(len(network.nodes[6*i+j].queue)), end="")
            if 6*i+j in network.links[6*i+j+1]:
                print(" ┤{:<2d} ".format(len(network.nodes[6*i+j+1].sent[6*i+j])), end="")
            else:
                print(" "*5, end="")
        print("│{:5d}│".format(len(network.nodes[6*i+5].queue)))
        print("└─────┘     "*6)
        if i == 5:
            print("="*6)
            break
        for j in range(6):
            if 6*i+j in network.links[6*i+j+6]:
                print("{:2d}┴ ┬{:<2d}     ".format(len(network.nodes[6*i+j+6].sent[6*i+j]), len(network.nodes[6*i+j].sent[6*i+j+6])), end="")
            else:
                print(" "*12, end="")
        print()
