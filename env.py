import random
import logging
from collections import OrderedDict

from config import *


class Packet:
    def __init__(self, ID, source, dest, birth):
        self.ID = ID
        self.source = source
        self.dest = dest
        self.birth = birth
        self.hops = 0
        self.queue_time = 0

    def __repr__(self):
        return "Packet<{}|{}->{}>".format(self.ID, self.source, self.dest)


class Event:
    """ Event records the packet passing through connection
        arrive_time is when the packet would really arrive to_node
    """
    def __init__(self, from_node, to_node, packet_ID, arrive_time):
        self.from_node = from_node
        self.to_node = to_node
        self.packet_ID = packet_ID
        self.arrive_time = arrive_time

    def __lt__(self, other):
        return self.arrive_time < other.arrive_time

    def __repr__(self):
        return "Event<{}->{} Packet<{}> at {}>".format(self.from_node, self.to_node, self.packet_ID, self.arrive_time)


class Reward:
    """ Reward is what Network.step() return.
        Reward contains 
            source, dest = where the packet from/destination
            action = which neibor the last node chose
            score = the receiving node's evaluation score: min_{a} Q(S_t, a)
            queue/trans_time = the time cost on queue/transmission on the last node/connection
    """
    def __init__(self, source, dest, action, score, queue_time, trans_time):
        self.source = source
        self.dest = dest
        self.action = action
        self.score = score
        self.queue_time = queue_time
        self.trans_time = trans_time

    def __repr__(self):
        return "Reward<{}->{} by {}|score: {}, queue: {}, trans: {}>".format(
            self.source, self.dest, self.action, self.score, self.queue_time, self.trans_time)


class Node:
    """ Node can receive and send Packets.
        Node.queue is where Packets waiting for delivered,
        Node.sent[neibor] is where Packets already sent to 'neibor' but not arrived.
    """
    def __init__(self, ID, clock):
        self.ID = ID
        self.clock = clock
        self.queue = []
        self.sent = {}

        self.timeout_failure = 0
        self.total_packets = 0
        self.total_hops = 0
        self.total_route_time = 0

    def link(self, neibor):
        self.sent[neibor] = []

    def __repr__(self):
        return "Node<{}, queue: {}, sent: {}>".format(self.ID, self.queue, self.sent)

    def receive(self, packet):
        """ receive packet and determine where the packet should be delivered. """
        logging.debug("{}: node {} receives packet {}".format(self.clock, self.ID, packet))
        if packet.dest == self.ID: # when the packet arrives its destination
            logging.debug("{}: packet {} reach destination node {}".format(self.clock, packet, self.ID))
            self.total_packets += 1
            self.total_hops += packet.hops
            self.total_route_time += self.clock - packet.birth
            return 0

        packet.start_queue = self.clock
        choice, score = self.agent.choose(self.ID, packet.dest)
        packet.choice = choice
        self.queue.append(packet)
        return score

    def send(self):
        """ send packet to its chosen neibor
            return an Event if a packet has been sent, None if not
        """
        i = 0
        while i < len(self.queue):
            p = self.queue[i]
            if len(self.sent[p.choice]) < BandwidthLimit:
                logging.debug("{}: node {} send packet {} to {}".format(self.clock, self.ID, p, p.choice))
                self.queue.pop(i)
                p.queue_time = self.clock - p.start_queue
                p.trans_time = TransTime # set the transmission delay
                event = Event(self.ID, p.choice, self.clock+p.trans_time)
                p.event = event
                self.sent[p.choice].append(p)
                return event
            else:
                i += 1


class Network:
    """ Network simulates packtes routing between connected nodes.
    """
    def __init__(self, file):
        self.clock = 0
        self.nodes = OrderedDict()
        self.links = OrderedDict()
        self.event_queue = []
        self.next_packet_time = 0
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

    def step(self, lambd=Lambda):
        """ step moves the whole network until one packet arrived next node.
            return a Reward
        """
        if self.next_packet_time == 0:
            p, self.next_packet_time = self.new_packet(lambd)
            self.inject(p)

        self.drop_timeout()
        next_event = self.next_event()
        next_step = next_event.arrive_time - self.clock
        if next_step > self.next_packet_time:
            self.clock += self.next_packet_time
            self.broadcast()
            p, self.next_packet_time = self.new_packet(lambd)
            self.inject(p)
            return self.step(lambd)
        else: # one of sending packets really arrives
            self.clock += next_step
            self.broadcast()
            self.event_queue.remove(next_event)
            self.next_packet_time -= next_step
            from_node, to_node = next_event.from_node, next_event.to_node
            try:
                p = self.nodes[from_node].sent[to_node].pop(0)
            except Exception as err:
                print(next_event)
                raise err
            self.next_packet_time -= next_step
            score = self.nodes[to_node].receive(p)
            # try to send
            event = self.nodes[to_node].send()
            if event is not None:
                self.event_queue.append(event)
            return Reward(from_node, p.dest, to_node, score, p.queue_time, p.trans_time)

    def new_packet(self, lambd):
        """ return a new packet having random source and destination, the time to send the packet
            following exponential distribution by given lambd
        """
        nodes_id = list(self.nodes.keys())
        source = random.choice(nodes_id)
        dest = random.choice(nodes_id)
        while dest == source:
            dest = random.choice(nodes_id)
        p = Packet(source, dest, self.clock)
        next_step = random.expovariate(lambd)
        return p, next_step

    def inject(self, packet):
        score = self.nodes[packet.source].receive(packet)
        event = self.nodes[packet.source].send()
        if isinstance(event, Event):
            self.event_queue.append(event)

    def broadcast(self):
        """ broadcast network's clock to every nodes """
        for node in self.nodes.values():
            node.clock = self.clock
    
    def drop_timeout(self):
        """ drop_timeout would compare the living time of packets in network with config.Timeout,
            drop the timeouted packets.
        """
        for node in self.nodes.values():
            i = 0
            while i < len(node.queue):
                if node.clock - node.queue[i].birth > Timeout:
                    node.timeout_failure += 1
                    node.queue.pop(i)
                else:
                    i += 1
            for sent in node.sent.values():
                j = 0
                while j < len(sent):
                    if node.clock - sent[j].birth > Timeout:
                        node.timeout_failure += 1
                        self.event_queue.remove(sent[j].event) # remove corresponding events from event_queue
                        sent.pop(j)
                    else:
                        j += 1

    def next_event(self):
        assert len(self.event_queue) > 0, "no event"
        event = self.event_queue[0]
        for e in self.event_queue:
            if e < event:
                event = e
        return event

    @property
    def hops(self):
        return {k: self.nodes[k].total_hops for k in self.nodes}

    @property
    def packets(self):
        return {k: self.nodes[k].total_packets for k in self.nodes}

    @property
    def route_time(self):
        return {k: self.nodes[k].total_route_time for k in self.nodes}

    @property
    def timeout_failure(self):
        return {k: self.nodes[k].timeout_failure for k in self.nodes}


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
