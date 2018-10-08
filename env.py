import random
import logging

from config import *


class Packet:
    def __init__(self, source, dest, birth):
        self.source = source
        self.dest = dest
        self.birth = birth
        self.hops = 0
        self.queue_time = 0

    def __repr__(self):
        return "Packet<{}->{}>".format(self.source, self.dest)


class Event:
    """ Event records the packet passing through connection
        arrive_time is when the packet would really arrive to_node
    """
    def __init__(self, from_node, to_node, arrive_time):
        self.from_node = from_node
        self.to_node = to_node
        self.arrive_time = arrive_time

    def __lt__(self, other):
        return self.arrive_time < other.arrive_time

    def __repr__(self):
        return "Event<{}->{} at {}>".format(self.from_node, self.to_node, self.arrive_time)


class Reward:
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
        """ receive packet and determine where the packet should be delivered.
        """
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
                self.sent[p.choice].append(p)
                return Event(self.ID, p.choice, self.clock+p.trans_time)
            else:
                i += 1


class Network:

    def __init__(self, file):
        self.clock = 0
        self.nodes = {}
        self.links = {}
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
        """ step would compare the time to next Event end with the time to next packet to inject
            step calls itself recursively if new packet needs to inject before an Event ends.
            return a Reward
        """
        if self.next_packet_time == 0:
            p, self.next_packet_time = self.new_packet(lambd)
            self.inject(p)

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
            print(next_event)
            print(self.nodes[from_node])
            p = self.nodes[from_node].sent[to_node].pop(0)
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
        """ broadcast network's clock to every nodes
            drop the timeout packets
        """
        for node in self.nodes.values():
            node.clock = self.clock
            i = 0
            while i < len(node.queue):
                if node.clock - node.queue[i].birth > Timeout:
                    node.timeout_failure += 1
                    node.queue.pop(i)
                else:
                    i += 1
            for sent in node.sent.values():
                i = 0
                while i < len(sent):
                    if node.clock - sent[i].birth > Timeout:
                        node.timeout_failure += 1
                        sent.pop(i)
                    else:
                        i += 1

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
