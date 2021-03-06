import numpy as np
import logging
from collections import OrderedDict
from heapq import *

from base_policy import Policy


class Packet:
    """
    Args:
        source, dest (int): The start/end node's ID.
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
        packet      (Packet): Which packet is being delivered.
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
        source (int): Where the packet sent from at the last step, NOT where the packet begins.
        dest   (int): Where the packet finally ends (destination).
        action (int): the action taken by the `source` node (which neighbor to send to).
        packet (Packet): the corresponding packet.
        agent_info (Dict): Extra information from agent.get_info
    """
    def __init__(self, source, packet, action, agent_info={}):
        self.source = source
        self.dest = packet.dest
        self.action = action
        self.packet = packet
        self.agent_info = agent_info

    def __repr__(self):
        return f"Reward<{self.source}->{self.dest} by {self.action}>"


class Node:
    """ Node is the unit in a network.

    Attributes:
        queue (List[Packet]): Where Packets waiting for being delivered.
        sent  (Dict[int, int]): An pseudo stage where Packets already sent but not arrive the next node yet.
            For instance, 'thisNode.sent[9] = 2' means 'the directed connection between `thisNode` and Node 9 has 2 packets under delivery'
    """
    def __init__(self, ID, network):
        self.ID = ID
        self.network = network
        self.reset()

        self.set_mode(None)

    def set_mode(self, mode):
        if mode == 'bp':
            self.send = self._send_bp
            self._build_info = self._build_info_default
        elif mode == 'dual':
            self.send = self._send_default
            self._build_info = self._build_info_dual
        else:
            self.send = self._send_default
            self._build_info = self._build_info_default

    def reset(self):
        self.queue = []  # Priority Queue
        self.sent = dict.fromkeys(self.links, 0)

    @property
    def clock(self):
        return self.network.clock

    @property
    def agent(self):
        return self.network._agent

    @property
    def links(self):
        return self.network.links[self.ID]

    def is_avaliable(self, action):
        return self.sent[action] < self.network.bandwidth

    def __repr__(self):
        return f"Node<{self.ID}, queue: {self.queue}, sent: {self.sent}>"

    def receive(self, packet):
        """ receive a packet """
        logging.debug(f"{self.clock}: {self.ID} receives {packet}")
        if self.ID == packet.dest:  # the packet ends here
            logging.debug(f"{self.clock}: {packet} ends in {self.ID}")
            self.network.end_packet(packet)
        else:
            # enter queue and wait for being deliveried.
            packet.start_queue = self.clock
            self.queue.append(packet)
            self.agent.receive(
                self.ID, packet.dest
            )  # for some algorithms need to know a packet received.

    def _send_packet(self, p, action):
        """
        a sub-function for self.send().
        send Packet `p` to Node `action`(int)
        """
        logging.debug(f"{self.clock}: {self.ID} sends {p} to {action}")
        p.hops += 1
        self.sent[action] += 1
        p.trans_time = self.network.transtime  # set the transmission delay
        heappush(self.network.event_queue,
                 Event(p, self.ID, action, self.clock + p.trans_time))

    def _build_info_default(self, agent_info, packet, action):
        # set the environment rewards
        # q: queuing delay; t: transmission delay
        agent_info['q_y'] = self.clock - packet.start_queue
        agent_info['t_y'] = 1
        return agent_info

    def _build_info_dual(self, agent_info, packet, action):
        # dual mode
        agent_info['q_y'] = max(1, len(self.network.nodes[action].queue))
        agent_info['t_y'] = 0
        agent_info['q_x'] = max(1, len(self.queue))
        agent_info['t_x'] = 0
        return agent_info

    def _send_default(self):
        """ Send a packet in queue order.
        agent.choose determines the action/next node

        Returns:
            List[Reward]
        """
        i = 0
        rewards = []
        avaliable_path = np.array(  # some condition to check path avaliable
            [self.is_avaliable(i) for i in self.links],
            dtype=bool)
        while i < len(self.queue) and any(avaliable_path):
            dest = self.queue[i].dest
            # if the connection to chosen `action` is full, skip the packet and send the next packet in queue
            action = self.agent.choose(self.ID, dest)
            if action is not None and avaliable_path[self.agent.action_idx[self.ID][action]]:
                p = self.queue.pop(i)
                self._send_packet(p, action)
                self.agent.send(self.ID, dest)
                avaliable_path[self.agent.action_idx[self.ID][action]] = self.is_avaliable(action)
                # then build Reward
                agent_info = self.agent.get_info(self.ID, action, p)
                agent_info = self._build_info(agent_info, p, action)
                rewards.append(Reward(self.ID, p, action, agent_info))
                # Remove/comment the next line if Multiple Packages Need to be Sent at once
                return rewards # only one packet can be sent
            else:
                i += 1
        return rewards

    def _send_bp(self):
        i = 0
        rewards = []
        avaliable_path = [n for n in self.links if self.is_avaliable(n)]
        while len(avaliable_path) > 0 and len(self.queue) > 0:
            dests = self.agent.choose(self.ID, avaliable_path)
            if not any(dests):
                break
            for dest, action in zip(dests, avaliable_path):
                if dest is None:
                    continue
                p = next(p for p in self.queue if p.dest == dest)
                self.queue.remove(p)
                self._send_packet(p, action)
                self.agent.send(self.ID, dest)
                rewards.append(Reward(self.ID, p, action, {}))
            avaliable_path = [n for n in self.links if self.is_avaliable(n)]
        return rewards


class Network:
    """ Network simulates packet routing between connected nodes.

    Args:
        file (string): The name of network file.
        bandwidth (int): the bandwidth limitation of a connection/the maximum number of transmitting packets simultaneously
        transtime (int, float): the time delay of transmitting a packet to next node
        is_drop (bool): whether the network drop packet on some condition (the packet hops overpass number of all nodes)

    Attributes:
        clock (int): The simulation time.
        nodes (Dict[Int, Node]): An ordered dictionary of all nodes in this network.
        links (Dict[Int, List[Int]]): lists of connected nodes' ID.
        agent (Policy): bind an agent, which follows class `Policy`
        mode (string): Network mode,
            None -> Default mode, 'dual' -> Duality, 'bp' -> BackPressure
        event_queue (heap): A queue of following happen events.
        all_packets (int): The total number of packets in this simulation.
        end_packets (int): The packets already ends in its destination.
        drop_packets (int): The number of dropped packets
        active_packets (int): The number of active packets
        hops (int): The number of total hops of all packets
        route_time (int): The total routing time of all ended packets.
    """
    def __init__(self, file, bandwidth=1, transtime=1, is_drop=False):
        self.bandwidth = bandwidth
        self.transtime = transtime
        self.nodes = OrderedDict()
        self.links = OrderedDict()
        self.is_drop = is_drop
        self.sample, self._sample_idx = [], 0

        self.read_network(file)
        for i in self.links.keys():
            self.nodes[i] = Node(i, self)

        self._agent = Policy(self)
        self.reset()

    @property
    def agent(self):
        return self._agent

    @property
    def mode(self):
        return self._agent.mode

    @agent.setter
    def agent(self, new_agent):
        if self.mode != new_agent.mode:
            for node in self.nodes.values():
                node.set_mode(new_agent.mode)
        self._agent = new_agent

    def reset(self):
        """ reset the network attributes """
        self.clock = 0
        self.event_queue = []
        self.all_packets = 0
        self.end_packets = 0
        self.drop_packets = 0
        self.active_packets = 0
        self.hops = 0
        self.route_time = 0
        for node in self.nodes.values():
            node.reset()
        self.agent.clean()  # tell agent the Network resetted
        # NOT reset the agent

    def read_network(self, file):
        " read_network constructs the Network.links "
        self.proj = {}  # project from file identity to node ID
        with open(file, 'r') as f:
            lines = [l.split() for l in f.readlines()]
        ID = 0
        for l in lines:
            if l[0] == "1000":  # declare a node
                self.proj[l[1]] = ID
                self.links[ID] = []
                ID += 1  # increase ID
            elif l[0] == "2000":  # delcare a connection
                src, dst = self.proj[l[1]], self.proj[l[2]]
                self.links[src].append(dst)
                self.links[dst].append(src)

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
            packets.append(Packet(source, dest, self.clock))
        return packets

    def inject(self, packets):
        """ Injects the packets into network """
        self.all_packets += len(packets)
        self.active_packets += len(packets)
        for packet in packets:
            self.nodes[packet.source].receive(packet)

    def end_packet(self, packet):
        """ a packet ends at this node """
        self.active_packets -= 1
        self.end_packets += 1
        if len(self.sample) > 0:
            self.sample[min(self._sample_idx,
                            len(self.sample) - 1)] = self.clock - packet.birth
            self._sample_idx += 1
        self.route_time += self.clock - packet.birth
        self.hops += packet.hops
        del packet

    def step(self, duration):
        """ step runs the network forward `duration`
        one sending and one agent learning

        Args:
            duration (int, duration): The duration of one step.

        Returns:
            List[Reward]: A list of rewards from sending events happended in the timeslot.
        """
        rewards = []
        for node in self.nodes.values():
            r = node.send()  # r: List[Reward]
            if r:  # len(r) > 0
                rewards += r

        end_time = self.clock + duration
        next_event = nsmallest(1, self.event_queue)
        while len(next_event) > 0 and next_event[0].arrive_time <= end_time:
            e = heappop(self.event_queue)
            self.nodes[e.from_node].sent[e.to_node] -= 1
            next_event = nsmallest(1, self.event_queue)
            if self.is_drop and e.packet.hops >= len(self.nodes):
                # drop the packet if too many hops
                self.drop_packets += 1
                self.active_packets -= 1
                self.agent.drop_penalty(e)
                continue
            self.clock = e.arrive_time
            self.nodes[e.to_node].receive(e.packet)

        self.clock = end_time
        return rewards

    def train(self,
              duration,
              lambd,
              slot=1,
              freq=1,
              lr={},
              droprate=False,
              hop=False):
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
            hop (bool): whether return hop times or not

        Returns:
            Result (Dict[Str, List[Real]]):
                route_time (List[Real]): the vector of routing time in this training duration.
                drop_rate (List[Real]): the vector of packet-drop rate in this train.
        """
        step_num = int(duration / slot)
        result = {'route_time': np.zeros(step_num)}
        if droprate:
            result['droprate'] = np.zeros(step_num)
        if hop:
            result['hop'] = np.zeros(step_num)
        for i in range(step_num):
            self.inject(self.new_packet(lambd * slot))
            for _ in range(freq):
                r = self.step(slot)
                if r is not None:
                    if lr:
                        self.agent.learn(r, lr=lr)
                    else:
                        self.agent.learn(r)
            result['route_time'][i] = self.ave_route_time
            if droprate:
                result['droprate'][i] = self.drop_rate
            if hop:
                result['hop'][i] = self.ave_hops
        return result

    def sample_route_time(self, size, lambd, slot=1, freq=1, lr={}):
        """
        sample_route_time is an alternative for `train`.
        `train` returns various information (in dictionary) in a specific **time duration**.
        Due to different "load setting", the numbers of arrived packages in a same duration can be different.
        `sample_route_time` runs the network as `train` does, however stops when the number of arrived packages reaches `size`.
        Or say, it returns a `size`-long array, which records 'routing_time' of arrived packages.
        """
        self.sample = np.zeros(size)
        self._sample_idx = 0
        while self._sample_idx < size:
            self.inject(self.new_packet(lambd * slot))
            for _ in range(freq):
                r = self.step(slot)
                if r is None:
                    continue
                if lr:
                    self.agent.learn(r, lr=lr)
                else:
                    self.agent.learn(r)
        sample = self.sample
        self.sample = []
        return sample

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
        print("┌─────┐     " * 6)
        for j in range(5):
            print("│No.{:2d}│".format(6 * i + j), end="")
            if 6 * i + j + 1 in network.links[6 * i + j]:
                print(" {:2d}├ ".format(network.nodes[6 * i + j].sent[6 * i +
                                                                      j + 1]),
                      end="")
            else:
                print(" " * 5, end="")
        print("│No.{:2d}│".format(6 * i + 5))
        for j in range(5):
            print("│{:5d}│".format(len(network.nodes[6 * i + j].queue)),
                  end="")
            if 6 * i + j in network.links[6 * i + j + 1]:
                print(" ┤{:<2d} ".format(network.nodes[6 * i + j +
                                                       1].sent[6 * i + j]),
                      end="")
            else:
                print(" " * 5, end="")
        print("│{:5d}│".format(len(network.nodes[6 * i + 5].queue)))
        print("└─────┘     " * 6)
        if i == 5:
            print("=" * 6)
            break
        for j in range(6):
            if 6 * i + j in network.links[6 * i + j + 6]:
                print("{:2d}┴ ┬{:<2d}     ".format(
                    network.nodes[6 * i + j + 6].sent[6 * i + j],
                    network.nodes[6 * i + j].sent[6 * i + j + 6]),
                      end="")
            else:
                print(" " * 12, end="")
        print()
